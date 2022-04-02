import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import pickle
import dgl
from dgl.utils import expand_as_pair

sys.path.append("../")
import utils
import math
import numpy as np
from scipy import linalg as la
import copy

from model.graphsage import GraphSAGE
from model.embedder import EmbeddingGenerator
from torch.nn.parameter import Parameter
import tqdm


class GConvLayers(nn.Module):
    """
    Multi-layer GCN/GAT/Multi-head GAT/GraphSAGE/Gated GAT
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_gcn_layers,
        g_conv="graphsage",
        activation_fn="relu",
        dropout=0.0,
        device=None,
        is_classifier=False,
        **kwargs
    ):
        super(GConvLayers, self).__init__()

        if g_conv not in ["gcn", "gat", "multihead_gat", "graphsage", "gaan", "gin"]:
            raise NotImplementedError

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_gcn_layers = num_gcn_layers
        self.g_conv = g_conv
        self.activation_fn = activation_fn
        self.device = device
        if self.activation_fn == "elu":
            self.activation = F.elu
        else:
            self.activation = F.relu
        self.dropout = nn.Dropout(p=dropout)
        self.is_classifier = is_classifier

        self.layers = nn.ModuleList()
        graphsage = GraphSAGE(
            in_feats=in_dim,
            n_hidden=hidden_dim,
            n_layers=num_gcn_layers,
            activation=None,
            dropout=dropout,
            aggregator_type=kwargs["aggregator_type"],
        )
        self.layers = graphsage.layers

        # optionally for non-temporal models
        if self.is_classifier:
            self.fc = nn.Linear(hidden_dim, kwargs["num_classes"])

    def forward(self, g, inputs):
        """
        Args:
            inputs: shape (batch, in_dim)
        Returns:
            h: shape (batch, hidden_dim) using "mean" aggregate or (batch, hidden_dim*num_heads) using
                "cat" aggregate
        """

        h = inputs

        for i in range(self.num_gcn_layers):
            h = self.layers[i](g, h)
            if i != self.num_gcn_layers - 1:
                h = self.activation(h)
                h = self.dropout(h)

        if self.is_classifier:
            logits = self.fc(h)
            if logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
            return logits, h
        else:
            return h


class GConvGRUCell(nn.Module):
    """
    Graph Convolution Gated Recurrent Unit Cell.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        g_conv="graphsage",
        num_gconv_layers=1,
        activation_fn="relu",
        dropout=0.0,
        device="cpu",
        add_bias=True,
        **kwargs
    ):
        """
        Args:
            input_dim: input feature dim
            hidden_dim: hidden dim
            g_conv: graph convolutional layer, options: 'gat', 'gcn', 'multihead_gat', or 'graphsage'
            num_gconv_layers: number of graph convolutional layers
            activation_fn: activaton function name, 'relu' or 'elu'
            dropout: dropout proba
            device: 'cpu' or 'cuda'
        """
        super(GConvGRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.g_conv = g_conv
        self.num_gconv_layers = num_gconv_layers
        self.activation_fn = activation_fn
        self.dropout = dropout
        self.device = device
        self.add_bias = add_bias

        # gconv_gate includes reset and update gates, that's why hidden_dim * 2
        self.gconv_gate = GConvLayers(
            in_dim=input_dim + hidden_dim,
            hidden_dim=hidden_dim * 2,
            num_gcn_layers=num_gconv_layers,
            g_conv=g_conv,
            activation_fn=activation_fn,
            dropout=dropout,
            device=device,
            **kwargs
        )

        self.gconv_candidate = GConvLayers(
            in_dim=input_dim + hidden_dim,
            hidden_dim=hidden_dim,
            num_gcn_layers=num_gconv_layers,
            g_conv=g_conv,
            activation_fn=activation_fn,
            dropout=dropout,
            device=device,
            **kwargs
        )

        # note that the biases are initialized as zeros
        if add_bias:
            self.gate_bias = nn.Parameter(torch.FloatTensor(size=(hidden_dim * 2,)))
            nn.init.constant_(self.gate_bias.data, val=0)
            self.candidate_bias = nn.Parameter(torch.FloatTensor(size=(hidden_dim,)))
            nn.init.constant_(self.candidate_bias.data, val=0)
        else:
            self.gate_bias = None
            self.candidate_bias = None

    def forward(self, graph, inputs, state):
        """
        Args:
            graph: DGL graph
            inputs: input at current time step, shape (num_nodes, input_dim)
            state: hidden state from previous time step, shape (num_nodes, hidden_dim)
        Returns:
            new_state: udpated hidden state, shape (num_nodes, hidden_dim)
        """
        # reset and update gates
        # graph conv layer input is [inputs, state]
        inputs_state = torch.cat(
            [inputs, state], dim=-1
        )  # (num_nodes, input_dim+hidden_dim)
        h = self.gconv_gate(graph, inputs_state)  # (num_nodes, hidden_dim*2)
        if self.add_bias:
            h = h + self.gate_bias
        h = torch.sigmoid(h)

        # split into reset and update gates, each shape (num_nodes, hidden_dim)
        r, u = torch.split(h, split_size_or_sections=self.hidden_dim, dim=-1)

        # candidate
        c = self.gconv_candidate(
            graph, torch.cat([inputs, r * state], dim=-1)
        )  # (num_nodes, hidden_dim)
        if self.add_bias:
            c = c + self.candidate_bias
        c = torch.tanh(c)

        new_state = u * state + (1 - u) * c

        return new_state


class GraphRNN(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_gcn_layers,
        num_gru_layers,
        g_conv="graphsage",
        n_classes=1,
        dropout=0.0,
        activation_fn="relu",
        device=None,
        is_classifier=True,
        t_model="gru",
        final_pool="last",
        add_bias=True,
        ehr_encoder_name=None,
        cat_idxs=None,
        cat_dims=None,
        cat_emb_dim=None,
        **kwargs
    ):
        super(GraphRNN, self).__init__()

        if g_conv not in ["gcn", "gat", "multihead_gat", "graphsage", "gaan", "gin"]:
            raise NotImplementedError

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_gcn_layers = num_gcn_layers
        self.num_gru_layers = num_gru_layers
        self.g_conv = g_conv
        self.n_classes = n_classes
        self.activation_fn = activation_fn
        self.device = device
        self.is_classifier = is_classifier
        self.t_model = t_model
        self.final_pool = final_pool
        self.add_bias = add_bias
        self.ehr_encoder_name = ehr_encoder_name
        self.cat_emb_dim = cat_emb_dim

        # ehr encoder
        if ehr_encoder_name is not None:
            print("Using embedder to embed ehr data...")
            self.embedder = EmbeddingGenerator(
                input_dim=in_dim,
                cat_dims=cat_dims,
                cat_idxs=cat_idxs,
                cat_emb_dim=cat_emb_dim,
            )
            in_dim = (in_dim - len(cat_idxs)) + len(cat_idxs) * cat_emb_dim
        else:
            self.embedder = None

        # GConvGRU layers
        self.layers = nn.ModuleList()
        if t_model == "gru":
            self.layers.append(
                GConvGRUCell(
                    input_dim=in_dim,
                    hidden_dim=hidden_dim,
                    g_conv=g_conv,
                    num_gconv_layers=num_gcn_layers,
                    activation_fn=activation_fn,
                    dropout=dropout,
                    device=device,
                    add_bias=add_bias,
                    **kwargs
                )
            )

            for i in range(1, num_gru_layers):
                self.layers.append(
                    GConvGRUCell(
                        input_dim=hidden_dim,
                        hidden_dim=hidden_dim,
                        g_conv=g_conv,
                        num_gconv_layers=num_gcn_layers,
                        activation_fn=activation_fn,
                        dropout=dropout,
                        device=device,
                        add_bias=add_bias,
                        **kwargs
                    )
                )
        else:
            raise NotImplementedError

        if is_classifier:
            self.fc = nn.Linear(hidden_dim, n_classes)

        self.dropout = nn.Dropout(p=dropout)
        if self.activation_fn == "elu":
            self.activation = F.elu
        else:
            self.activation = F.relu
        self.kwargs = kwargs

    def forward(self, graph, inputs, init_state=None):
        """
        Args:
            graph: list of graphs, if non-time-varying, len(graph) = 1; otherwise len(graph) = max_seq_len
            inputs: input features, shape (num_nodes, seq_len, input_dim), where batch is number of nodes
            init_state: GRU hidden state, shape (num_nodes, gru_dim).
                If None, will initialize init_state
        """
        num_nodes, max_seq_len, in_dim = inputs.shape

        if self.ehr_encoder_name == "embedder":
            inputs = inputs.reshape(num_nodes * max_seq_len, -1)
            inputs = self.embedder(inputs).reshape(num_nodes, max_seq_len, -1)

        inputs = torch.transpose(
            inputs, dim0=0, dim1=1
        )  # (max_seq_len, num_nodes, in_dim)

        # initialize GRU hidden states
        if init_state is None:
            hidden_state = self.init_hidden(num_nodes)
        else:
            hidden_state = init_state

        # loop over GRU layers
        curr_inputs = inputs
        for idx_gru in range(self.num_gru_layers):
            state = hidden_state[idx_gru, :]

            outputs_inner = []  # inner outputs within GRU layers
            # loop over time
            for t in range(max_seq_len):
                state = self.layers[idx_gru](graph, curr_inputs[t, :, :], state)
                outputs_inner.append(state)

            # input to next GRU layer is the previous GRU layer's last hidden state * dropout
            # https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
            curr_inputs = torch.stack(
                outputs_inner, dim=0
            )  # (seq_len, num_nodes, hidden_dim)
            if idx_gru != (self.num_gru_layers - 1):
                curr_inputs = self.activation(self.dropout(curr_inputs))
        gru_out = curr_inputs  # (seq_len, num_nodes, hidden_dim)

        if self.final_pool == "last":
            # get last relevant time step output
            out = gru_out[-1, :, :]
        elif self.final_pool == "mean":
            out = torch.mean(gru_out, dim=0)
        else:
            out, _ = torch.max(gru_out, dim=0)

        if self.is_classifier:
            logits = self.fc(self.dropout(out))
            return logits, out
        else:
            return out

    def init_hidden(self, batch_size):
        init_states = []
        for _ in range(self.num_gru_layers):
            curr_init = torch.zeros(
                batch_size, self.hidden_dim, requires_grad=False
            ).to(self.device)
            init_states.append(curr_init)
        init_states = torch.stack(init_states, dim=0).to(
            self.device
        )  # (num_gru_layers, num_nodes, gru_dim)
        return init_states
