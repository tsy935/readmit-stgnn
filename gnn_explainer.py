""" 
Adapted from DGL implementation of GNN Explainer: https://github.com/dmlc/dgl/blob/master/examples/pytorch/gnn_explainer
"""

import numpy as np
import dgl
import sys
import os
import torch
import torch.nn as nn
import pickle
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math

sys.path.append("../")
from model.model import GraphRNN
from model.fusion import JointFusionModel
from utils import get_config
import copy
from tqdm import tqdm
from sklearn.metrics import roc_curve
import matplotlib.pylab as plt
import torch as th
import dgl
from dgl.sampling import sample_neighbors
from dotted_dict import DottedDict
import json
import pandas as pd
import argparse
import wandb
from constants import CATEGORICAL_DIMS, CATEGORICAL_IDXS


def extract_subgraph(graph, seed_nodes, hops=2):
    """
    For the explainability, extract the subgraph of a seed node with the hops specified.
    Parameters
    ----------
    graph:      DGLGraph, the full graph to extract from. This time, assume it is a homograph
    seed_nodes:  Tensor, index of a node in the graph
    hops:       Integer, the number of hops to extract
    Returns
    -------
    sub_graph: DGLGraph, a sub graph
    origin_nodes: List, list of node ids in the origin graph, sorted from small to large, whose order is the new id. e.g
               [2, 51, 53, 79] means in the new sug_graph, their new node id is [0,1,2,3], the mapping is 2<>0, 51<>1, 53<>2,
               and 79 <> 3.
    new_seed_node: Scalar, the node index of seed_nodes
    """
    seeds = seed_nodes
    for i in range(hops):
        i_hop = sample_neighbors(graph, seeds, -1)
        seeds = th.cat([seeds, i_hop.edges()[0]])

    ori_src, ori_dst = i_hop.edges()
    edge_all = th.cat([ori_src, ori_dst])
    origin_nodes, new_edges_all = th.unique(edge_all, return_inverse=True)

    n = int(new_edges_all.shape[0] / 2)
    new_src = new_edges_all[:n]
    new_dst = new_edges_all[n:]

    sub_graph = dgl.graph((new_src, new_dst))
    new_seed_node = th.nonzero(origin_nodes == seed_nodes, as_tuple=True)[0][0]

    # also get edge weights if available
    if "weight" in graph.edata.keys():
        sub_graph_edge_id = graph.edge_ids(
            ori_src.type(torch.int32), ori_dst.type(torch.int32)
        )
        sub_graph.edata["weight"] = graph.edata["weight"][sub_graph_edge_id.long()]

    if "label" in graph.ndata.keys():
        sub_graph.ndata["label"] = graph.ndata["label"][origin_nodes.long()]

    return sub_graph, origin_nodes, new_seed_node


class NodeExplainerModule(nn.Module):
    """
    A Pytorch module for explaining a node's prediction based on its computational graph and node features.
    Use two masks: One mask on edges, and another on nodes' features.
    So far due to the limit of DGL on edge mask operation, this explainer need the to-be-explained models to
    accept an additional input argument, edge mask, and apply this mask in their inner message parse operation.
    This is current walk_around to use edge masks.
    """

    # Class inner variables
    loss_coef = {"g_size": 0.05, "feat_size": 1.0, "g_ent": 0.1, "feat_ent": 0.1}

    def __init__(
        self,
        model,
        num_edges,
        node_feat_dim,
        activation="sigmoid",
        agg_fn="sum",
        mask_bias=False,
    ):
        super(NodeExplainerModule, self).__init__()
        self.model = model
        self.model.eval()
        self.num_edges = num_edges
        self.node_feat_dim = node_feat_dim
        self.activation = activation
        self.agg_fn = agg_fn
        self.mask_bias = mask_bias

        # Initialize parameters on masks
        self.edge_mask, self.edge_mask_bias = self.create_edge_mask(self.num_edges)
        self.node_feat_mask = self.create_node_feat_mask(self.node_feat_dim)

        self.loss_fn = nn.BCEWithLogitsLoss()

    def create_edge_mask(self, num_edges, init_strategy="normal", const=1.0):
        """
        Based on the number of nodes in the computational graph, create a learnable mask of edges.
        To adopt to DGL, change this mask from N*N adjacency matrix to the No. of edges
        Parameters
        ----------
        num_edges: Integer N, specify the number of edges.
        init_strategy: String, specify the parameter initialization method
        const: Float, a value for constant initialization
        Returns
        -------
        mask and mask bias: Tensor, all in shape of N*1
        """
        mask = nn.Parameter(th.Tensor(num_edges, 1))

        if init_strategy == "normal":
            std = nn.init.calculate_gain("relu") * math.sqrt(1.0 / num_edges)
            with th.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "const":
            nn.init.constant_(mask, const)

        if self.mask_bias:
            mask_bias = nn.Parameter(th.Tensor(num_edges, 1))
            nn.init.constant_(mask_bias, 0.0)
        else:
            mask_bias = None

        return mask, mask_bias

    def create_node_feat_mask(self, node_feat_dim, init_strategy="normal"):
        """
        Based on the dimensions of node feature in the computational graph, create a learnable mask of features.
        Parameters
        ----------
        node_feat_dim: Integer N, dimensions of node feature
        init_strategy: String, specify the parameter initialization method
        Returns
        -------
        mask: Tensor, in shape of N
        """
        mask = nn.Parameter(th.Tensor(node_feat_dim))

        if init_strategy == "normal":
            std = 0.1
            with th.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "constant":
            with th.no_grad():
                nn.init.constant_(mask, 0.0)
        return mask

    def forward(self, graph, feats):
        """
        Calculate prediction results after masking input of the given model.
        Parameters
        ----------
        graph: DGLGraph, Should be a sub_graph of the target node to be explained.
        n_idx: Tensor, an integer, index of the node to be explained.
        Returns
        -------
        new_logits: Tensor, in shape of N * Num_Classes
        """

        # Step 1: Mask node feature with the inner feature mask
        new_feats = feats * self.node_feat_mask.sigmoid()
        edge_mask = self.edge_mask.sigmoid().reshape(-1)

        graph_copy = copy.deepcopy(graph)

        # Step 2: Add compute logits after mask node features and edges
        if "weight" in graph_copy.edata.keys():
            graph_copy.edata["weight"] = graph_copy.edata["weight"] * edge_mask
        else:

            graph_copy.edata["weight"] = edge_mask
        new_logits, _ = self.model([graph_copy], new_feats)

        return new_logits

    def _loss(self, pred_logits, pred_label):
        """
        Compute the losses of this explainer, which include 6 parts in author's codes:
        1. The prediction loss between predict logits before and after node and edge masking;
        2. Loss of edge mask itself, which tries to put the mask value to either 0 or 1;
        3. Loss of node feature mask itself,  which tries to put the mask value to either 0 or 1;
        4. L2 loss of edge mask weights, but in sum not in mean;
        5. L2 loss of node feature mask weights, which is NOT used in the author's codes;
        6. Laplacian loss of the adj matrix.
        In the PyG implementation, there are 5 types of losses:
        1. The prediction loss between logits before and after node and edge masking;
        2. Sum loss of edge mask weights;
        3. Loss of edge mask entropy, which tries to put the mask value to either 0 or 1;
        4. Sum loss of node feature mask weights;
        5. Loss of node feature mask entropy, which tries to put the mask value to either 0 or 1;
        Parameters
        ----------
        pred_logits：Tensor, N-dim logits output of model
        pred_label: Tensor, N-dim one-hot label of the label
        Returns
        -------
        loss: Scalar, the overall loss of this explainer.
        """
        # 1. prediction loss
        # log_logit = - F.log_softmax(pred_logits, dim=-1)
        # pred_loss = th.sum(log_logit * pred_label)
        pred_loss = self.loss_fn(pred_logits.float(), pred_label.float())

        # 2. edge mask loss
        if self.activation == "sigmoid":
            edge_mask = th.sigmoid(self.edge_mask)
        elif self.activation == "relu":
            edge_mask = F.relu(self.edge_mask)
        else:
            raise ValueError()
        edge_mask_loss = self.loss_coef["g_size"] * th.sum(edge_mask)

        # 3. edge mask entropy loss
        edge_ent = -edge_mask * th.log(edge_mask + 1e-8) - (1 - edge_mask) * th.log(
            1 - edge_mask + 1e-8
        )
        edge_ent_loss = self.loss_coef["g_ent"] * th.mean(edge_ent)

        # 4. node feature mask loss
        if self.activation == "sigmoid":
            node_feat_mask = th.sigmoid(self.node_feat_mask)
        elif self.activation == "relu":
            node_feat_mask = F.relu(self.node_feat_mask)
        else:
            raise ValueError()
        node_feat_mask_loss = self.loss_coef["feat_size"] * th.sum(node_feat_mask)

        # 5. node feature mask entry loss
        node_feat_ent = -node_feat_mask * th.log(node_feat_mask + 1e-8) - (
            1 - node_feat_mask
        ) * th.log(1 - node_feat_mask + 1e-8)
        node_feat_ent_loss = self.loss_coef["feat_ent"] * th.mean(node_feat_ent)

        total_loss = (
            pred_loss
            + edge_mask_loss
            + edge_ent_loss
            + node_feat_mask_loss
            + node_feat_ent_loss
        )

        return total_loss


def gnn_explain(graph, feats, model, node_idx, feature_dim, args):

    # extract subgraphs
    sub_graph, ori_n_idxes, new_n_idx = extract_subgraph(graph, node_idx, hops=args.hop)
    sub_graph = dgl.remove_self_loop(sub_graph)
    sub_graph = dgl.add_self_loop(sub_graph)
    print(sub_graph)

    # sub-graph features
    sub_feats = feats[ori_n_idxes.long(), :]

    # create an explainer
    explainer = NodeExplainerModule(
        model=model, num_edges=sub_graph.number_of_edges(), node_feat_dim=feature_dim
    )

    # define optimizer
    optim = torch.optim.Adam(
        explainer.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # get trained model predictions
    model.eval()
    model_logits, _ = model([sub_graph], sub_feats)
    model_predict = (torch.sigmoid(model_logits) >= args.pred_thresh).long()

    # train explainer
    prev_loss = 1e10
    for epoch in tqdm(range(args.num_epochs)):
        explainer.train()
        exp_logits = explainer(sub_graph, sub_feats)
        loss = explainer._loss(exp_logits[new_n_idx], model_predict[new_n_idx])
        optim.zero_grad()
        #         loss.backward(retain_graph=True)
        loss.backward()
        optim.step()

        if loss.item() > prev_loss:
            patience += 1
        else:
            patience = 0

        if patience == args.patience:
            print("Early stopping at epoch {}...".format(epoch))
            break
        prev_loss = loss.item()

        wandb.log({"loss": loss.item(), "epoch": epoch})

    return explainer, sub_graph, ori_n_idxes, new_n_idx


######## For Fusion MM-STGNN ########
class NodeExplainerModule_JointFusion(nn.Module):
    """
    A Pytorch module for explaining a node's prediction based on its computational graph and node features.
    Use two masks: One mask on edges, and another on nodes' features.
    So far due to the limit of DGL on edge mask operation, this explainer need the to-be-explained models to
    accept an additional input argument, edge mask, and apply this mask in their inner message parse operation.
    This is current walk_around to use edge masks.
    """

    # Class inner variables
    loss_coef = {"g_size": 0.05, "feat_size": 1.0, "g_ent": 0.1, "feat_ent": 0.1}

    def __init__(
        self,
        model,
        num_edges,
        ehr_node_feat_dim,
        img_node_feat_dim,
        activation="sigmoid",
        agg_fn="sum",
        mask_bias=False,
    ):
        super(NodeExplainerModule_JointFusion, self).__init__()
        self.model = model
        self.model.eval()
        self.num_edges = num_edges
        self.ehr_node_feat_dim = ehr_node_feat_dim
        self.img_node_feat_dim = img_node_feat_dim
        self.activation = activation
        self.agg_fn = agg_fn
        self.mask_bias = mask_bias

        # Initialize parameters on masks
        self.edge_mask, self.edge_mask_bias = self.create_edge_mask(self.num_edges)
        self.ehr_node_feat_mask = self.create_node_feat_mask(self.ehr_node_feat_dim)
        self.img_node_feat_mask = self.create_node_feat_mask(self.img_node_feat_dim)

        self.loss_fn = nn.BCEWithLogitsLoss()

    def create_edge_mask(self, num_edges, init_strategy="normal", const=1.0):
        """
        Based on the number of nodes in the computational graph, create a learnable mask of edges.
        To adopt to DGL, change this mask from N*N adjacency matrix to the No. of edges
        Parameters
        ----------
        num_edges: Integer N, specify the number of edges.
        init_strategy: String, specify the parameter initialization method
        const: Float, a value for constant initialization
        Returns
        -------
        mask and mask bias: Tensor, all in shape of N*1
        """
        mask = nn.Parameter(th.Tensor(num_edges, 1))

        if init_strategy == "normal":
            std = nn.init.calculate_gain("relu") * math.sqrt(1.0 / num_edges)
            with th.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "const":
            nn.init.constant_(mask, const)

        if self.mask_bias:
            mask_bias = nn.Parameter(th.Tensor(num_edges, 1))
            nn.init.constant_(mask_bias, 0.0)
        else:
            mask_bias = None

        return mask, mask_bias

    def create_node_feat_mask(self, node_feat_dim, init_strategy="normal"):
        """
        Based on the dimensions of node feature in the computational graph, create a learnable mask of features.
        Parameters
        ----------
        node_feat_dim: Integer N, dimensions of node feature
        init_strategy: String, specify the parameter initialization method
        Returns
        -------
        mask: Tensor, in shape of N
        """
        mask = nn.Parameter(th.Tensor(node_feat_dim))

        if init_strategy == "normal":
            std = 0.1
            with th.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == "constant":
            with th.no_grad():
                nn.init.constant_(mask, 0.0)
        return mask

    def forward(self, graph, img_feats, ehr_feats):
        """
        Calculate prediction results after masking input of the given model.
        Parameters
        ----------
        graph: DGLGraph, Should be a sub_graph of the target node to be explained.
        n_idx: Tensor, an integer, index of the node to be explained.
        Returns
        -------
        new_logits: Tensor, in shape of N * Num_Classes
        """

        # Step 1: Mask node feature with the inner feature mask
        new_ehr_feats = ehr_feats * self.ehr_node_feat_mask.sigmoid()
        new_img_feats = img_feats * self.img_node_feat_mask.sigmoid()
        edge_mask = self.edge_mask.sigmoid().reshape(-1)

        graph_copy = copy.deepcopy(graph)

        # Step 2: Add compute logits after mask node features and edges
        if "weight" in graph_copy.edata.keys():
            graph_copy.edata["weight"] = graph_copy.edata["weight"] * edge_mask
        else:
            graph_copy.edata["weight"] = edge_mask
        new_logits = self.model([graph_copy], new_img_feats, new_ehr_feats)

        return new_logits

    def _loss(self, pred_logits, pred_label):
        """
        Compute the losses of this explainer, which include 6 parts in author's codes:
        1. The prediction loss between predict logits before and after node and edge masking;
        2. Loss of edge mask itself, which tries to put the mask value to either 0 or 1;
        3. Loss of node feature mask itself,  which tries to put the mask value to either 0 or 1;
        4. L2 loss of edge mask weights, but in sum not in mean;
        5. L2 loss of node feature mask weights, which is NOT used in the author's codes;
        6. Laplacian loss of the adj matrix.
        In the PyG implementation, there are 5 types of losses:
        1. The prediction loss between logits before and after node and edge masking;
        2. Sum loss of edge mask weights;
        3. Loss of edge mask entropy, which tries to put the mask value to either 0 or 1;
        4. Sum loss of node feature mask weights;
        5. Loss of node feature mask entropy, which tries to put the mask value to either 0 or 1;
        Parameters
        ----------
        pred_logits：Tensor, N-dim logits output of model
        pred_label: Tensor, N-dim one-hot label of the label
        Returns
        -------
        loss: Scalar, the overall loss of this explainer.
        """
        # 1. prediction loss
        # log_logit = - F.log_softmax(pred_logits, dim=-1)
        # pred_loss = th.sum(log_logit * pred_label)
        pred_loss = self.loss_fn(pred_logits.float(), pred_label.float())

        # 2. edge mask loss
        if self.activation == "sigmoid":
            edge_mask = th.sigmoid(self.edge_mask)
        elif self.activation == "relu":
            edge_mask = F.relu(self.edge_mask)
        else:
            raise ValueError()
        edge_mask_loss = self.loss_coef["g_size"] * th.sum(edge_mask)

        # 3. edge mask entropy loss
        edge_ent = -edge_mask * th.log(edge_mask + 1e-8) - (1 - edge_mask) * th.log(
            1 - edge_mask + 1e-8
        )
        edge_ent_loss = self.loss_coef["g_ent"] * th.mean(edge_ent)

        # 4. node feature mask loss
        if self.activation == "sigmoid":
            ehr_node_feat_mask = th.sigmoid(self.ehr_node_feat_mask)
            img_node_feat_mask = th.sigmoid(self.img_node_feat_mask)
        elif self.activation == "relu":
            ehr_node_feat_mask = F.relu(self.ehr_node_feat_mask)
            img_node_feat_mask = F.relu(self.img_node_feat_mask)
        else:
            raise ValueError()
        ehr_node_feat_mask_loss = self.loss_coef["feat_size"] * th.sum(
            ehr_node_feat_mask
        )
        img_node_feat_mask_loss = self.loss_coef["feat_size"] * th.sum(
            img_node_feat_mask
        )

        # 5. node feature mask entry loss
        ehr_node_feat_ent = -ehr_node_feat_mask * th.log(ehr_node_feat_mask + 1e-8) - (
            1 - ehr_node_feat_mask
        ) * th.log(1 - ehr_node_feat_mask + 1e-8)
        ehr_node_feat_ent_loss = self.loss_coef["feat_ent"] * th.mean(ehr_node_feat_ent)
        img_node_feat_ent = -img_node_feat_mask * th.log(img_node_feat_mask + 1e-8) - (
            1 - img_node_feat_mask
        ) * th.log(1 - img_node_feat_mask + 1e-8)
        img_node_feat_ent_loss = self.loss_coef["feat_ent"] * th.mean(img_node_feat_ent)

        total_loss = (
            pred_loss
            + edge_mask_loss
            + edge_ent_loss
            + ehr_node_feat_mask_loss
            + img_node_feat_mask_loss
            + ehr_node_feat_ent_loss
            + img_node_feat_ent_loss
        )

        return total_loss


def gnn_explain_fusion(
    graph, ehr_feats, img_feats, model, node_idx, ehr_feature_dim, img_feature_dim, args
):

    # extract subgraphs
    sub_graph, ori_n_idxes, new_n_idx = extract_subgraph(graph, node_idx, hops=args.hop)
    sub_graph = dgl.remove_self_loop(sub_graph)
    sub_graph = dgl.add_self_loop(sub_graph)
    print(sub_graph)

    # sub-graph features
    sub_ehr_feats = ehr_feats[ori_n_idxes.long(), :]
    sub_img_feats = img_feats[ori_n_idxes.long(), :]

    # create an explainer
    explainer = NodeExplainerModule_JointFusion(
        model=model,
        num_edges=sub_graph.number_of_edges(),
        ehr_node_feat_dim=ehr_feature_dim,
        img_node_feat_dim=img_feature_dim,
        activation="sigmoid",
        agg_fn="sum",
        mask_bias=False,
    )

    # define optimizer
    optim = torch.optim.Adam(
        explainer.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # get trained model predictions
    model.eval()
    model_logits = model([sub_graph], sub_img_feats, sub_ehr_feats)
    model_predict = (torch.sigmoid(model_logits) >= args.pred_thresh).long()

    # train explainer
    prev_loss = 1e10
    for epoch in tqdm(range(args.num_epochs)):
        explainer.train()
        exp_logits = explainer(sub_graph, sub_img_feats, sub_ehr_feats)
        loss = explainer._loss(exp_logits[new_n_idx], model_predict[new_n_idx])
        optim.zero_grad()
        loss.backward()
        optim.step()

        if loss.item() > prev_loss:
            patience += 1
        else:
            patience = 0

        if patience == args.patience:
            print("Early stopping at epoch {}...".format(epoch))
            break
        prev_loss = loss.item()

        wandb.log({"loss": loss.item(), "epoch": epoch})

    return explainer, sub_graph, ori_n_idxes, new_n_idx


#############################


def main(params):
    graph = load_graphs(params.graph_dir)
    graph = g[0][0]

    if params.modality == "fusion":
        ehr_feats = graph.ndata["ehr_feat"]
        img_feats = graph.ndata["img_feat"]
        img_config = get_config("stgcn", args)
        ehr_config = get_config("stgcn", args)

        img_in_dim = img_feats.shape[-1]
        ehr_in_dim = ehr_feats.shape[-1]
        model = JointFusionModel(
            img_in_dim=img_in_dim,
            ehr_in_dim=ehr_in_dim,
            img_config=img_config,
            ehr_config=ehr_config,
            cat_idxs=CATEGORICAL_IDXS,
            cat_dims=CATEGORICAL_DIMS,
            ehr_encoder_name=args.ehr_encoder_name,
            cat_emb_dim=args.cat_emb_dim,
            joint_hidden=args.joint_hidden,
            num_classes=args.num_classes,
            dropout=args.dropout,
            device="cpu",
        )
    else:
        feats = graph.ndata["feat"]
        config = get_config("stgcn", args)
        in_dim = feats.shape[-1]
        model = GraphRNN(
            in_dim=in_dim,
            n_classes=args.num_classes,
            device="cpu",
            is_classifier=True,
            ehr_encoder_name=args.ehr_encoder_name
            if args.feature_type != "imaging"
            else None,
            ehr_config=None,
            cat_idxs=CATEGORICAL_IDXS,
            cat_dims=CATEGORICAL_DIMS,
            cat_emb_dim=args.cat_emb_dim,
            **config
        )
    model = utils.load_model_checkpoint(params.checkpoint_file, model)

    node_idx = torch.tensor([params.node_to_explain]).type(torch.int32)
    args_explainer = {
        "hop": params.hop,
        "lr": params.lr,
        "weight_decay": params.weight_decay,
        "pred_thresh": optimal_thresh,
        "num_epochs": params.num_epochs,
        "patience": 5,
    }
    args_explainer = DottedDict(args_explainer)

    print("Explaining node: ", params.node_to_explain)
    if params.modality == "fusion":
        explainer, sub_graph, ori_n_idxes, _ = gnn_explain_fusion(
            graph,
            ehr_feats,
            img_feats,
            model,
            node_idx=node_idx,
            ehr_feature_dim=ehr_in_dim,
            img_feature_dim=img_in_dim,
            args=args_explainer,
        )
    else:
        explainer, sub_graph, ori_n_idxes, _ = gnn_explain(
            graph,
            feats,
            model,
            node_idx=node_idx,
            feature_dim=in_dim,
            args=args_explainer,
        )
    with open(
        os.path.join(params.save_dir, "node{}_explainer.pkl".format(node_idx)),
        "wb",
    ) as pf:
        pickle.dump(
            {
                "explainer": explainer,
                "sub_graph": sub_graph,
                "ori_n_idxes": ori_n_idxes,
                "args_explainer": args_explainer,
            },
            pf,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN Explainer for STGNN/MM-STGNN.")
    parser.add_argument(
        "--graph_dir", type=str, default=None, help="Dir to preprocessed graph."
    )
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        default=None,
        help="Trained model checkpoint file.",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="fusion",
        choices=("fusion", "ehr", "imaging"),
        help="Modality of the model.",
    )
    parser.add_argument(
        "--save_dir", type=str, default="", help="Dir to save node explanation results."
    )
    parser.add_argument(
        "--hop", type=int, default=1, help="Number of hops for the subgraph."
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument(
        "--weight_decay", type=float, default=5e-4, help="Weight decay."
    )
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs.")
    args = parser.parse_args()
    main(args)
