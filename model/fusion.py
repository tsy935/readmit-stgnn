import sys
import torch
import torch.nn as nn

sys.path.append("../")
from model.model import *
import copy


class JointFusionModel(nn.Module):
    """
    Joint fusion model combining imaging and EHR data
    """

    def __init__(
        self,
        img_in_dim,
        ehr_in_dim,
        img_config,
        ehr_config,
        cat_idxs=None,
        cat_dims=None,
        ehr_encoder_name="embedder",
        ehr_checkpoint_path=None,
        cat_emb_dim=1,
        joint_hidden=[128],
        num_classes=1,
        freeze_pretrained=False,
        dropout=0.0,
        device="cpu",
    ):
        """
        Args:
            img_encoder: e.g. a lightweight CNN or linear layer
            ehr_encoder: e.g. RNN/GRU/LSTM or linear layer
            joint_hidden: list of hidden sizes for joint layer
        """
        super(JointFusionModel, self).__init__()

        if img_config["hidden_dim"] != ehr_config["hidden_dim"]:
            raise ValueError(
                "hidden_dim for img_config and ehr_config must be the same!"
            )

        # image encoder
        self.img_model = GraphRNN(
            in_dim=img_in_dim,
            n_classes=num_classes,
            device=device,
            is_classifier=False,
            **img_config
        )

        # ehr encoder
        self.ehr_model = GraphRNN(
            in_dim=ehr_in_dim,
            n_classes=num_classes,
            device=device,
            is_classifier=False,
            ehr_encoder_name=ehr_encoder_name,
            ehr_checkpoint_path=ehr_checkpoint_path,
            freeze_pretrained=freeze_pretrained,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=cat_emb_dim,
            **ehr_config
        )

        self.joint_hidden = joint_hidden
        self.num_classes = num_classes
        self.dropout = dropout

        # joint MLP layer
        self.mlp = []
        self.mlp.append(nn.Linear(img_config["hidden_dim"] * 2, joint_hidden[0]))
        self.mlp.append(nn.ReLU())
        self.mlp.append(nn.Dropout(p=dropout))
        for idx_hid in range(1, len(joint_hidden)):
            self.mlp.append(nn.Linear(joint_hidden[idx_hid - 1], joint_hidden[idx_hid]))
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Dropout(p=dropout))
        self.mlp.append(nn.Linear(joint_hidden[-1], num_classes))
        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, g, img_inputs, ehr_inputs):
        """
        Args:
            g: list of dgl graph
            img_inputs: shape (batch, max_seq_len, img_input_dim)
            ehr_inputs: shape (batch, max_seq_len, ehr_input_dim)
        """
        img_inputs = self.img_model(g, img_inputs)

        ehr_inputs = self.ehr_model(g, ehr_inputs)

        h = torch.cat([img_inputs, ehr_inputs], dim=-1)  # (batch, hidden_dim*2)

        logits = self.mlp(h)  # (batch, num_classes)

        if logits.shape[-1] == 1:
            logits = logits.squeeze(-1)

        return logits
