import torch
import dgl
import os
import pickle
import pandas as pd
import numpy as np
from dgl.data import DGLDataset
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from data.readmission_utils import *


def construct_graph_readmission(
    df_demo,
    ehr_feature_file=None,
    edge_ehr_file=None,
    edge_modality=["demo"],
    feature_type="multimodal",
    img_feature_dir=None,
    top_perc=0.01,
    gauss_kernel=True,
    standardize=True,
    max_seq_len_img=9,
    max_seq_len_ehr=9,
    sim_measure="euclidean",
    ehr_types=["demo", "icd", "lab", "med"],
):
    """
    Construct an admission graph
    Args:
        df_demo: dataframe of cohort with demographic and imaging information
        ehr_feature_file: file of preprocessed EHR feature
        edge_ehr_file: file of preprocesdded EHR feature for edges
        edge_modality: list of EHR sources for edge
        feature_type: "multimodal", "imaging" or "non-imaging"
        img_feature_dir: dir to extracted imaging features
        top_perc: top percentage edges to be kept for graph
        gauss_kernel: whether to use Gaussian kernel for edge weights
        standardize: whether to standardize node features
        max_seq_len_img: maximum sequence length of imaging features
        max_seq_len_ehr: maximum sequence length of EHR features
        sim_measure: metric to measure node similarity for edges
        ehr_types: list of EHR sources for node features
    Returns:
        node2idx: dict, key is node name, value is node index
        dgl_G: dgl graph
        cat_idxs: list of categorical column indices
        cat_dims: list of categorical dimensions
    """
    if feature_type not in ["imaging", "non-imaging", "multimodal"]:
        raise NotImplementedError

    if sim_measure not in ["cosine", "euclidean"]:
        raise NotImplementedError

    # node labels
    (
        labels,
        node_included_files,
        splits,
        _,
        hospital_stays,
        time_idxs,
    ) = get_readmission_label_mimic(df_demo, max_seq_len=None)
    train_idxs = np.array([ind for ind in range(len(splits)) if splits[ind] == "train"])
    node_names = list(node_included_files.keys())

    # node name to node index dict
    node2idx = {}
    for idx, name in enumerate(node_names):
        node2idx[name] = idx

    # node features (i.e. imaging features) to ndata
    if feature_type == "imaging":
        assert img_feature_dir is not None
        img_feat_dict = get_img_features(img_feature_dir, node_included_files)
        node_features, seq_lengths = get_feat_seq(
            node_included_files,
            img_feat_dict,
            max_seq_len_img,
            pad_front=False,
            time_deltas=None,
            padding_val=None,
        )
        cat_idxs = []
        cat_dims = []

    elif feature_type == "non-imaging":
        assert ehr_feature_file is not None
        with open(ehr_feature_file, "rb") as pf:
            raw_feat_dict = pickle.load(pf)
        feat_dict = raw_feat_dict["feat_dict"]
        feat_cols = raw_feat_dict["feature_cols"]
        cols_to_keep = []

        for ehr_name in ehr_types:
            cols_to_keep = cols_to_keep + raw_feat_dict["{}_cols".format(ehr_name)]

        col_idxs = np.array(
            [feat_cols.index(col) for col in cols_to_keep]
        )  # wrt original cols
        feat_dict = {
            name: feat_dict[name][:, col_idxs] for name in node_names
        }  # get relevant cols
        node_features, seq_lengths = get_feat_seq(
            node_included_files,
            feat_dict,
            max_seq_len_ehr,
            pad_front=False,
            time_deltas=None,
            padding_val=None,
        )

        if "cat_idxs" in raw_feat_dict:
            cat_col2dim = {
                feat_cols[raw_feat_dict["cat_idxs"][ind]]: raw_feat_dict["cat_dims"][
                    ind
                ]
                for ind in range(len(raw_feat_dict["cat_dims"]))
            }

            # reindex categorical variables
            cat_cols = [
                col
                for col in cols_to_keep
                if (feat_cols.index(col) in raw_feat_dict["cat_idxs"])
            ]
            cat_idxs = [cols_to_keep.index(col) for col in cat_cols]
            cat_dims = [cat_col2dim[col] for col in cat_cols]
        else:
            cat_idxs = []
            cat_dims = []

        assert np.all(node_features != -1)
        assert node_features.shape[1] == max_seq_len_ehr
        del feat_dict

        img_feat_dict = None

    elif feature_type == "multimodal":
        assert img_feature_dir is not None
        assert ehr_feature_file is not None

        # imaging features
        img_feat_dict = get_img_features(img_feature_dir, node_included_files)
        img_node_features, seq_lengths = get_feat_seq(
            node_included_files,
            img_feat_dict,
            max_seq_len_img,
            pad_front=False,
            time_deltas=None,
            padding_val=None,
        )

        # ehr features
        with open(ehr_feature_file, "rb") as pf:
            raw_ehr_dict = pickle.load(pf)
        ehr_feat_dict = raw_ehr_dict["feat_dict"]
        ehr_feat_cols = raw_ehr_dict["feature_cols"]
        cols_to_keep = []

        for ehr_name in ehr_types:
            cols_to_keep = cols_to_keep + raw_ehr_dict["{}_cols".format(ehr_name)]

        col_idxs = np.array(
            [ehr_feat_cols.index(col) for col in cols_to_keep]
        )  # wrt original cols
        ehr_feat_dict = {
            name: ehr_feat_dict[name][:, col_idxs] for name in node_names
        }  # get relevant cols
        ehr_node_features, seq_lengths = get_feat_seq(
            node_included_files,
            ehr_feat_dict,
            max_seq_len_ehr,
            pad_front=False,
            time_deltas=None,
            padding_val=None,
        )

        if "cat_idxs" in raw_ehr_dict:
            cat_col2dim = {
                ehr_feat_cols[raw_ehr_dict["cat_idxs"][ind]]: raw_ehr_dict["cat_dims"][
                    ind
                ]
                for ind in range(len(raw_ehr_dict["cat_dims"]))
            }

            # reindex categorical variables
            cat_cols = [
                col
                for col in cols_to_keep
                if (ehr_feat_cols.index(col) in raw_ehr_dict["cat_idxs"])
            ]
            cat_idxs = [cols_to_keep.index(col) for col in cat_cols]
            cat_dims = [cat_col2dim[col] for col in cat_cols]
        else:
            cat_idxs = []
            cat_dims = []

        del ehr_feat_dict
        del raw_ehr_dict

        node_features = None
    else:
        raise NotImplementedError

    # standardize
    if standardize:
        if feature_type == "non-imaging":
            seq_len = max_seq_len_ehr
            scaler = StandardScaler()
            train_feat = node_features[train_idxs].reshape(
                (len(train_idxs) * seq_len, -1)
            )
            # NOTE: only standardize non-categorical features
            continuous_cols = np.array(
                [ind for ind in range(train_feat.shape[-1]) if ind not in cat_idxs]
            )
            scaler.fit(train_feat[:, continuous_cols])
            continuous_node_features = scaler.transform(
                node_features[:, :, continuous_cols].reshape(
                    (len(node_names) * seq_len, -1)
                )
            )
            node_features[:, :, continuous_cols] = continuous_node_features.reshape(
                (len(node_names), seq_len, -1)
            )
            del train_feat
        elif feature_type == "multimodal":
            ehr_scaler = StandardScaler()
            ehr_train_feat = ehr_node_features[train_idxs].reshape(
                (len(train_idxs) * max_seq_len_ehr, -1)
            )
            # NOTE: only standardize non-categorical features
            continuous_cols = np.array(
                [ind for ind in range(ehr_train_feat.shape[-1]) if ind not in cat_idxs]
            )
            ehr_scaler.fit(ehr_train_feat[:, continuous_cols])
            continuous_ehr_node_features = ehr_scaler.transform(
                ehr_node_features[:, :, continuous_cols].reshape(
                    (len(node_names) * max_seq_len_ehr, -1)
                )
            )
            ehr_node_features[
                :, :, continuous_cols
            ] = continuous_ehr_node_features.reshape(
                (len(node_names), max_seq_len_ehr, -1)
            )
            del ehr_train_feat

    # edges
    node_edge_dict = {}
    if (
        ("demo" in edge_modality)
        or ("cpt" in edge_modality)
        or ("icd" in edge_modality)
        or ("lab" in edge_modality)
        or ("imaging" in edge_modality)
        or ("med" in edge_modality)
    ):
        assert edge_ehr_file is not None
        with open(edge_ehr_file, "rb") as pf:
            raw_ehr_dict = pickle.load(pf)
        feat_cols = raw_ehr_dict["feature_cols"]
        node_edge_dict = {}

        if "demo" in edge_modality:
            demo_col_idxs = np.array(
                [feat_cols.index(col) for col in raw_ehr_dict["demo_cols"]]
            )  # wrt original cols
            demo_dict = {
                name: raw_ehr_dict["feat_dict"][name][0, demo_col_idxs]
                for name in node_names
            }
            for name in node_names:
                if name not in node_edge_dict:
                    node_edge_dict[name] = []
                node_edge_dict[name] = np.concatenate(
                    [node_edge_dict[name], demo_dict[name]], axis=-1
                )

        if "med" in edge_modality:
            med_col_idxs = np.array(
                [feat_cols.index(col) for col in raw_ehr_dict["med_cols"]]
            )  # wrt original cols
            med_dict = {
                name: raw_ehr_dict["feat_dict"][name][0, med_col_idxs]
                for name in node_names
            }
            for name in node_names:
                if name not in node_edge_dict:
                    node_edge_dict[name] = []
                node_edge_dict[name] = np.concatenate(
                    [node_edge_dict[name], med_dict[name]], axis=-1
                )

        # time varying edges
        if (
            ("cpt" in edge_modality)
            or ("icd" in edge_modality)
            or ("lab" in edge_modality)
            or ("imaging" in edge_modality)
            or ("med" in edge_modality)
        ):
            if "cpt" in edge_modality:
                cpt_col_idxs = np.array(
                    [feat_cols.index(col) for col in raw_ehr_dict["cpt_cols"]]
                )  # wrt original cols
                cpt_dict = {
                    name: raw_ehr_dict["feat_dict"][name][:, cpt_col_idxs]
                    for name in node_names
                }
            else:
                cpt_dict = None
            if "icd" in edge_modality:
                icd_col_idxs = np.array(
                    [feat_cols.index(col) for col in raw_ehr_dict["icd_cols"]]
                )  # wrt original cols
                icd_dict = {
                    name: raw_ehr_dict["feat_dict"][name][:, icd_col_idxs]
                    for name in node_names
                }
            else:
                icd_dict = None
            if "lab" in edge_modality:
                lab_col_idxs = np.array(
                    [feat_cols.index(col) for col in raw_ehr_dict["lab_cols"]]
                )  # wrt original cols
                lab_dict = {
                    name: raw_ehr_dict["feat_dict"][name][:, lab_col_idxs]
                    for name in node_names
                }
            else:
                lab_dict = None
            if "med" in edge_modality:
                med_col_idxs = np.array(
                    [feat_cols.index(col) for col in raw_ehr_dict["med_cols"]]
                )  # wrt original cols
                med_dict = {
                    name: raw_ehr_dict["feat_dict"][name][:, med_col_idxs]
                    for name in node_names
                }
            else:
                med_dict = None
            ehr_edge_dict = get_time_varying_edges(
                node_included_files=node_included_files,
                edge_dict=node_edge_dict,
                edge_modality=edge_modality,
                hospital_stay=hospital_stays,
                cpt_dict=cpt_dict,
                icd_dict=icd_dict,
                lab_dict=lab_dict,
                med_dict=med_dict,
            )
        else:
            ehr_edge_dict = node_edge_dict
    else:
        ehr_edge_dict = {}

    print("Using {} for similarity/distance measure...".format(sim_measure))
    if sim_measure == "cosine":
        ehr_dist_dict = compute_cos_sim_mat(ehr_edge_dict, scale=True)
    elif sim_measure == "euclidean":
        ehr_dist_dict = compute_dist_mat(ehr_edge_dict, scale=True)
    else:
        raise NotImplementedError

    # Construct graphs
    edges = compute_edges(
        ehr_dist_dict,
        node_names,
        top_perc=top_perc,
        gauss_kernel=gauss_kernel,
    )

    edge_dict = {
        "From": ehr_dist_dict["From"],
        "To": ehr_dist_dict["To"],
        "Weight": [],
    }

    src_nodes = []
    dst_nodes = []
    weights = []
    for idx in range(len(edge_dict["From"])):
        from_node_name = edge_dict["From"][idx]
        to_node_name = edge_dict["To"][idx]

        if (from_node_name not in node2idx) or (to_node_name not in node2idx):
            raise ValueError

        from_node = node2idx[from_node_name]
        to_node = node2idx[to_node_name]

        if edges[idx] == 0:
            edge_dict["Weight"].append(0)  # no edge
        else:
            edge_dict["Weight"].append(edges[idx])
            src_nodes.append(from_node)
            dst_nodes.append(to_node)
            weights.append(edges[idx])
    src_nodes = torch.tensor(src_nodes)
    dst_nodes = torch.tensor(dst_nodes)

    del edge_dict

    g_directed = dgl.graph((src_nodes, dst_nodes), idtype=torch.int32)
    g_directed.edata["weight"] = torch.FloatTensor(weights)

    dgl_G = dgl.add_reverse_edges(g_directed, copy_ndata=True, copy_edata=True)
    dgl_G = dgl.to_simple(dgl_G, return_counts=None, copy_ndata=True, copy_edata=True)

    num_nodes = dgl_G.num_nodes()
    train_masks = torch.zeros(num_nodes, dtype=torch.int32)
    val_masks = torch.zeros(num_nodes, dtype=torch.int32)
    test_masks = torch.zeros(num_nodes, dtype=torch.int32)

    train_ind = torch.LongTensor(
        [ind for ind in range(len(splits)) if splits[ind] == "train"]
    )
    val_ind = torch.LongTensor(
        [ind for ind in range(len(splits)) if splits[ind] == "val"]
    )
    test_ind = torch.LongTensor(
        [ind for ind in range(len(splits)) if splits[ind] == "test"]
    )
    train_masks[train_ind] = 1
    val_masks[val_ind] = 1
    test_masks[test_ind] = 1

    dgl_G.ndata["train_mask"] = train_masks
    dgl_G.ndata["val_mask"] = val_masks
    dgl_G.ndata["test_mask"] = test_masks

    dgl_G.ndata["label"] = torch.FloatTensor(labels)
    dgl_G.ndata["seq_lengths"] = torch.FloatTensor(seq_lengths)

    if feature_type == "multimodal":
        dgl_G.ndata["img_feat"] = torch.FloatTensor(img_node_features)
        dgl_G.ndata["ehr_feat"] = torch.FloatTensor(ehr_node_features)
    else:
        dgl_G.ndata["feat"] = torch.FloatTensor(node_features)

    return node2idx, dgl_G, cat_idxs, cat_dims


class ReadmissionDataset(DGLDataset):
    def __init__(
        self,
        demo_file,
        edge_ehr_file=None,
        ehr_feature_file=None,
        edge_modality=["demo"],
        feature_type="multimodal",
        img_feature_dir=None,
        top_perc=None,
        gauss_kernel=False,
        max_seq_len_img=6,
        max_seq_len_ehr=8,
        sim_measure="euclidean",
        standardize=True,
        ehr_types=["demo", "cpt", "icd", "lab", "med"],
    ):
        """
        Args:
            demo_file: file of cohort with demographic and imaging information
            ehr_feature_file: file of preprocessed EHR feature
            edge_ehr_file: file of preprocesdded EHR feature for edges
            edge_modality: list of EHR sources for edge
            feature_type: "multimodal", "imaging" or "non-imaging"
            img_feature_dir: dir to extracted imaging features
            top_perc: top percentage edges to be kept for graph
            gauss_kernel: whether to use Gaussian kernel for edge weights
            standardize: whether to standardize node features
            max_seq_len_img: maximum sequence length of imaging features
            max_seq_len_ehr: maximum sequence length of EHR features
            sim_measure: metric to measure node similarity for edges
            ehr_types: list of EHR sources for node features
        """
        self.demo_file = demo_file
        self.edge_modality = edge_modality
        self.feature_type = feature_type
        self.img_feature_dir = img_feature_dir
        self.ehr_feature_file = ehr_feature_file
        self.edge_ehr_file = edge_ehr_file
        self.top_perc = top_perc
        self.gauss_kernel = gauss_kernel
        self.max_seq_len_img = max_seq_len_img
        self.max_seq_len_ehr = max_seq_len_ehr
        self.sim_measure = sim_measure
        self.standardize = standardize
        self.ehr_types = ehr_types

        if sim_measure not in ["cosine", "euclidean"]:
            raise NotImplementedError

        print("Edge modality:", edge_modality)
        print("EHR types:", ehr_types)

        # get patients
        self.df_all = pd.read_csv(demo_file)

        super().__init__(name="readmission")

    def process(self):
        (
            self.node2idx,
            self.graph,
            self.cat_idxs,
            self.cat_dims,
        ) = construct_graph_readmission(
            df_demo=self.df_all,
            ehr_feature_file=self.ehr_feature_file,
            edge_ehr_file=self.edge_ehr_file,
            edge_modality=self.edge_modality,
            feature_type=self.feature_type,
            img_feature_dir=self.img_feature_dir,
            top_perc=self.top_perc,
            gauss_kernel=self.gauss_kernel,
            standardize=self.standardize,
            max_seq_len_img=self.max_seq_len_img,
            max_seq_len_ehr=self.max_seq_len_ehr,
            sim_measure=self.sim_measure,
            ehr_types=self.ehr_types,
        )

        self.targets = self.graph.ndata["label"].cpu().numpy()

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1
