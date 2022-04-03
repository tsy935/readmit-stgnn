import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import pickle


def compute_dist_mat(demo_dict, scale=False):
    """
    Args:
        demo_dict: dict, key is node name, value is EHR feature vector
        scale: if True, will perform min-max scaling.
    Returns:
        dist_dict: dict of pairwise distances between nodes
    """

    demo_arr = []
    for _, arr in demo_dict.items():
        demo_arr.append(arr)
    demo_arr = np.stack(demo_arr, axis=0)

    # Scaler to scale each continuous variable to be between 0 and 1
    if scale:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(demo_arr)
        demo_arr = scaler.transform(demo_arr)

    distances = euclidean_distances(X=demo_arr, Y=demo_arr)

    dist_dict = {"From": [], "To": [], "Distance": [], "Mask": []}
    node_names = list(demo_dict.keys())
    for idx_node1 in tqdm(range(len(node_names))):
        for idx_node2 in range(idx_node1, len(node_names)):
            node1 = node_names[idx_node1]
            node2 = node_names[idx_node2]

            dist_dict["From"].append(node1)
            dist_dict["To"].append(node2)
            dist_dict["Distance"].append(distances[idx_node1, idx_node2])

    return dist_dict


def compute_cos_sim_mat(demo_dict, scale=False):
    """
    Args:
        demo_dict: key is patient, value is EHR feature vector
        scale: if True, will perform min-max scaling
    Returns:
        cos_sim_dict: dict of pairwise cosine similarity between nodes
    """

    # Scaler to scale each variable to be between 0 and 1
    demo_arr = []
    for _, arr in demo_dict.items():
        demo_arr.append(arr)
    demo_arr = np.stack(demo_arr, axis=0)

    # Scaler to scale each continuous variable to be between 0 and 1
    if scale:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(demo_arr)
        demo_arr = scaler.transform(demo_arr)

    cos_sim = cosine_similarity(demo_arr, demo_arr)

    cos_sim_dict = {"From": [], "To": [], "CosineSim": [], "Mask": []}
    node_names = list(demo_dict.keys())
    for idx_node1 in tqdm(range(len(node_names))):
        for idx_node2 in range(idx_node1, len(node_names)):
            node1 = node_names[idx_node1]
            node2 = node_names[idx_node2]

            cos_sim_dict["From"].append(node_names[idx_node1])
            cos_sim_dict["To"].append(node_names[idx_node2])
            cos_sim_dict["CosineSim"].append(cos_sim[idx_node1, idx_node2])

    return cos_sim_dict


def get_feat_seq(
    node_included_files,
    feature_dict,
    max_seq_len,
    pad_front=False,
    time_deltas=None,
    padding_val=None,
):
    """
    Args:
        node_included_files: dict, key is node name, value is list of imaging files
        feature_dict: dict, key is node name, value is imaging/EHR feature vector
        max_seq_len: int, maximum sequence length
        pad_front: if True, will pad to the front with the first timestep, else pad to the end with the last timestep
        time_deltas: if not None, will pad time deltas to features
        padding_val: if not None, will pad with this value instead of last/first time step
    Returns:
        padded_features: numpy array, shape (sample_size, max_seq_len, feature_dim)
        seq_len: original sequence length without any padding
    """
    seq_lengths = []
    padded_features = []
    padded_time_deltas = []
    for name, files in node_included_files.items():
        feature = feature_dict[name]
        orig_seq_len = len(files)
        feature = feature[-max_seq_len:, :]  # get last max_seq_len time steps

        if time_deltas != None:
            time_dt = time_deltas[name][-max_seq_len:]  # (max_seq_len,)
            assert len(time_dt) == feature.shape[0]

        if feature.shape[0] < max_seq_len:
            if not pad_front:
                # pad with last timestep or padding_val
                if padding_val is None:
                    padded = np.repeat(
                        feature[-1, :].reshape(1, -1),
                        repeats=max_seq_len - feature.shape[0],
                        axis=0,
                    )
                else:
                    padded = (
                        np.ones((max_seq_len - feature.shape[0], feature.shape[1]))
                        * padding_val
                    )
                feature = np.concatenate([feature, padded], axis=0)
                if time_deltas != None:
                    padded_dt = np.zeros((max_seq_len - time_dt.shape[0]))
                    time_dt = np.concatenate([time_dt, padded_dt], axis=0)
            else:
                # pad with first timestep or padding_val
                if padding_val is None:
                    padded = np.repeat(
                        feature[0, :].reshape(1, -1),
                        repeats=max_seq_len - feature.shape[0],
                        axis=0,
                    )
                else:
                    padded = (
                        np.ones((max_seq_len - feature.shape[0], feature.shape[1]))
                        * padding_val
                    )
                feature = np.concatenate([padded, feature], axis=0)
                if time_deltas != None:
                    padded_dt = np.zeros((max_seq_len - time_dt.shape[0]))
                    time_dt = np.concatenate([padded_dt, time_dt], axis=0)
        padded_features.append(feature)
        if time_deltas != None:
            padded_time_deltas.append(time_dt)
        seq_len = np.minimum(max_seq_len, orig_seq_len)
        seq_lengths.append(seq_len)

    padded_features = np.stack(padded_features)
    seq_lengths = np.stack(seq_lengths)
    if time_deltas != None:
        padded_time_deltas = np.expand_dims(np.stack(padded_time_deltas), axis=-1)
        padded_features = np.concatenate([padded_features, padded_time_deltas], axis=-1)

    return padded_features, seq_lengths


def get_img_features(feature_dir, node_included_files):
    """
    Args:
        feature_dir: dir to imaging features
        node_included_files: dict, key is node name, value is list of image paths
    Returns:
        img_feature_dict: dict, key is image path, value is image features within one hospitalization, shape (num_cxrs, feature_dim)
    """

    img_feature_dict = {}
    for name, files in tqdm(node_included_files.items()):
        curr_feat = []

        for img_dir in files:
            with open(
                os.path.join(feature_dir, img_dir.split("/")[-1] + ".pkl"), "rb"
            ) as pf:
                feature = pickle.load(pf)
            curr_feat.append(feature)
        curr_feat = np.stack(curr_feat, axis=0)  # (num_cxrs, feature_dim)
        img_feature_dict[name] = curr_feat

    return img_feature_dict


def get_time_varying_edges(
    node_included_files,
    edge_dict,
    edge_modality,
    hospital_stay,
    cpt_dict=None,
    icd_dict=None,
    lab_dict=None,
    med_dict=None,
):
    """
    Args:
        node_included_files: dict, key is node name, value is list of image paths
        edge_dict: dict, key is node name, value is EHR features
        edge_modality: list of EHR sources for edges
        hospital_stay: numpy array, lengths of hospital stays, shape (sample_size,)
        cpt_dict: dict, key is node name, value is preprocessed CPT features
        icd_dict: dict, key is node name, value is preprocessed ICD features
        lab_dict: dict, key is node name, value is preprocessed lab features
        med_dict: dict, key is node name, value is preprocessed medication features
    Returns:
        edge_dict: dict, key is node name, value is EHR features for edges
    """
    if edge_dict is None:
        edge_dict = {}

    for i, name in enumerate(list(node_included_files.keys())):
        if name not in edge_dict:
            edge_dict[name] = []
        # for cpt or icd or med, we sum over all days & average by length of stay (in days)
        if "cpt" in edge_modality:
            edge_dict[name] = np.concatenate(
                [
                    edge_dict[name],
                    np.sum(cpt_dict[name], axis=0) / hospital_stay[name],
                ],
                axis=-1,
            )
        if "icd" in edge_modality:
            # MIMIC ICD is non-temporal
            edge_dict[name] = np.concatenate(
                [
                    edge_dict[name],
                    icd_dict[name][-1, :] / hospital_stay[name],
                ],
                axis=-1,
            )
        if "med" in edge_modality:
            edge_dict[name] = np.concatenate(
                [
                    edge_dict[name],
                    np.sum(med_dict[name], axis=0) / hospital_stay[name],
                ],
                axis=-1,
            )
        # NOTE: for lab, take the last time step
        if "lab" in edge_modality:
            edge_dict[name] = np.concatenate(
                [edge_dict[name], lab_dict[name][-1, :]], axis=-1
            )
    return edge_dict


def compute_edges(
    dist_dict,
    node_names,
    top_perc=0.01,
    gauss_kernel=True,
):
    """
    Computes edge weights
    Args:
        dist_dict: dict with computed distance measures between nodes
        node_names: list of node names
        top_perc: top percentage of edges to be kept
        gauss_kernel: if True, will apply Gaussian kernel to Euclidean distance measures
    Returns:
        edges: numpy array of edge weights, shape (num_edges,)
    """
    if "CosineSim" in dist_dict:
        cos_sim = np.array(dist_dict["CosineSim"])
        dist = 1 - cos_sim
    else:
        cos_sim = None
        dist = np.array(dist_dict["Distance"])

    # sanity check shape, (num_nodes) * (num_nodes + 1) / 2, if consider self-edges
    assert len(dist) == (len(node_names) * (len(node_names) + 1) / 2)

    # apply gaussian kernel, use cosine distance instead of cosine similarity
    if gauss_kernel or (cos_sim is None):
        std = dist.std()
        edges = np.exp(-np.square(dist / std))
    else:
        edges = cos_sim

    # mask the edges
    if top_perc is not None:
        num = len(edges)
        num_to_keep = int(num * top_perc)
        sorted_dist = np.sort(edges)[::-1]  # descending order
        thresh = sorted_dist[:num_to_keep][-1]
        mask = edges >= thresh
        mask[edges < 0] = 0  # no edge for negative "distance"
        edges = edges * mask

    return edges


def get_readmission_label_mimic(df_demo, max_seq_len=None):
    """
    Args:
        df_demo: dataframe with patient readmission info and demographics:
        max_seq_len: maximum number of cxrs to use, count backwards from last cxr within the hospitalization
                    if max_seq_len=None, will use all cxrs
    Returns:
        labels: numpy array, readmission labels, same order as rows in df_demo, shape (num_admissions,)
        node_included_files: dict, key is node name, value is list of image files
        label_splits: list indicating the split of each datapoint in labels and node_included_files
        time_deltas: dict, key is node name, value is an array of day difference between currenct cxr to previous cxr
        total_stay: dict, key is node name, value is total length of stay (in days)
        time_idxs: dict, key is node name, value is the index of each cxr in terms of the day within hospitalization
    """
    labels = []
    node_included_files = {}
    label_splits = []
    time_deltas = {}
    total_stay = {}
    time_idxs = {}

    for _, row in tqdm(df_demo.iterrows(), total=len(df_demo)):
        pat = row["subject_id"]
        admit_dt = row["admittime"]
        discharge_dt = row["dischtime"]
        admit_id = row["hadm_id"]
        split = row["splits"]

        curr_name = str(pat) + "_" + str(admit_id)
        if curr_name not in node_included_files:
            node_included_files[curr_name] = []
        else:
            continue

        label_splits.append(split)

        # label
        if str(row["readmitted_within_30days"]).lower() == "true":
            labels.append(1)
        else:
            labels.append(0)

        admit_df = df_demo[df_demo["hadm_id"] == admit_id]

        # sort by study datetime
        admit_df = admit_df.sort_values(
            by=["StudyDate", "StudyTime"], ascending=[True, True]
        )

        # get list of cxrs sorted by study datetime
        for _, admit_row in admit_df.iterrows():
            node_included_files[curr_name].append(admit_row["image_path"])

        # get last max_seq_len cxrs if max_seq_len is specified
        if max_seq_len is not None:
            node_included_files[curr_name] = node_included_files[curr_name][
                -max_seq_len:
            ]

        # time delta & total hospital stay
        curr_timedelta = np.zeros((len(node_included_files[curr_name])))

        prev_date = pd.to_datetime(admit_dt)
        addmission_date = pd.to_datetime(admit_dt)
        discharge_date = pd.to_datetime(discharge_dt)

        hospital_stay = (
            discharge_date.date() - pd.to_datetime(addmission_date).date()
        ).days + 1  # in days
        curr_timeidxs = []
        for t, fn in enumerate(node_included_files[curr_name]):
            study_date = pd.to_datetime(
                df_demo[df_demo["image_path"] == fn]["StudyDate"].values[0],
                format="%Y%m%d",
            )
            if t > 0:  # first time delta is always 0
                curr_timedelta[t] = (
                    study_date.date() - prev_date.date()
                ).days  # in days
                prev_date = study_date

            time_index = (
                study_date.date() - pd.to_datetime(addmission_date).date()
            ).days  # index starts from 0
            curr_timeidxs.append(time_index)

        assert hospital_stay != np.nan
        assert len(curr_timeidxs) > 0  # because at least one cxr
        total_stay[curr_name] = hospital_stay
        time_idxs[curr_name] = curr_timeidxs

        # normalize by hospital stay
        time_deltas[curr_name] = curr_timedelta / hospital_stay

    return (
        np.array(labels),
        node_included_files,
        label_splits,
        time_deltas,
        total_stay,
        time_idxs,
    )
