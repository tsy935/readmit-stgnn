import pandas as pd
import numpy as np
import os
import pickle
import copy
import argparse
from tqdm import tqdm
import sys

sys.path.append("../")
from constants import DEMO_COLS, LAB_COLS
from data.readmission_utils import get_readmission_label_mimic

from collections import Counter
from sklearn.preprocessing import LabelEncoder


COLS_IRRELEVANT = [
    "subject_id",
    "hadm_id",
    "admittime",
    "dischtime",
    "splits",
    "date",
    "node_name",
]
CAT_COLUMNS = LAB_COLS + ["gender", "ethnicity"]

SUBGOUPRS_EXCLUDED = [
    "Z00-Z13",
    "Z14-Z15",
    "Z16-Z16",
    "Z17-Z17",
    "Z18-Z18",
    "Z19-Z19",
    "Z20-Z29",
    "Z30-Z39",
    "Z40-Z53",
    "Z55-Z65",
    "Z66-Z66",
    "Z67-Z67",
    "Z68-Z68",
    "Z69-Z76",
    "Z77-Z99",
]


def ehr_bag_of_words_mimic(
    df_demo, df_ehr, col_name, time_step_by="day", filter_freq=None
):
    """
    Get EHR sequence using naive bag-of-words method
    Args:
        df_demo: demographics dataframe
        df_ehr: CPT/ICD dataframe
        ehr_type: 'cpt' or 'icd'
        time_step_by: 'day', what is the time step size?
    Returns:
        ehr_seq_padded: shape (num_admissions, max_seq_len, num_ehr_subgroups),
            short sequences are padded with -1
    """

    all_values = list(set(df_ehr[col_name]))
    all_values = [
        val
        for val in all_values
        if isinstance(val, str) and (val not in SUBGOUPRS_EXCLUDED)
    ]

    df_ehr_count = {
        "subject_id": [],
        "hadm_id": [],
        "admittime": [],
        "dischtime": [],
        "date": [],
        "target": [],
        "node_name": [],
        "splits": [],
    }
    initial_cols = len(df_ehr_count) + len(DEMO_COLS)
    # add demographic columns
    for demo in DEMO_COLS:
        df_ehr_count[demo] = []

    # add subgroup columns
    for subgrp in all_values:
        df_ehr_count[subgrp] = []

    for _, row in tqdm(df_demo.iterrows(), total=len(df_demo)):
        pat = row["subject_id"]
        admit_id = row["hadm_id"]
        admit_dt = row["admittime"]
        discharge_dt = row["dischtime"]
        label = row["readmitted_within_30days"]

        if (str(pat) + "_" + str(admit_id)) in df_ehr_count["node_name"]:
            continue

        if time_step_by == "day":
            dt_range = pd.date_range(
                start=pd.to_datetime(admit_dt).date(),
                end=pd.to_datetime(discharge_dt).date(),
            )  # both inclusive
        else:
            raise NotImplementedError
        assert len(dt_range) > 1

        curr_ehr_df = df_ehr[df_ehr["hadm_id"] == admit_id]

        for dt in dt_range:
            day_num = (dt.date() - pd.to_datetime(admit_dt).date()).days + 1

            if "charttime" in df_ehr.columns:
                curr_day_ehrs = curr_ehr_df[
                    curr_ehr_df["Day_Number"] == float(day_num)
                ][col_name]
            else:
                # not time-varying, i.e., diagnoses ICD code
                curr_day_ehrs = curr_ehr_df[col_name]

            df_ehr_count["subject_id"].append(pat)
            df_ehr_count["hadm_id"].append(admit_id)
            df_ehr_count["admittime"].append(admit_dt)
            df_ehr_count["dischtime"].append(discharge_dt)
            df_ehr_count["date"].append(str(dt))
            df_ehr_count["target"].append(label)
            df_ehr_count["splits"].append(row["splits"])
            df_ehr_count["node_name"].append(str(pat) + "_" + str(admit_id))

            for demo in DEMO_COLS:
                if (demo in CAT_COLUMNS) and isinstance(row[demo], float):  # nan
                    df_ehr_count[demo].append("UNKNOWN")
                else:
                    df_ehr_count[demo].append(row[demo])

            if len(curr_day_ehrs) > 0:
                ehr_counts = Counter(curr_day_ehrs)
                for subgrp in all_values:
                    if subgrp in ehr_counts.keys():
                        df_ehr_count[subgrp].append(ehr_counts[subgrp])
                    else:
                        df_ehr_count[subgrp].append(0)
            else:
                for subgrp in all_values:
                    df_ehr_count[subgrp].append(0)

    df_ehr_count = pd.DataFrame.from_dict(df_ehr_count)

    # drop zero occurrence subgroups
    if filter_freq is not None:
        freq = df_ehr_count[all_values].sum(axis=0)
        drop_col_idxs = freq.values < filter_freq
        df_ehr_count = df_ehr_count.drop(columns=freq.loc[drop_col_idxs].index)
    else:
        freq = df_ehr_count[all_values].sum(axis=0)
        drop_col_idxs = freq.values == 0
        df_ehr_count = df_ehr_count.drop(columns=freq.loc[drop_col_idxs].index)

    print("Final subgroups:", len(df_ehr_count.columns) - initial_cols)

    return df_ehr_count


def lab_one_hot_mimic(df_demo, df_lab, col_name, time_step_by="day", filter_freq=None):
    """
    Get EHR sequence using naive bag-of-words method
    Args:
        df_demo: demographics dataframe
        df_ehr: CPT/ICD dataframe
        ehr_type: 'cpt' or 'icd'
        time_step_by: 'day', what is the time step size?
    Returns:
        ehr_seq_padded: shape (num_admissions, max_seq_len, num_ehr_subgroups),
            short sequences are padded with -1
    """

    lab_cols = list(set(df_lab[col_name]))
    lab_cols = [col for col in lab_cols if isinstance(col, str)]

    df_lab_onehot = {
        "subject_id": [],
        "hadm_id": [],
        "admittime": [],
        "dischtime": [],
        "date": [],
        "target": [],
        "node_name": [],
        "splits": [],
    }
    initial_cols = len(df_lab_onehot) + len(DEMO_COLS)
    # add demographic columns
    for demo in DEMO_COLS:
        df_lab_onehot[demo] = []

    # add subgroup columns
    for col in lab_cols:
        df_lab_onehot[col] = []

    for _, row in tqdm(df_demo.iterrows(), total=len(df_demo)):
        pat = row["subject_id"]
        admit_id = row["hadm_id"]
        admit_dt = row["admittime"]
        discharge_dt = row["dischtime"]
        label = row["readmitted_within_30days"]

        if (str(pat) + "_" + str(admit_id)) in df_lab_onehot["node_name"]:
            continue

        if time_step_by == "day":
            dt_range = pd.date_range(
                start=pd.to_datetime(admit_dt).date(),
                end=pd.to_datetime(discharge_dt).date(),
            )  # both inclusive
        else:
            raise NotImplementedError
        assert len(dt_range) > 1

        curr_ehr_df = df_lab[df_lab["hadm_id"] == admit_id]

        for dt in dt_range:
            day_num = (dt.date() - pd.to_datetime(admit_dt).date()).days + 1
            curr_day_lab = curr_ehr_df[curr_ehr_df["Day_Number"] == float(day_num)]

            df_lab_onehot["subject_id"].append(pat)
            df_lab_onehot["hadm_id"].append(admit_id)
            df_lab_onehot["admittime"].append(admit_dt)
            df_lab_onehot["dischtime"].append(discharge_dt)
            df_lab_onehot["date"].append(str(dt))
            df_lab_onehot["target"].append(label)
            df_lab_onehot["splits"].append(row["splits"])
            df_lab_onehot["node_name"].append(str(pat) + "_" + str(admit_id))

            for demo in DEMO_COLS:
                if (demo in CAT_COLUMNS) and isinstance(row[demo], float):  # nan
                    df_lab_onehot[demo].append("UNKNOWN")
                else:
                    df_lab_onehot[demo].append(row[demo])

            for lab in lab_cols:
                if len(curr_day_lab) == 0:
                    df_lab_onehot[lab].append("nan")
                else:
                    if (
                        curr_day_lab.loc[curr_day_lab[col_name] == lab, "flag"]
                        == "abnormal"
                    ).any():
                        df_lab_onehot[lab].append("abnormal")
                    else:
                        df_lab_onehot[lab].append("nan")

    df_lab_onehot = pd.DataFrame.from_dict(df_lab_onehot)

    # drop zero abnormal subgroups
    if filter_freq is not None:
        freq = (df_lab_onehot[lab_cols] == "abnormal").sum(
            axis=0
        )  # number of abnormals
        drop_col_idxs = freq.values < filter_freq
        df_lab_onehot = df_lab_onehot.drop(columns=freq.loc[drop_col_idxs].index)
    else:
        freq = (df_lab_onehot[lab_cols] == "abnormal").sum(
            axis=0
        )  # number of abnormals
        drop_col_idxs = freq.values == 0
        df_lab_onehot = df_lab_onehot.drop(columns=freq.loc[drop_col_idxs].index)

    print("Final labs:", len(df_lab_onehot.columns) - initial_cols)

    return df_lab_onehot


def preproc_ehr_cat_embedding(X):

    train_indices = X[X["splits"] == "train"].index
    target = "target"

    types = X.dtypes

    # encode categorical variables
    categorical_columns = []
    categorical_dims = {}
    for col in tqdm(X.columns):
        if col in COLS_IRRELEVANT:
            continue
        if col in CAT_COLUMNS:
            l_enc = LabelEncoder()
            X[col] = X[col].fillna("VV_likely")
            X[col] = X[col].replace(
                {0.23990602999011235: "UNKNOWN"}
            )  # TODO: confirm if removing this works
            print(col, X[col].unique())
            X[col] = l_enc.fit_transform(X[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = len(l_enc.classes_)
        else:
            print(col)
            X.fillna(X.loc[train_indices, col].mean(), inplace=True)

    feature_cols = [
        col for col in X.columns if (col != target) and (col not in COLS_IRRELEVANT)
    ]
    cat_idxs = [i for i, f in enumerate(feature_cols) if f in categorical_columns]
    cat_dims = [
        categorical_dims[f]
        for i, f in enumerate(feature_cols)
        if f in categorical_columns
    ]

    return {
        "X": X,
        "feature_cols": feature_cols,
        "cat_idxs": cat_idxs,
        "cat_dims": cat_dims,
    }


def preproc_ehr(X):
    """
    Args:
        X: pandas dataframe
    Returns:
        X_enc: pandas dataframe, with one-hot encoded columns for categorical variables
    """
    train_indices = X[X["splits"] == "train"].index

    # encode categorical variables
    X_enc = []
    num_cols = 0
    categorical_columns = []
    categorical_dims = {}
    for col in tqdm(X.columns):
        if col in COLS_IRRELEVANT:
            X_enc.append(X[col])
            num_cols += 1
        elif col in CAT_COLUMNS:
            print(col, X[col].unique())
            curr_enc = pd.get_dummies(
                X[col], prefix=col
            )  # this will transform into one-hot encoder
            X_enc.append(curr_enc)
            num_cols += curr_enc.shape[-1]
            categorical_columns.append(col)
            categorical_dims[col] = curr_enc.shape[-1]
        else:
            X.fillna(X.loc[train_indices, col].mean(), inplace=True)
            curr_enc = X[col]
            X_enc.append(curr_enc)
            num_cols += 1

    X_enc = pd.concat(X_enc, axis=1)
    assert num_cols == X_enc.shape[-1]

    feature_cols = [
        col
        for col in X_enc.columns
        if (col != "target") and (col not in COLS_IRRELEVANT)
    ]
    cat_idxs = [i for i, f in enumerate(feature_cols) if f in categorical_columns]
    cat_dims = [
        categorical_dims[f]
        for _, f in enumerate(feature_cols)
        if f in categorical_columns
    ]

    return {
        "X": X_enc,
        "feature_cols": feature_cols,
        "cat_idxs": cat_idxs,
        "cat_dims": cat_dims,
    }


def ehr2sequence(preproc_dict, df_demo, by="day"):
    """
    Arrange EHR into sequences for temporal models
    """

    X = preproc_dict["X"]
    df = copy.deepcopy(X)
    feature_cols = preproc_dict["feature_cols"]

    print("Rearranging to sequences by {}...".format(by))
    X = X[feature_cols].values

    X_dict = {}
    for i in range(X.shape[0]):
        key = (
            str(df.iloc[i]["subject_id"])
            + "_"
            + str(pd.to_datetime(df.iloc[i]["date"]).date())
        )
        X_dict[key] = X[i]

    _, node_included_files, _, _, _, _ = get_readmission_label_mimic(
        df_demo, max_seq_len=None
    )

    # arrange X by day or by cxr
    feat_dict = {}
    for node_name, _ in tqdm(node_included_files.items()):
        # print(node_name)
        ehr_row = df[df["node_name"] == node_name]
        curr_admit = ehr_row["admittime"].values[0]
        curr_discharge = ehr_row["dischtime"].values[0]
        curr_pat = ehr_row["subject_id"].values[0]

        if by == "day":
            dt_range = pd.date_range(
                start=pd.to_datetime(curr_admit).date(),
                end=pd.to_datetime(curr_discharge).date(),
            )
        else:
            raise NotImplementedError

        curr_features = []
        for dt in dt_range:
            if by == "cxr":
                key = str(curr_pat) + "_" + str(dt)
            else:
                key = str(curr_pat) + "_" + str(dt.date())
            feat = X_dict[key]
            curr_features.append(feat)

        curr_features = np.stack(curr_features)  # (num_days, feature_dim)
        feat_dict[node_name] = curr_features

    if "cat_idxs" in preproc_dict:
        cat_idxs = preproc_dict["cat_idxs"]
        cat_dims = preproc_dict["cat_dims"]
        return {
            "feat_dict": feat_dict,
            "feature_cols": feature_cols,
            "cat_idxs": cat_idxs,
            "cat_dims": cat_dims,
        }
    else:
        return {"feat_dict": feat_dict, "feature_cols": feature_cols}


def main(args):
    # read csv files
    df_demo = pd.read_csv(
        args.demo_file, dtype={k: str for k in CAT_COLUMNS}, low_memory=False
    )
    df_lab = pd.read_csv(args.lab_file, dtype={"flag": str}, low_memory=False)
    df_icd = pd.read_csv(args.icd_file, low_memory=False)
    df_med = pd.read_csv(args.med_file, low_memory=False)

    # # TODO, remove this, for debugging only!
    # df_demo = df_demo.iloc[:1000].copy()

    # icd
    df_icd_count = ehr_bag_of_words_mimic(
        df_demo, df_icd, col_name="SUBGROUP", time_step_by="day", filter_freq=None
    )

    # lab
    df_lab_onehot = lab_one_hot_mimic(
        df_demo, df_lab, col_name="label_fluid", time_step_by="day", filter_freq=None
    )

    # medication
    df_med_count = ehr_bag_of_words_mimic(
        df_demo,
        df_med,
        col_name="MED_THERAPEUTIC_CLASS_DESCRIPTION",
        time_step_by="day",
        filter_freq=None,
    )

    # combine
    df_combined = pd.concat([df_icd_count, df_lab_onehot, df_med_count], axis=1)

    # drop duplicated columns
    df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]
    df_combined.to_csv(os.path.join(args.save_dir, "ehr_combined.csv"), index=False)

    for format in ["cat_embedding", "one_hot"]:
        if format == "cat_embedding":
            preproc_dict = preproc_ehr_cat_embedding(df_combined)
        else:
            preproc_dict = preproc_ehr(df_combined)

        feature_cols = preproc_dict["feature_cols"]
        demo_cols = [
            col for col in feature_cols if any([s for s in DEMO_COLS if s in col])
        ]
        icd_cols = [
            col for col in feature_cols if col in list(set(df_icd["SUBGROUP"].tolist()))
        ]
        lab_cols = [
            col for col in feature_cols if any([s for s in LAB_COLS if s in col])
        ]
        med_cols = [
            col
            for col in feature_cols
            if col in list(set(df_med["MED_THERAPEUTIC_CLASS_DESCRIPTION"].tolist()))
        ]

        preproc_dict["demo_cols"] = demo_cols
        preproc_dict["icd_cols"] = icd_cols
        preproc_dict["lab_cols"] = lab_cols
        preproc_dict["med_cols"] = med_cols

        # save
        with open(
            os.path.join(args.save_dir, "ehr_preprocessed_all_{}.pkl".format(format)),
            "wb",
        ) as pf:
            pickle.dump(preproc_dict, pf)
        print(
            "Saved to {}".format(
                os.path.join(
                    args.save_dir, "ehr_preprocessed_all_{}.pkl".format(format)
                )
            )
        )

        # also save it into sequences for temporal models
        seq_dict = ehr2sequence(preproc_dict, df_demo, by="day")

        seq_dict["demo_cols"] = demo_cols
        seq_dict["icd_cols"] = icd_cols
        seq_dict["lab_cols"] = lab_cols
        seq_dict["med_cols"] = med_cols
        with open(
            os.path.join(
                args.save_dir, "ehr_preprocessed_seq_by_day_{}.pkl".format(format)
            ),
            "wb",
        ) as pf:
            pickle.dump(seq_dict, pf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing EHR.")

    parser.add_argument(
        "--demo_file",
        type=str,
        default=None,
        help="Dir to filtered cohort demographics file.",
    )
    parser.add_argument(
        "--icd_file",
        type=str,
        default=None,
        help="Dir to filtered cohort ICD-10 file.",
    )
    parser.add_argument(
        "--lab_file",
        type=str,
        default=None,
        help="Dir to filtered cohort lab file.",
    )
    parser.add_argument(
        "--med_file",
        type=str,
        default=None,
        help="Dir to filtered cohort medication file.",
    )
    parser.add_argument(
        "--save_dir", type=str, default=None, help="Dir to save preprocessed files."
    )

    args = parser.parse_args()
    main(args)
