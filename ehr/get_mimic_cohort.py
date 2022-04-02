import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import pickle
import argparse
import sys
from sklearn.model_selection import train_test_split

sys.path.append("../")
from constants import LAB_COLS


ADMISSION_TYPES_EXCLUDED = [
    "AMBULATORY OBSERVATION",
    "EU OBSERVATION",
    "DIRECT OBSERVATION",
]
DISCHARGE_LOCATION_EXCLUDED = [
    "ACUTE HOSPITAL",
    "HEALTHCARE FACILITY",
    # "SKILLED NURSING FACILITY",
    "AGAINST ADVICE",
]
VIEW_POSITIONS_INCLUDED = ["PA", "AP"]


def find_icd_group(df_icd, code):
    group = ""
    letter = code[0]
    number = code[1:].split(".")[0]
    if number.isnumeric():
        number = float(number)
        icd_sel = df_icd.loc[df_icd.SUBGROUP.str.startswith(letter)].copy()
        icd_sel = icd_sel.loc[
            (icd_sel.START_IDX.str.isnumeric()) & (icd_sel.END_IDX.str.isnumeric())
        ].copy()
        icd_sel = icd_sel.loc[
            (icd_sel.START_IDX.astype(float) <= number)
            & (icd_sel.END_IDX.astype(float) >= number)
        ].copy()
        if len(icd_sel) > 0:
            group = icd_sel.at[icd_sel.index[0], "SUBGROUP"]
        else:
            group = "UNKNOWN"
    else:
        icd_sel = df_icd.loc[df_icd.SUBGROUP.str.startswith(letter)].copy()
        icd_sel = icd_sel.loc[
            (icd_sel.START_IDX.str.isnumeric() == False)
            & (icd_sel.END_IDX.str.isnumeric() == False)
        ].copy()
        numheader = number[:-1]
        icd_sel = icd_sel.loc[
            (icd_sel.START_IDX.str.startswith(numheader))
            & (icd_sel.END_IDX.str.startswith(numheader))
        ].copy()
        if len(icd_sel) > 0:
            group = icd_sel.at[icd_sel.index[0], "SUBGROUP"]
        else:
            group = "UNKNOWN"
    return group


def main(args):
    MIMIC_CORE_DIR = os.path.join(args.raw_data_dir, "core")
    MIMIC_HOSP_DIR = os.path.join(args.raw_data_dir, "hosp")

    df_admission = pd.read_csv(os.path.join(MIMIC_CORE_DIR, "admissions.csv.gz"))
    df_admission_orig = df_admission.copy()

    ## cxr metadata
    df_cxr = pd.read_csv(
        os.path.join(args.cxr_data_dir, "mimic-cxr-2.0.0-metadata.csv.gz")
    )

    ## Get CXRs for all admissions
    print("Matching CXRs for admissions...")
    df_admission_w_cxr = {}
    for col in df_admission.columns:
        df_admission_w_cxr[col] = []
    for col in df_cxr.columns:
        df_admission_w_cxr[col] = []

    for _, row in tqdm(df_admission.iterrows(), total=len(df_admission)):
        subject_id = row["subject_id"]
        admit_id = row["hadm_id"]
        admit_time = pd.to_datetime(row["admittime"])
        discharge_time = pd.to_datetime(row["dischtime"])

        curr_cxr = df_cxr[df_cxr["subject_id"] == subject_id]

        for _, row_cxr in curr_cxr.iterrows():
            study_date = pd.to_datetime(row_cxr["StudyDate"], format="%Y%m%d")
            # filter out cxrs not within current hospitalization
            if (study_date < admit_time) or (study_date > discharge_time):
                continue
            for col in df_admission.columns:
                df_admission_w_cxr[col].append(row[col])
            for col in df_cxr.columns:
                if col in df_admission.columns:
                    continue
                df_admission_w_cxr[col].append(row_cxr[col])

    df_admission_w_cxr = pd.DataFrame.from_dict(df_admission_w_cxr)
    df_admission_w_cxr.to_csv(
        os.path.join(args.save_dir, "admission_original_w_cxr.csv"), index=False
    )

    ### Filter out <48hr, certain discharge locations
    print(
        "Filtering out admissions <48hr and certain admission types and discharge locations..."
    )
    row_idxs = []
    for i_row, row in tqdm(
        df_admission_w_cxr.iterrows(), total=len(df_admission_w_cxr)
    ):
        hosp_stay = pd.to_datetime(row["dischtime"]) - pd.to_datetime(row["admittime"])
        admit_type = row["admission_type"]
        discharge_loc = row["discharge_location"]
        admissioin_loc = row["admission_location"]
        if admit_type in ADMISSION_TYPES_EXCLUDED:
            continue
        if discharge_loc in DISCHARGE_LOCATION_EXCLUDED:
            continue
        if isinstance(discharge_loc, float) and np.isnan(discharge_loc):
            continue
        if hosp_stay.days < 2:
            continue

        row_idxs.append(i_row)
    df_admission_w_cxr = df_admission_w_cxr.iloc[np.array(row_idxs)].copy(deep=True)
    df_admission_w_cxr = df_admission_w_cxr.reset_index(drop=True)
    df_admission_w_cxr.to_csv(
        os.path.join(args.save_dir, "admission_48hr_discharge_filtered.csv"),
        index=False,
    )

    ### Filter out non-AP and non-PA views
    print("Filtering out CXR views that are not AP or PA...")
    row_idxs = []
    for i_row, row in tqdm(
        df_admission_w_cxr.iterrows(), total=len(df_admission_w_cxr)
    ):
        view = row["ViewPosition"]
        if view != "AP" and view != "PA":
            continue
        row_idxs.append(i_row)

    df_admit_48hr_cxr_filtered = df_admission_w_cxr.iloc[np.array(row_idxs)].copy(
        deep=True
    )

    df_admit_48hr_cxr_filtered = df_admit_48hr_cxr_filtered.reset_index(drop=True)
    assert len(df_admit_48hr_cxr_filtered["ViewPosition"].unique()) == 2

    ### Keep admissions with >=2 cxrs
    print("Finding admissions with at least 2 CXRs...")
    idxs_to_include = []
    for i_row, row in tqdm(
        df_admit_48hr_cxr_filtered.iterrows(), total=len(df_admit_48hr_cxr_filtered)
    ):
        curr_admit_df = df_admit_48hr_cxr_filtered[
            df_admit_48hr_cxr_filtered["hadm_id"] == row["hadm_id"]
        ]
        study_ids = curr_admit_df["study_id"].unique()
        if len(study_ids) >= 2:
            idxs_to_include.append(i_row)
    df_admit_48hr_cxr_filtered = df_admit_48hr_cxr_filtered.iloc[
        np.array(idxs_to_include)
    ].copy(deep=True)
    df_admit_48hr_cxr_filtered = df_admit_48hr_cxr_filtered.reset_index(drop=True)

    ## Find absolute paths of cxrs
    print("Matching CXR file paths...")
    df_cxr_split = pd.read_csv(
        os.path.join(args.cxr_data_dir, "mimic-cxr-2.0.0-split.csv.gz")
    )
    dicom2fullpath = {}

    cxr_paths = []
    for _, row in tqdm(
        df_admit_48hr_cxr_filtered.iterrows(), total=len(df_admit_48hr_cxr_filtered)
    ):
        dicom_id = row["dicom_id"]
        study_id = df_cxr_split[df_cxr_split["dicom_id"] == dicom_id][
            "study_id"
        ].values[0]
        subject_id = row["subject_id"]

        subdir = "p" + str(subject_id)[:2]
        path = os.path.join(
            args.cxr_data_dir,
            "files",
            subdir,
            "p" + str(subject_id),
            "s" + str(study_id),
            dicom_id + ".jpg",
        )
        # assert os.path.exists(path)
        cxr_paths.append(path)
        dicom2fullpath[dicom_id] = path

    df_admit_48hr_cxr_filtered["image_path"] = cxr_paths
    df_admit_48hr_cxr_filtered.to_csv(
        os.path.join(args.save_dir, "admission_48hr_discharge_cxr_filtered.csv"),
        index=False,
    )

    ### Get readmission info
    print("Getting readmission information...")
    readmission_gap_in_days = []
    readmission_id = []
    readmitted_within_30 = []
    for _, row in tqdm(
        df_admit_48hr_cxr_filtered.iterrows(), total=len(df_admit_48hr_cxr_filtered)
    ):
        subject_id = row["subject_id"]
        admit_id = row["hadm_id"]
        admit_time = pd.to_datetime(row["admittime"])
        discharge_time = pd.to_datetime(row["dischtime"])

        pat_admissions = df_admission_orig[
            df_admission_orig["subject_id"] == subject_id
        ]

        pat_admissions = df_admit_48hr_cxr_filtered[
            df_admit_48hr_cxr_filtered["subject_id"] == subject_id
        ]

        # sort current patient's all admissions by admittime
        pat_admissions = pat_admissions.sort_values(by=["admittime"], ascending=True)
        readmit_gap = None
        readmit_id = None
        for _, curr_row in pat_admissions.iterrows():
            if curr_row["admission_type"] in ADMISSION_TYPES_EXCLUDED:
                continue

            curr_admit_time = pd.to_datetime(curr_row["admittime"])
            curr_discharge_time = pd.to_datetime(curr_row["dischtime"])
            if (curr_discharge_time - curr_admit_time).days < 2:
                continue
            if pd.to_datetime(curr_row["admittime"]) > discharge_time:
                readmit_gap = (
                    pd.to_datetime(curr_row["admittime"]) - discharge_time
                ).days
                readmit_id = int(curr_row["hadm_id"])
                break

        if readmit_gap is None:
            readmission_gap_in_days.append(np.nan)
            readmission_id.append(np.nan)

            if not (
                isinstance(row["deathtime"], float) and np.isnan(row["deathtime"])
            ):  # died within hospital, treat as readmitted
                readmitted_within_30.append("True")
            elif row["discharge_location"] == "DIED":
                readmitted_within_30.append("True")
            else:
                readmitted_within_30.append("False")
        else:
            readmission_gap_in_days.append(readmit_gap)
            readmission_id.append(readmit_id)
            if readmit_gap <= 30:
                readmitted_within_30.append("True")
            else:
                readmitted_within_30.append("False")

    df_admit_48hr_cxr_filtered["readmission_gap_in_days"] = readmission_gap_in_days
    df_admit_48hr_cxr_filtered["readmission_id"] = readmission_id
    df_admit_48hr_cxr_filtered["readmitted_within_30days"] = readmitted_within_30
    df_admit_48hr_cxr_filtered.to_csv(
        os.path.join(
            args.save_dir, "admission_48hr_discharge_cxr_filtered_labeled.csv"
        ),
        index=False,
    )

    ### Split patients into train/val/test
    all_patients = list(set(df_admit_48hr_cxr_filtered["subject_id"].tolist()))
    train_val_patients, test_patients = train_test_split(
        all_patients, test_size=0.2, random_state=12
    )
    train_patients, val_patients = train_test_split(
        train_val_patients, test_size=0.2, random_state=12
    )
    df_train = df_admit_48hr_cxr_filtered[
        df_admit_48hr_cxr_filtered["subject_id"].isin(train_patients)
    ][["hadm_id", "readmitted_within_30days"]].drop_duplicates()
    df_val = df_admit_48hr_cxr_filtered[
        df_admit_48hr_cxr_filtered["subject_id"].isin(val_patients)
    ][["hadm_id", "readmitted_within_30days"]].drop_duplicates()
    df_test = df_admit_48hr_cxr_filtered[
        df_admit_48hr_cxr_filtered["subject_id"].isin(test_patients)
    ][["hadm_id", "readmitted_within_30days"]].drop_duplicates()
    print(
        "Train pos ratio:",
        (df_train["readmitted_within_30days"] == "True").sum() / len(df_train),
    )
    print(
        "Val pos ratio:",
        (df_val["readmitted_within_30days"] == "True").sum() / len(df_val),
    )
    print(
        "Test pos ratio:",
        (df_test["readmitted_within_30days"] == "True").sum() / len(df_test),
    )

    splits = []
    for _, row in tqdm(
        df_admit_48hr_cxr_filtered.iterrows(), total=len(df_admit_48hr_cxr_filtered)
    ):
        subject_id = row["subject_id"]
        if subject_id in train_patients:
            splits.append("train")
        elif subject_id in val_patients:
            splits.append("val")
        elif subject_id in test_patients:
            splits.append("test")
        else:
            raise ValueError
    df_admit_48hr_cxr_filtered["splits"] = splits

    df_admit_48hr_cxr_filtered.to_csv(
        os.path.join(
            args.save_dir, "admission_48hr_discharge_cxr_filtered_w_splits.csv"
        ),
        index=False,
    )

    ### Get demographics
    df_patients = pd.read_csv(os.path.join(MIMIC_CORE_DIR, "patients.csv.gz"))

    print("Getting age, gender, ethnicity, splits...")
    ages = []
    genders = []
    ethnicity = []
    for _, row in tqdm(
        df_admit_48hr_cxr_filtered.iterrows(), total=len(df_admit_48hr_cxr_filtered)
    ):
        subject_id = row["subject_id"]
        admit_id = row["hadm_id"]
        admit_time = pd.to_datetime(row["admittime"])

        age_anchor = df_patients[df_patients["subject_id"] == subject_id][
            "anchor_age"
        ].values[0]
        anchor_yr = df_patients[df_patients["subject_id"] == subject_id][
            "anchor_year"
        ].values[0]

        gender = df_patients[df_patients["subject_id"] == subject_id]["gender"].values[
            0
        ]

        eth = df_admission[df_admission["subject_id"] == subject_id][
            "ethnicity"
        ].values[0]

        age_at_admit = (int(admit_time.year) - int(anchor_yr)) + int(age_anchor)

        ages.append(age_at_admit)
        genders.append(gender)
        ethnicity.append(eth)

    df_admit_48hr_cxr_filtered["age"] = ages
    df_admit_48hr_cxr_filtered["gender"] = genders
    df_admit_48hr_cxr_filtered["ethnicity"] = ethnicity
    df_admit_48hr_cxr_filtered.loc[
        df_admit_48hr_cxr_filtered["ethnicity"] == "UNABLE TO OBTAIN", "ethnicity"
    ] = "UNKNOWN"

    df_admit_48hr_cxr_filtered.to_csv(
        os.path.join(args.save_dir, "mimic_admission_demo.csv"), index=False
    )
    print("Admission basic information saved...")

    df_admission_list = (
        df_admit_48hr_cxr_filtered[["hadm_id", "subject_id"]].copy().drop_duplicates()
    )

    ### Medication
    df_presb = pd.read_csv(os.path.join(MIMIC_HOSP_DIR, "prescriptions.csv.gz"))
    df_presb.head()

    df_presb_filtered = df_presb[
        df_presb["subject_id"].isin(df_admit_48hr_cxr_filtered["subject_id"].tolist())
    ]
    df_presb_filtered = df_presb_filtered[
        df_presb_filtered["hadm_id"].isin(df_admission_list["hadm_id"].tolist())
    ]

    ## Map NDC to therapeutic classes
    df_med_map = pd.read_csv("../data/ndc2therapeutic.csv")

    idxs_to_include = []
    med_classes = []
    df_presb_filtered = df_presb_filtered.reset_index(drop=True)
    print("Mapping NDC to therapeutic classes...")
    for i_row, row in tqdm(df_presb_filtered.iterrows(), total=len(df_presb_filtered)):
        ndc = row["ndc"]
        starttime = row["starttime"]
        if isinstance(starttime, float) and np.isnan(starttime):
            continue
        med_class = df_med_map[df_med_map["NDC_MEDICATION_CODE"] == ndc][
            "MED_THERAPEUTIC_CLASS_DESCRIPTION"
        ]
        if len(med_class) > 0:
            # assert len(med_class) == 1
            med_class = list(set(med_class.values))
            med_class = [c for c in med_class if isinstance(c, str)]
            # print(med_class)
            if len(med_class) > 1:
                print(med_class)
            if len(med_class) == 0:  # only nan
                continue
            idxs_to_include.append(i_row)
            med_classes.append(med_class[0])

    df_presb_filtered_mapped = df_presb_filtered.iloc[np.array(idxs_to_include)].copy(
        deep=True
    )
    df_presb_filtered_mapped = df_presb_filtered_mapped.reset_index(drop=True)
    df_presb_filtered_mapped["MED_THERAPEUTIC_CLASS_DESCRIPTION"] = med_classes

    ## Add day number
    med_day_numbers = []
    for _, row in tqdm(
        df_presb_filtered_mapped.iterrows(), total=len(df_presb_filtered_mapped)
    ):
        if isinstance(starttime, float) and np.isnan(starttime):
            continue
        starttime = pd.to_datetime(row["starttime"])

        admit_id = row["hadm_id"]
        subject_id = row["subject_id"]

        admittime = pd.to_datetime(
            df_admit_48hr_cxr_filtered[
                df_admit_48hr_cxr_filtered["hadm_id"] == admit_id
            ]["admittime"].values[0]
        )
        dischargetime = pd.to_datetime(
            df_admit_48hr_cxr_filtered[
                df_admit_48hr_cxr_filtered["hadm_id"] == admit_id
            ]["dischtime"].values[0]
        )

        if starttime > dischargetime or starttime < admittime:
            med_day_numbers.append(np.nan)
            continue

        day_num = (starttime.date() - admittime.date()).days + 1
        med_day_numbers.append(day_num)

    df_presb_filtered_mapped["Day_Number"] = med_day_numbers
    df_presb_filtered_mapped = df_presb_filtered_mapped[
        ~df_presb_filtered_mapped["Day_Number"].isnull()
    ]

    df_presb_filtered_mapped.to_csv(
        os.path.join(args.save_dir, "mimic_hosp_med_filtered.csv"), index=False
    )

    ### ICD-10
    df_diag = pd.read_csv(os.path.join(MIMIC_HOSP_DIR, "diagnoses_icd.csv.gz"))

    df_diag_table = pd.read_csv(os.path.join(MIMIC_HOSP_DIR, "d_icd_procedures.csv.gz"))

    df_icd = pd.read_csv("../data/ICD10_Groups.csv")

    df_diag_subgroup = {k: [] for k in df_diag.columns}
    df_diag_subgroup["SUBGROUP"] = []

    print("Mapping ICD-10 code to subgroups...")
    for _, row in tqdm(df_admission_list.iterrows(), total=len(df_admission_list)):
        subject_id = row["subject_id"]
        admit_id = row["hadm_id"]

        curr_diag = df_diag[df_diag["subject_id"] == subject_id]
        curr_diag = curr_diag[curr_diag["hadm_id"] == admit_id]

        for _, diag_row in curr_diag.iterrows():
            code = str(diag_row["icd_code"])
            version = int(diag_row["icd_version"])
            if version != 10:
                continue
            subgroup = find_icd_group(df_icd, code)

            if subgroup == "" or subgroup == "UNKNOWN":
                continue
            else:
                df_diag_subgroup["SUBGROUP"].append(subgroup)
                for col in df_diag.columns:
                    df_diag_subgroup[col].append(diag_row[col])

    df_diag_subgroup = pd.DataFrame.from_dict(df_diag_subgroup)
    df_diag_subgroup.to_csv(
        os.path.join(args.save_dir, "mimic_hosp_icd_subgroups.csv"), index=False
    )

    ### Get labs
    df_lab_item = pd.read_csv(os.path.join(MIMIC_HOSP_DIR, "d_labitems.csv.gz"))
    lab_item_idxs = []
    for i_row, row in df_lab_item.iterrows():
        label = row["label"]
        if isinstance(label, str) and label != " ":
            fluid = row["fluid"]
            if (str(label) + " " + str(fluid)) in LAB_COLS:
                lab_item_idxs.append(i_row)

    df_lab_item_filtered = df_lab_item.iloc[np.array(lab_item_idxs)].copy(deep=True)
    df_lab_item_filtered = df_lab_item_filtered.reset_index(drop=True)

    ## labevents file is big, process using chunks
    print("Reading lab events...")
    df_lab_filtered = []
    with pd.read_csv(
        os.path.join(MIMIC_HOSP_DIR, "labevents.csv.gz"), chunksize=1000
    ) as reader:
        for chunk in reader:
            chunk = chunk[
                chunk["hadm_id"].isin(df_admit_48hr_cxr_filtered["hadm_id"].tolist())
            ]
            chunk = chunk[chunk["itemid"].isin(df_lab_item_filtered["itemid"].tolist())]
            df_lab_filtered.append(chunk)

    df_lab_filtered = pd.concat(df_lab_filtered)

    ## Add lab name by label + fluids, and day number
    label_fluids = []
    categories = []
    day_numbers = []
    print("Getting lab information...")
    for _, row in tqdm(df_lab_filtered.iterrows(), total=len(df_lab_filtered)):
        itemid = row["itemid"]
        admit_id = row["hadm_id"]

        label = df_lab_item_filtered[df_lab_item_filtered["itemid"] == itemid][
            "label"
        ].values[0]
        fluid = df_lab_item_filtered[df_lab_item_filtered["itemid"] == itemid][
            "fluid"
        ].values[0]
        cat = df_lab_item_filtered[df_lab_item_filtered["itemid"] == itemid][
            "category"
        ].values[0]
        label_fluids.append(str(label) + " " + str(fluid))
        categories.append(cat)

        # day number
        charttime = pd.to_datetime(row["charttime"])
        admittime = pd.to_datetime(
            df_admit_48hr_cxr_filtered[
                df_admit_48hr_cxr_filtered["hadm_id"] == admit_id
            ]["admittime"].values[0]
        )
        dischargetime = pd.to_datetime(
            df_admit_48hr_cxr_filtered[
                df_admit_48hr_cxr_filtered["hadm_id"] == admit_id
            ]["dischtime"].values[0]
        )

        if charttime > dischargetime or charttime < admittime:
            day_numbers.append(np.nan)
            continue

        day_num = (charttime.date() - admittime.date()).days + 1  # starts from 1
        day_numbers.append(day_num)

    df_lab_filtered["label_fluid"] = label_fluids
    df_lab_filtered["category"] = categories
    df_lab_filtered["Day_Number"] = day_numbers

    df_lab_filtered = df_lab_filtered[~df_lab_filtered["Day_Number"].isnull()]

    df_lab_filtered.to_csv(
        os.path.join(args.save_dir, "mimic_hosp_lab_filtered.csv"), index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filtering admission info from MIMIC-IV."
    )

    parser.add_argument(
        "--raw_data_dir", type=str, help="Dir to downloaded MIMIC-IV data."
    )
    parser.add_argument(
        "--cxr_data_dir", type=str, help="Dir to downloaded MIMIC-CXR-JPG data."
    )
    parser.add_argument(
        "--save_dir", type=str, help="Dir to save filtered cohort files."
    )
    args = parser.parse_args()
    main(args)
