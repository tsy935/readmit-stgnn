"""
Because our problem setup is based on a semisupervised node classification setting, all the training/validation/test datapoints' features, ut not labels, are available to the model at training time.

However, in real-world cases, we often want to use a trained model to predict an unseen patient's readmission risk, where unseen test patients' features are not available at training time. 

Here, we simulate this scenario by splitting the original test set into half, where the first half is combined with the train/validation sets to be used to construct the graph for model training, and the second
half are used to evaluate the model (i.e., unseen nodes).

The evaluate the model on the unseen nodes, you simply need to pass the corresponding subset csv file to the dataloader.
"""

import pandas as pd
import os
import numpy as np
import argparse


def main(args):

    df_orig = pd.read_csv(args.demo_file)

    test_patients = df_orig[df_orig["splits"] == "test"]["subject_id"].unique().tolist()
    non_test_idxs = df_orig[~df_orig["subject_id"].isin(test_patients)].index.values

    # split test set into half
    np.random.seed(123)

    np.random.shuffle(test_patients)

    num_unseen = int(len(test_patients) / 2)

    test_patients1 = test_patients[:num_unseen]
    test_patients2 = test_patients[num_unseen:]

    df_test1 = df_orig[df_orig["subject_id"].isin(test_patients1)]
    df_test2 = df_orig[df_orig["subject_id"].isin(test_patients2)]

    # add train and val
    df_test1 = pd.concat([df_orig.iloc[non_test_idxs], df_test1])
    df_test2 = pd.concat([df_orig.iloc[non_test_idxs], df_test2])
    df_test1 = df_test1.reset_index(drop=True)
    df_test2 = df_test2.reset_index(drop=True)

    np.random.seed()

    df_test1.to_csv(
        os.path.join(args.save_dir, "mimic_admission_demo_half_test.csv"), index=False
    )
    df_test2.to_csv(
        os.path.join(args.save_dir, "mimic_admission_demo_unseen_test.csv"), index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Getting unseen nodes.")

    parser.add_argument(
        "--demo_file", type=str, help="Path to csv file containing the full cohort."
    )
    parser.add_argument("--save_dir", type=str, help="Dir to save the generated files.")
    args = parser.parse_args()
    main(args)
