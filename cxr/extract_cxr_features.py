import torchvision.transforms as transforms

from dotted_dict import DottedDict
import pickle
import argparse
import json
import pandas as pd
from tqdm import tqdm
import requests
from pathlib import Path
import torchvision.models as models
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import os
import torch.nn as nn
import torch
import sys
import numpy as np

CXR_MEAN = [0.5004, 0.5004, 0.5004]
CXR_STD = [0.2881, 0.2881, 0.2881]


def get_transform(args, training, standardize=True, rgb=True):
    """
    Source: https://github.com/stanfordmlgroup/MoCo-CXR/blob/main/moco_pretraining/moco/aihc_utils/image_transform.py"""
    # Shorter side scaled to args.img_size
    if args.maintain_ratio:
        transforms_list = [transforms.Resize(args.img_size)]
    else:
        transforms_list = [transforms.Resize((args.img_size, args.img_size))]

    # Data augmentation
    if training:
        transforms_list += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(args.rotate),
            transforms.RandomCrop((args.crop, args.crop)) if args.crop != 0 else None,
        ]
    else:
        transforms_list += [
            transforms.CenterCrop((args.crop, args.crop)) if args.crop else None
        ]

    transforms_list += [transforms.ToTensor()]

    # Normalization
    if standardize:
        if rgb:
            normalize = transforms.Normalize(mean=CXR_MEAN, std=CXR_STD)
        else:
            normalize = transforms.Normalize(mean=[CXR_MEAN[0]], std=[CXR_STD[0]])
        transforms_list += [normalize]

    transform = [t for t in transforms_list if t]
    return transform


class CXRDatasetMoCo(Dataset):
    def __init__(self, img_dirs, transform):
        self.img_dirs = img_dirs
        self.transform = transform

    def __len__(self):
        return len(self.img_dirs)

    def __getitem__(self, idx):
        image = Image.open(self.img_dirs[idx])
        image = image.convert("RGB")

        q = self.transform(image)
        k = self.transform(image)

        return q, k, self.img_dirs[idx]


def load_cxrs_moco_single_split(
    batch_size,
    transform,
    csv_file,
    file_col_name,
    num_workers=8,
):

    df = pd.read_csv(csv_file)
    img_dirs = df[file_col_name].tolist()

    dataset = CXRDatasetMoCo(
        img_dirs,
        transform=transform,
    )

    print("Default to shuffling...")
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    return dataloader


def main(args):

    checkpoint = torch.load(args.model_checkpoint_dir)

    args_pretrained = {
        "moco_dim": 128,
        "img_size": 320,
        "maintain_ratio": True,
        "crop": 320,
        "arch": "densenet121",
    }
    args_pretrained = DottedDict(args_pretrained)

    model = models.__dict__[checkpoint["arch"]](num_classes=args_pretrained.moco_dim)

    # rename moco pretrained weights
    # https://github.com/facebookresearch/moco/blob/master/main_lincls.py#L161
    state_dict = checkpoint["state_dict"]
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if "module" not in k:
            if k.startswith("encoder_q") and not k.startswith("encoder_q.fc"):
                # remove prefix
                state_dict[k[len("encoder_q.") :]] = state_dict[k]
        else:
            if k.startswith("module.encoder_q") and not k.startswith(
                "module.encoder_q.fc"
            ):
                # remove prefix
                state_dict[k[len("module.encoder_q.") :]] = state_dict[k]

    # delete renamed or unused k
    del state_dict[k]

    model.load_state_dict(state_dict, strict=False)

    encoder = nn.Sequential(*list(model.children())[:-1])

    args_dataloader = {
        "img_size": args_pretrained.img_size,
        "maintain_ratio": args_pretrained.maintain_ratio,
        "crop": args_pretrained.crop,
    }
    args_dataloader = DottedDict(args_dataloader)

    aug = get_transform(args_dataloader, training=False)
    transform = transforms.Compose(aug)

    dataloader = load_cxrs_moco_single_split(
        batch_size=args.batch_size,
        transform=transform,
        csv_file=args.csv_file,
        file_col_name="image_path",
        num_workers=args.num_workers,
    )

    if not (os.path.exists(args.feature_dir)):
        os.mkdir(args.feature_dir)

    # Extract features
    failed_images = []
    feat_dim = 1024  # densenet121 feature dimension
    for img, _, img_dir in tqdm(dataloader):
        batch_size = img.shape[0]

        try:
            feat = encoder(img)
            feat = feat.detach().cpu().numpy().mean((-2, -1))
        except:
            failed_images.extend(img_dir)
            pass

        for i in range(batch_size):
            image_path = img_dir[i]
            write_fn = os.path.join(
                args.feature_dir, image_path.split("/")[-1] + ".pkl"
            )
            if os.path.exists(write_fn):
                print("Skipping exsisting {}".format(write_fn))
                continue
            with open(write_fn, "wb") as pf:
                pickle.dump(feat[i], pf)
        sys.stdout.flush()
    failed_images = pd.DataFrame.from_dict({"files": failed_images})
    failed_images.to_csv(
        os.path.join(args.feature_dir, "failed_images.csv"), index=False
    )
    print(
        "DONE. Feature saved to {}; {} failed.".format(
            args.feature_dir, len(failed_images)
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extracting features from pretrained MoCo."
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        default=None,
        help="Path to save features.",
    )
    parser.add_argument(
        "--model_checkpoint_dir",
        type=str,
        default=None,
        help="Path to pretrained Densenet121 model checkpoint.",
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default=None,
        help="Path to csv file listing CXRs whose features to be extracted.",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers.")
    args = parser.parse_args()

    main(args)
