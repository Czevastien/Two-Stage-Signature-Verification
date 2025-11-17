import os
import random
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset
import torch


class OfflinePremadeTripletDataset(Dataset):
    """
    Expects a CSV where each row has exactly three columns:
      image1,image2,image3
    corresponding to anchor, positive, and negative image paths (relative to img_dir).
    """

    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        anchor_fname = self.df.iat[idx, 0]
        positive_fname = self.df.iat[idx, 1]
        negative_fname = self.df.iat[idx, 2]

        def load(fname):
            folder = fname.rsplit("_")[0]
            rel_path = os.path.join(folder, fname)
            full_path = os.path.join(self.img_dir, rel_path)
            img = Image.open(full_path).convert("L")
            return self.transform(img) if self.transform else img

        anchor = load(anchor_fname)
        positive = load(positive_fname)
        negative = load(negative_fname)

        debug = (anchor_fname, positive_fname, negative_fname)

        return anchor, positive, negative, debug


class OnlineRandomTripletDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None, **kwargs):
        self.df = pd.read_csv(csv_path)
        """
         CSV columns: ["person_id", "signature_id", "label", "filename"]
        """
        self.img_dir = img_dir
        self.transform = transform

        # Unique person IDs as integers
        self.pid_list = self.df["person_id"].unique().astype(int).tolist()

        # Configuration options
        self.config = kwargs
        self.sets_per_person = kwargs.get("sets_per_person", 10)
        self.forgery_negative_ratio = kwargs.get("forgery_negative_ratio", 1)
        self.external_negative_ratio = kwargs.get("external_negative_ratio", 1)

        # Precompute indices by person_id and label for faster sampling
        self.person_label_to_indices = {}
        for pid in self.pid_list:
            self.person_label_to_indices[pid] = {
                1: self.df[
                    (self.df.person_id == pid) & (self.df.label == 1)
                ].index.tolist(),
                0: self.df[
                    (self.df.person_id == pid) & (self.df.label == 0)
                ].index.tolist(),
            }

        # Precompute external negatives (positives of other people)
        self.external_indices = self.df[self.df.label == 1].index.tolist()

    def __len__(self):
        return len(self.pid_list) * self.sets_per_person

    def __getitem__(self, idx):
        # Pick anchor person
        anchor_pid = self.pid_list[idx % len(self.pid_list)]

        # Sample anchor and positive (genuine) images
        anchor_idx, positive_idx = random.sample(
            self.person_label_to_indices[anchor_pid][1], 2
        )

        # Weighted negative selection
        weight = self.forgery_negative_ratio / (
            self.forgery_negative_ratio + self.external_negative_ratio
        )
        random_num = random.random()
        if random_num < weight and len(self.person_label_to_indices[anchor_pid][0]) > 0:
            # Sample negative from forgeries of the same person
            negative_idx = random.choice(self.person_label_to_indices[anchor_pid][0])
        else:
            # Sample negative from positives of other people
            other_pids = [pid for pid in self.pid_list if pid != anchor_pid]
            external_pid = random.choice(other_pids)
            negative_idx = random.choice(self.person_label_to_indices[external_pid][1])

        # Image loader helper
        def load(idx):
            row = self.df.iloc[idx]
            folder = str(row["person_id"])
            fname = row["filename"]
            full_path = os.path.join(self.img_dir, folder, fname)
            img = Image.open(full_path).convert("L")
            return self.transform(img) if self.transform else img

        anchor = load(anchor_idx)
        positive = load(positive_idx)
        negative = load(negative_idx)

        debug = [
            self.df.iloc[idx].tolist()
            for idx in [anchor_idx, positive_idx, negative_idx]
        ]

        return idx % len(self.pid_list), anchor, positive, negative, debug


class OnlineHardTripletDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None, cfg={}):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.sampler_size = cfg["training_hard"]["epoch_multiplier"]

        # Group dataframe rows by person_id
        self.groups = {pid: g for pid, g in self.df.groupby("person_id")}
        self.pids = list(self.groups.keys())

    def __len__(self):
        return len(self.pids) * self.sampler_size

    def __getitem__(self, idx):
        pid = self.pids[idx % len(self.pids)]
        group = self.groups[pid]

        def load(row):
            folder = str(row["person_id"])
            fname = row["filename"]
            full_path = os.path.join(self.img_dir, folder, fname)
            img = Image.open(full_path).convert("L")
            return self.transform(img) if self.transform else img

        # stack images into tensor [N, C, H, W]
        images = torch.stack([load(row) for _, row in group.iterrows()])

        # labels as tensor [N]
        labels = torch.tensor(group["label"].tolist(), dtype=torch.long)

        return pid, images, labels
