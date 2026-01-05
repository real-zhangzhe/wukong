import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CriteoDataset(Dataset):
    def __init__(
        self,
        npz_file_path: str,
        split: str = "train",
    ):
        data = np.load(npz_file_path)
        self.labels = torch.from_numpy(data["y"].astype(np.float32))  # 0 or 1
        self.dense_features = torch.log(
            torch.from_numpy(data["X_int"].astype(np.float32)) + 1.0
        )  # float32
        self.sparse_features = torch.from_numpy(data["X_cat"].astype(np.int32))  # int32
        del data
        assert split in ["train", "valid"], "split must be 'train' or 'valid'"
        assert (
            self.labels.shape[0]
            == self.dense_features.shape[0]
            == self.sparse_features.shape[0]
            == 45840617
        ), "The dataset size does not match the expected size of 45840617 samples."
        if split == "train":
            self.labels = self.labels[:39291958]
            self.dense_features = self.dense_features[:39291958]
            self.sparse_features = self.sparse_features[:39291958]
        else:  # split == "valid"
            self.labels = self.labels[39291958:]
            self.dense_features = self.dense_features[39291958:]
            self.sparse_features = self.sparse_features[39291958:]

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        sparse_feat = self.sparse_features[idx]
        dense_feat = self.dense_features[idx]
        label = self.labels[idx]
        return sparse_feat, dense_feat, label


def get_dataloader(
    npz_file_path: str,
    split: str = "train",
    batch_size: int = 1024,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    dataset = CriteoDataset(npz_file_path, split)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader
