"""
Dataset and DataLoader utilities for CSE-CIC-IDS2018 windowed tensors.

Loads preprocessed .pt files from data/processed/ and provides
PyTorch DataLoaders for training, validation, and testing.
"""

import os

import torch
from torch.utils.data import DataLoader, Dataset


class FlowWindowDataset(Dataset):
    """Dataset of fixed-size flow windows with binary labels.

    Each sample is a window of shape (window_size, n_features) with a
    label: 0 = benign, 1 = attack.
    """

    def __init__(self, data_dir: str, split: str):
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"), weights_only=True)
        self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"), weights_only=True)
        assert len(self.X) == len(self.y), "X and y length mismatch"

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def get_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """Create DataLoaders for train, val, and test splits.

    Training loader is shuffled; val and test are not.
    """
    pin_memory = torch.cuda.is_available()

    loaders = {}
    for split in ["train", "val", "test"]:
        dataset = FlowWindowDataset(data_dir, split)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split == "train"),
        )
    return loaders
