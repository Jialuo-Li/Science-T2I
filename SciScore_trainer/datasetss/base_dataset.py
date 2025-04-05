from dataclasses import dataclass

import torch


@dataclass
class BaseDatasetConfig:
    train_split_name: str = "train"
    valid_split_name: str = "validation"
    test_split_name: str = "test"

    batch_size: int = 1
    num_workers: int = 1
    drop_last: bool = True


class BaseDataset(torch.utils.data.Dataset):
    pass
