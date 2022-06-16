import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_lightning import LightningDataModule

from preprocessing.metadata import MetaDataReader


class MissourianImageData(Dataset):
    def __init__(self, dir_path: str, dsettype: str = "train"):
        super().__init__()
        reader = MetaDataReader(img_dir=dir_path)
        cc_dir = reader.get_cc_from_dir()
        data = np.array(list(cc_dir.keys()))
        labels = np.array(list(cc_dir.values()))

        split_idx = int(.75 * data.shape[0])
        if dsettype == "train":
            self.data = torch.tensor(data[:split_idx], dtype=torch.float)
            self.labels = torch.tensor(labels[:split_idx], dtype=torch.float)
        elif dsettype == "validation":
            self.data = torch.tensor(data[split_idx:], dtype=torch.float)
            self.labels = torch.tensor(labels[split_idx:], dtype=torch.float)
        else:
            self.data = torch.tensor(data, dtype=torch.float)
            self.labels = torch.tensor(labels, dtype=torch.float)

        # TODO Add return values

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx, :, :], self.labels[idx, :, :]


class MissourianImageDataset(LightningDataModule):
    # TODO Complete the Dataset
    def __init__(self):
        super().__init__()
        pass
