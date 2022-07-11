import os
from collections.abc import Callable
from pathlib import Path
from typing import Union, Sequence, Optional, List

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from torchvision.datasets.folder import default_loader
from torchvision.transforms import transforms

from preprocessing.metadata import MetaDataReader


class MissourianImageData(Dataset):
    def __init__(
            self,
            dir_path: str,
            dsettype: str = "train",
            transform: Callable = None,
            include_cc: bool = False,
            **kwargs
    ):
        super().__init__()
        self.data_dir = Path(dir_path)
        self.transforms = transform
        self.include_cc = include_cc

        imgs = sorted([
            os.path.join(d, x)
            for d, dirs, files in os.walk(self.data_dir)
            for x in files if x.endswith((".jpg", ".JPG", ".png", ".PNG"))
        ])

        # imgs = sorted([f for f in self.data_dir.iterdir() if (f.suffix == '.jpg'
        #                                                       or f.suffix == '.JPG'
        #                                                       or f.suffix == '.png'
        #                                                       or f.suffix == '.PNG')
        #                ])

        if dsettype == "train":
            self.imgs = imgs[:int(len(imgs) * 0.75)]
        elif dsettype == "validation":
            self.imgs = imgs[int(len(imgs) * 0.75):]
        else:
            self.imgs = imgs
        # self.imgs = imgs[:int(len(imgs) * 0.75)] if dsettype == "train" else imgs[int(len(imgs)*0.75):]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if self.include_cc:
            reader = MetaDataReader()
            cc = reader.get_cc_from_file(self.imgs[idx])
        else:
            cc = 0.0

        img = default_loader(self.imgs[idx])
        if self.transforms is not None:
            img = self.transforms(img)
        return img, float(cc)


class MissourianImageDataset(LightningDataModule):
    def __init__(
            self,
            dir_path: str,
            train_batch_size: int = 32,
            val_batch_size: int = 32,
            crop_size: Union[int, Sequence[int]] = (128, 128),
            num_workers: int = 0,
            pin_memory: bool = False,
            include_cc: bool = False,
            **kwargs
    ):
        super().__init__()
        self.dir_path = dir_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.crop_size = crop_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.include_cc = include_cc

    def setup(self, stage: Optional[str] = None) -> None:
        train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                               # transforms.CenterCrop(self.crop_size),
                                               transforms.Resize(self.crop_size),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                             # transforms.CenterCrop(self.crop_size),
                                             transforms.Resize(self.crop_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        self.train_dataset = MissourianImageData(
            self.dir_path,
            dsettype="train",
            transform=train_transforms,
            include_cc=self.include_cc
        )

        self.val_dataset = MissourianImageData(
            self.dir_path,
            dsettype="validation",
            transform=val_transforms,
            include_cc=self.include_cc
        )

        self.test_dataset = MissourianImageData(
            self.dir_path,
            dsettype="test",
            transform=train_transforms,
            include_cc=self.include_cc
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory
        )
