from typing import Any, Optional

from litsr.data import DownsampledDataset
from litsr.utils.registry import DataModuleRegistry
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .sal_dataset import SalHRDownsampledDatasetMS, SalHRDownsampledDataset


@DataModuleRegistry.register()
class SalMSDataModule(LightningDataModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    def setup(self, stage: Optional[str]):
        if stage == "fit":
            self.train_dataset = SalHRDownsampledDatasetMS(
                hr_path=self.opt.train.hr_path,
                sal_path=self.opt.train.sal_path,
                min_scale=self.opt.train.min_scale,
                max_scale=self.opt.train.max_scale,
                is_train=True,
                batch_size=self.opt.train.batch_size,
                repeat=self.opt.train.get("data_repeat"),
                lr_img_sz=self.opt.train.lr_img_sz,
                rgb_range=self.opt.get("rgb_range"),
                cache=self.opt.train.get("data_cache"),
                first_k=self.opt.train.get("data_first_k"),
                data_length=self.opt.train.get("data_length"),
                mean=self.opt.get("mean"),
                std=self.opt.get("std"),
                downsample_mode=self.opt.train.get("downsample_mode"),
                curriculum_learning=self.opt.train.get("curriculum_learning", False),
                return_img_name=False,
            )

        self.val_dataset = {}
        for s in self.opt.valid.scales:
            self.val_dataset[s] = DownsampledDataset(
                datapath=self.opt.valid.data_path,
                scale=s,
                is_train=False,
                rgb_range=self.opt.get("rgb_range"),
                mean=self.opt.get("mean"),
                std=self.opt.get("std"),
                return_img_name=True,
            )

    def train_dataloader(self):
        current_epoch = self.trainer.current_epoch
        self.train_dataset.random_sample_scale(current_epoch)
        return DataLoader(
            self.train_dataset,
            batch_size=self.opt.train.batch_size,
            num_workers=self.opt.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                dataset,
                batch_size=1,
                num_workers=self.opt.num_workers,
                pin_memory=True,
            )
            for _, dataset in self.val_dataset.items()
        ]

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=self.opt.num_workers,
            pin_memory=True,
        )


@DataModuleRegistry.register()
class SalDataModule(LightningDataModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    def setup(self, stage: Optional[str]):
        if stage == "fit":
            self.train_dataset = SalHRDownsampledDataset(
                hr_path=self.opt.train.hr_path,
                sal_path=self.opt.train.sal_path,
                scale=self.opt.train.scale,
                is_train=True,
                repeat=self.opt.train.get("data_repeat"),
                lr_img_sz=self.opt.train.lr_img_sz,
                rgb_range=self.opt.get("rgb_range"),
                cache=self.opt.train.get("data_cache"),
                first_k=self.opt.train.get("data_first_k"),
                data_length=self.opt.train.get("data_length"),
                mean=self.opt.get("mean"),
                std=self.opt.get("std"),
                downsample_mode=self.opt.train.get("downsample_mode"),
                return_img_name=False,
            )

        self.val_dataset = DownsampledDataset(
            datapath=self.opt.valid.data_path,
            scale=self.opt.valid.scale,
            is_train=False,
            rgb_range=self.opt.get("rgb_range"),
            mean=self.opt.get("mean"),
            std=self.opt.get("std"),
            return_img_name=True,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.opt.train.batch_size,
            num_workers=self.opt.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=self.opt.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=self.opt.num_workers,
            pin_memory=True,
        )
