import numpy as np
import torch.utils.data as data
from litsr import transforms
from litsr.data.image_folder import PairedImageFolder, ImageFolder
from litsr.utils.registry import DatasetRegistry
from torchvision import transforms as tv_transforms
from torchvision.transforms import functional as TF
from litsr.utils.logger import logger


@DatasetRegistry.register()
class SalHRDownsampledDataset(data.Dataset):
    def __init__(
        self,
        hr_path,
        sal_path,
        scale,
        is_train,
        lr_img_sz=None,
        rgb_range=1,
        repeat=1,
        cache=None,
        first_k=None,
        data_length=None,
        mean=None,
        std=None,
        downsample_mode="bicubic",
        return_img_name=False,
    ):
        assert not is_train ^ bool(lr_img_sz)

        self.scale = scale
        self.lr_img_sz = lr_img_sz
        self.repeat = repeat or 1
        self.rgb_range = rgb_range or 1
        self.is_train = is_train
        self.mean = mean
        self.std = std
        self.return_img_name = return_img_name
        self.downsample_mode = downsample_mode or "bicubic"
        self.dataset = PairedImageFolder(
            hr_path,
            sal_path,
            repeat=self.repeat,
            cache=cache,
            first_k=first_k,
            data_length=data_length,
        )
        self.file_names = self.dataset.filenames

    def __getitem__(self, idx):
        hr, sal = self.dataset[idx]
        if self.is_train:
            lr, hr, sal = self._transform_train(hr, sal)
        else:
            lr, hr, sal = self._transform_test(hr, sal)

        if self.mean and self.std:
            transforms.normalize(lr, self.mean, self.std, inplace=True)
            transforms.normalize(hr, self.mean, self.std, inplace=True)
            transforms.normalize(sal, self.mean, self.std, inplace=True)

        if self.return_img_name:
            file_name = self.file_names[idx % len(self.file_names)]
            return lr, hr, file_name
        else:
            return lr, hr

    def _transform_train(self, hr, sal):
        lr_img_sz = self.lr_img_sz
        hr_img_sz = int(lr_img_sz * self.scale)

        hr, sal = transforms.random_crop([hr, sal], hr_img_sz)
        hr, sal = transforms.augment([hr, sal])
        hr = TF.to_pil_image(hr)
        lr = transforms.resize_pillow(hr, size=(lr_img_sz, lr_img_sz))
        sal = sal[:, :, 0:1]
        hr, sal = [TF.to_tensor(_).float() for _ in [hr, sal]]
        lr = transforms.pil2tensor(lr, self.rgb_range)
        return lr, hr, sal

    def _transform_test(self, hr, sal):
        raise NotImplementedError

    def __len__(self):
        return len(self.dataset)


@DatasetRegistry.register()
class SalHRDownsampledDatasetMS(data.Dataset):
    def __init__(
        self,
        hr_path,
        sal_path,
        min_scale,
        max_scale,
        is_train,
        batch_size,
        lr_img_sz=None,
        rgb_range=1,
        repeat=1,
        cache=None,
        first_k=None,
        data_length=None,
        mean=None,
        std=None,
        downsample_mode="bicubic",
        curriculum_learning=False,
        return_img_name=False,
    ):
        assert is_train and isinstance(lr_img_sz, int)

        self.min_scale, self.max_scale = min_scale, max_scale
        self.lr_img_sz = lr_img_sz
        self.repeat = repeat or 1
        self.rgb_range = rgb_range or 1
        self.is_train = is_train
        self.mean = mean
        self.std = std
        self.return_img_name = return_img_name
        self.downsample_mode = downsample_mode or "bicubic"
        self.curriculum_learning = curriculum_learning
        self.dataset = PairedImageFolder(
            hr_path,
            sal_path,
            repeat=self.repeat,
            cache=cache,
            first_k=first_k,
            data_length=data_length,
        )
        self.file_names = self.dataset.filenames
        self.batch_size = batch_size

        self.batch_num = int(len(self.dataset) / self.batch_size) + 1
        self.random_sample_scale(0)

    def random_sample_scale(self, epoch):
        if self.curriculum_learning:
            if epoch < 30:
                self.max_scale = 2
            elif epoch < 50:
                self.max_scale = 3
            else:
                self.max_scale = 4
        logger.info(
            "Dataset resampled! scale range is [{:.1f}, {:.1f}]".format(
                self.min_scale, self.max_scale
            )
        )
        self.index_list = list(range(len(self.dataset)))
        np.random.shuffle(self.index_list)
        self.scale_list = (self.max_scale - self.min_scale) * np.random.rand(
            self.batch_num
        ) + self.min_scale

    def __getitem__(self, idx):
        img_idx = self.index_list[idx]
        scale_idx = int(idx / self.batch_size)
        scale = self.scale_list[scale_idx]

        hr, sal = self.dataset[img_idx]
        lr, hr, sal = self._transform_train(hr, sal, scale)

        if self.mean and self.std:
            transforms.normalize(lr, self.mean, self.std, inplace=True)
            transforms.normalize(hr, self.mean, self.std, inplace=True)
            transforms.normalize(sal, self.mean, self.std, inplace=True)

        if self.return_img_name:
            file_name = self.file_names[idx % (len(self.dataset) // self.repeat)]
            return lr, hr, sal, file_name
        else:
            return lr, hr, sal

    def _transform_train(self, hr, sal, scale):
        lr_img_sz = self.lr_img_sz
        hr_img_sz = int(lr_img_sz * scale)

        hr, sal = transforms.random_crop([hr, sal], hr_img_sz)
        hr, sal = transforms.augment([hr, sal])
        hr = TF.to_pil_image(hr)
        lr = transforms.resize_pillow(hr, size=(lr_img_sz, lr_img_sz))
        sal = sal[:, :, 0:1]
        hr, sal = [TF.to_tensor(_).float() for _ in [hr, sal]]
        lr = transforms.pil2tensor(lr, self.rgb_range)

        return lr, hr, sal

    def __len__(self):
        return len(self.dataset)
