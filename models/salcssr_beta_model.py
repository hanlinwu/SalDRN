import time
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from litsr.archs import create_net
from litsr.metrics import calc_psnr_ssim
from litsr.transforms import tensor2uint8
from litsr.utils.logger import logger
from litsr.utils.registry import ModelRegistry
from scipy.ndimage.filters import uniform_filter
from skimage import exposure
from torch.nn import functional as F
from archs import *


@ModelRegistry.register()
class SalCSSRModelBeta(pl.LightningModule):
    """
    Basic SR Model optimized by pixel-wise loss
    """

    def __init__(self, opt):
        """
        opt: in_channels, out_channels, num_features, num_blocks, num_layers
        """
        super().__init__()

        # save hyperparameters
        self.save_hyperparameters(opt)
        self.opt = opt.lit_model.args

        # init super-resolution network
        self.sr_net = create_net(self.opt["network"])

        self.loss_fn = nn.L1Loss()
        self.bce_loss = nn.BCELoss()

    def forward(self, lr, out_size):
        return self.sr_net(lr, out_size)

    def training_step(self, batch, batch_idx):
        lr, hr, smap = batch
        out_size = hr.shape[2:]
        sr, smap_pred, saliency_p = self.forward(lr, out_size)

        smap_pred = F.interpolate(
            smap_pred, size=smap.shape[2:], mode="bicubic", align_corners=False
        )
        if self.current_epoch < self.hparams.trainer.pretrain_epochs:
            loss = self.bce_loss(smap_pred, smap)
        else:
            loss_sr_list = []
            for sr_ in sr:
                loss_sr_list.append(
                    F.l1_loss(sr_, hr, reduction="none").mean(dim=[1, 2, 3])
                )
            loss_sr = (
                (torch.stack(loss_sr_list, dim=1) * saliency_p.squeeze())
                .sum(dim=1)
                .mean(0)
            )

            loss_weights = self.opt.get("loss_weights", [1, 0.1, 0.15])
            psnr_map = self.check_MSE_Map(hr, sr[0])
            loss_sal = self.bce_loss(smap_pred, smap)
            loss_psnr = self.loss_fn(smap_pred, psnr_map)
            loss = (
                loss_weights[0] * loss_sr
                + loss_weights[1] * loss_sal
                + loss_weights[2] * loss_psnr
            )
        return loss

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([out["loss"] for out in outputs]).mean()
        self.log("train/loss", avg_loss, on_epoch=True)
        return

    def validation_step(self, batch, batch_idx, *args):
        lr, hr, name = batch
        out_size = hr.shape[2:]
        sr, sMap, sal_value_list = self.sr_net.forward_test(lr, out_size)

        loss = self.loss_fn(sr, hr)
        sr_np, hr_np, sMap = tensor2uint8(
            [sr.cpu()[0], hr.cpu()[0], sMap.cpu()[0]], self.opt.rgb_range
        )

        if self.opt.valid.get("no_crop_border"):
            crop_border = 0
        else:
            crop_border = int(np.ceil(float(hr.shape[2]) / lr.shape[2]))

        test_Y = self.opt.valid.get("test_Y", True)

        if batch_idx == 0:
            logger.warning("Test_Y: {0}, crop_border: {1}".format(test_Y, crop_border))

        psnr, ssim = calc_psnr_ssim(
            sr_np, hr_np, crop_border=crop_border, test_Y=test_Y
        )

        return {
            "val_loss": loss,
            "val_psnr": psnr,
            "val_ssim": ssim,
            "log_img_sr": sr_np,
            "log_img_smap": sMap,
            "sal_value_list": sal_value_list,
        }

    def validation_epoch_end(self, outputs):
        tensorboard = self.logger.experiment
        psnr_list = []
        if type(outputs[0]) == dict:
            outputs = [outputs]
        scales = self.hparams.data_module.args.valid.scales
        for idx, output in enumerate(outputs):
            scale = scales[idx]
            log_img = output[0]["log_img_sr"]
            log_img_smap = output[0]["log_img_smap"]

            avg_psnr = np.array([x["val_psnr"] for x in output]).mean()

            self.log("val/psnr_x{0}".format(str(scale)), avg_psnr, on_epoch=True)
            tensorboard.add_image(
                "SR/{0}".format(str(scale)),
                log_img,
                self.global_step,
                dataformats="HWC",
            )
            tensorboard.add_image(
                "SalMap/{0}".format(str(scale)),
                log_img_smap,
                self.global_step,
                dataformats="HWC",
            )
            psnr_list.append(avg_psnr)

            sal_value_list = []
            for i in range(len(output[0]["sal_value_list"])):
                sal_value_list.append(
                    np.array([x["sal_value_list"][i] for x in output]).mean()
                )
                self.log(
                    "sal_value/{0}/{1}".format(str(i), str(scale)), sal_value_list[i]
                )

        self.log(
            "val/psnr",
            np.array(psnr_list).mean(),
            on_epoch=True,
            prog_bar=True,
            logger=False,
        )
        return

    def test_step(self, batch, batch_idx, **kwargs):
        lr, hr, name = batch
        out_size = hr.shape[2:]
        torch.cuda.synchronize()
        start = time.time()
        sr, sMap, sal_value_list = self.sr_net.forward_test(
            lr,
            out_size,
            patch_sz=kwargs.get("patch_sz"),
            overlap=kwargs.get("overlap", 8),
            thresholds=kwargs.get("thresholds"),
        )
        torch.cuda.synchronize()
        end = time.time()

        loss = self.loss_fn(sr, hr)
        sr_np, hr_np, sMap = tensor2uint8(
            [sr.cpu()[0], hr.cpu()[0], sMap.cpu()[0]], self.opt.rgb_range
        )

        if kwargs.get("no_crop_border", self.opt.valid.get("no_crop_border")):
            crop_border = 0
        else:
            crop_border = int(np.ceil(float(hr.shape[2]) / lr.shape[2]))

        test_Y = kwargs.get("test_Y", self.opt.valid.get("test_Y", True))

        if batch_idx == 0:
            logger.warning("Test_Y: {0}, crop_border: {1}".format(test_Y, crop_border))

        psnr, ssim = calc_psnr_ssim(
            sr_np, hr_np, crop_border=crop_border, test_Y=test_Y
        )

        return {
            "val_loss": loss,
            "val_psnr": psnr,
            "val_ssim": ssim,
            "log_img_sr": sr_np,
            "log_img_smap": sMap,
            "name": name[0],
            "time": end - start,
            "sal_value_list": sal_value_list,
        }

    def configure_optimizers(self):
        betas = self.opt.optimizer.get("betas") or (0.9, 0.999)
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.opt.optimizer.lr, betas=betas
        )
        if self.opt.optimizer.get("lr_scheduler_step"):
            LR_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.opt.optimizer.lr_scheduler_step,
                gamma=self.opt.optimizer.lr_scheduler_gamma,
            )
        elif self.opt.optimizer.get("lr_scheduler_milestones"):
            LR_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.opt.optimizer.lr_scheduler_milestones,
                gamma=self.opt.optimizer.lr_scheduler_gamma,
            )
        else:
            raise Exception("No lr settings found! ")
        return [optimizer], [LR_scheduler]

    def check_MSE_Map(self, hr, sr):
        device = hr.device

        hr = hr.cpu().detach().numpy()
        sr = sr.cpu().detach().numpy()

        err = (hr - sr) ** 2
        err = err / err.max()
        err = err
        err_f = err
        err_f = uniform_filter(err, size=16)
        err_f = err_f / err_f.max()
        err_f_img = exposure.equalize_hist(err_f)
        err_f_img = torch.from_numpy(err_f_img).to(device)
        err_f_img = err_f_img.mean(dim=1, keepdim=True)
        return err_f_img
