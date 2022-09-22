import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from litsr.archs import common as LSC
from .upsampler import UpsamplerRegistry
from .saliency_arch import SaliencyDetectorLight
from litsr.utils.registry import ArchRegistry
from .common import IMDModule, conv_block, conv_layer


def threshold2tensor(thresholds):
    thresholds.append(1)
    thresholds = thresholds
    rslt = []
    for idx in range(len(thresholds) - 1):
        rslt.append([thresholds[idx], thresholds[idx + 1]])
    return torch.Tensor(rslt).transpose(1, 0)


class FRU(nn.Module):
    def __init__(self, nf=64, num_modules=4):
        super().__init__()

        self.IMDB = nn.ModuleList()
        for i in range(num_modules):
            self.IMDB.append(IMDModule(in_channels=nf))

        self.c = conv_block(nf * num_modules, nf, kernel_size=1, act_type="lrelu")

        self.LR_conv = conv_layer(nf, nf, kernel_size=3)

    def forward(self, input):
        out_list = []
        out = input

        for module in self.IMDB:
            out = module(out)
            out_list.append(out)

        out_B = self.c(torch.cat(out_list, dim=1))
        output = self.LR_conv(out_B) + input
        return output


@ArchRegistry.register()
class SalCSSRNetEpsilonFB(nn.Module):
    def __init__(
        self,
        in_channels=3,
        patch_size=32,
        num_features=64,
        stride=None,
        num_modules=[1, 2, 4],
        thresholds=[0, 0.3, 0.6],
        rgb_mean=(0, 0, 0),
        threshold_p_multi=10,
        upsampler=None,
    ):
        super().__init__()
        self.patch_sz = patch_size
        self.stride = stride
        self.thresholds_ = thresholds
        self.threshold_p_multi = threshold_p_multi

        assert (len(set(num_modules))) == 1

        self.num_modules = num_modules
        self.register_buffer("thresholds", threshold2tensor(self.thresholds_))
        self.saliencyDetector = SaliencyDetectorLight(in_channel=3, channel=16)
        self.head = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.compress = nn.Conv2d(num_features * 2, num_features, 1)
        self.net = FRU(num_features, num_modules=num_modules[0])
        upsampelr = UpsamplerRegistry.get(upsampler["name"])
        self.tail = upsampelr(**upsampler["args"])

        rgb_mean = rgb_mean
        rgb_std = (1.0, 1.0, 1.0)

        self.sub_mean = LSC.MeanShift(1, rgb_mean, rgb_std)
        self.add_mean = LSC.MeanShift(1, rgb_mean, rgb_std, 1)

    def pad(self, img, patch_sz, stride):
        b, c, h, w = img.shape

        if (h - patch_sz) % stride == 0:
            pad_h = 0
        else:
            n_row = (h - patch_sz) // stride + 1
            pad_h = n_row * stride + patch_sz - h
        if (w - patch_sz) % stride == 0:
            pad_w = 0
        else:
            n_col = (w - patch_sz) // stride + 1
            pad_w = n_col * stride + patch_sz - w
        return F.pad(img, pad=(0, pad_w, 0, pad_h), mode="reflect"), (pad_h, pad_w)

    def img2patches(self, img, patch_sz, stride):
        img, (pad_h, pad_w) = self.pad(img, patch_sz, stride)
        b, c, h, w = img.shape
        patches = F.unfold(img, kernel_size=patch_sz, stride=stride)
        patches = (
            patches.reshape(b, c, patch_sz, patch_sz, -1)
            .permute(0, 4, 1, 2, 3)
            .reshape(-1, c, patch_sz, patch_sz)
        )
        num_w = (w - patch_sz) // stride + 1
        num_h = (h - patch_sz) // stride + 1
        return patches, num_h, num_w, pad_h, pad_w

    def patches2img(self, patches, patch_sz, stride, num_h, num_w, pad_h, pad_w):
        patches = rearrange(
            patches, "(b nh nw) c ph pw -> b (c ph pw) (nh nw)", nh=num_h, nw=num_w
        )
        img = F.fold(
            patches,
            output_size=(
                (num_h - 1) * stride + patch_sz,
                (num_w - 1) * stride + patch_sz,
            ),
            kernel_size=patch_sz,
            stride=stride,
        )

        for j in range(1, num_w):
            img[:, :, :, j * stride : j * stride + (patch_sz - stride)] /= 2

        for i in range(1, num_h):
            img[:, :, i * stride : i * stride + (patch_sz - stride), :] /= 2

        h, w = img.shape[2:]
        return img[:, :, : h - pad_h, : w - pad_w]

    def forward(self, x, out_size, *args):
        if not self.training:
            return self.forward_test(x, out_size, *args)

        x = self.sub_mean(x)
        sMap = self.saliencyDetector(x)

        head = self.head(x)
        outList = [head]

        for _ in self.num_modules:
            _input = torch.cat([head, outList[-1]], dim=1)
            _input = self.compress(_input)
            outList.append(self.net(_input))

        outList = outList[1:]

        saliency = sMap.mean(dim=[1, 2, 3]).view(-1, 1)
        saliency = (
            (self.thresholds[1] - saliency) * (saliency - self.thresholds[0])
        ) * self.threshold_p_multi
        saliency_p = torch.softmax(saliency, dim=1).view(-1, len(outList), 1, 1, 1)

        rsltList = []
        for out in outList:
            x_up = F.interpolate(x, out_size, mode="bicubic", align_corners=False)
            out = self.tail(out, out_size) + x_up
            out = self.add_mean(out)
            rsltList.append(out)

        return rsltList, sMap, saliency_p

    def forward_test(self, x, out_size, patch_sz=None, overlap=8, thresholds=None):
        if thresholds:
            self.thresholds_ = thresholds
        x = self.sub_mean(x)
        sMap = self.saliencyDetector(x)
        self.patch_sz = patch_sz if patch_sz else self.patch_sz
        self.stride = self.patch_sz - overlap

        imgPatches, num_h, num_w, pad_h, pad_w = self.img2patches(
            x, self.patch_sz, self.stride
        )
        sMapPatches, _, _, _, _ = self.img2patches(sMap, self.patch_sz, self.stride)
        saliencyList = sMapPatches.mean(dim=[1, 2, 3])

        head = self.head(imgPatches)
        out = head.clone()

        sal_value_list = []

        for idx in range(len(self.num_modules)):
            mask = (saliencyList >= self.thresholds_[idx]).cpu().numpy()
            sal_value_list.append(1.0 * mask.sum() / len(mask))
            index = np.where(mask)[0].tolist()
            next_in = torch.cat((head[index, ...], out[index, ...]), dim=1)
            next_in = self.compress(next_in)
            next_out = self.net(next_in)
            out[index, ...] = next_out

        out = self.patches2img(
            out, self.patch_sz, self.stride, num_h, num_w, pad_h, pad_w
        )

        x_up = F.interpolate(x, out_size, mode="bicubic", align_corners=False)
        out = self.tail(out, out_size) + x_up

        out = self.add_mean(out)
        return out, sMap, sal_value_list
