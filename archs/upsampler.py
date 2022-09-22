import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from litsr.utils.registry import Registry

UpsamplerRegistry = Registry("upsampler")


################
# Upsampler
################


class PA(nn.Module):
    """PA is pixel attention"""

    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out


class ScaleEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        half_dim = self.dim // 2
        self.inv_freq = torch.exp(
            torch.arange(half_dim, dtype=torch.float32)
            * (-math.log(10000) / (half_dim - 1))
        )

    def forward(self, input):
        shape = input.shape
        input = input.view(-1).to(torch.float32)
        sinusoid_in = torch.ger(input, self.inv_freq.to(input.device))
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)

        return pos_emb


class SAPA(nn.Module):
    """Scale aware pixel attention"""

    def __init__(self, nf):
        super().__init__()

        self.scale_embing = ScaleEmbedding(nf)

        self.conv = nn.Sequential(
            nn.Conv2d(nf * 2, nf // 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf // 2, nf, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, scale):

        scale_emb = self.scale_embing(scale)
        scale_emb = (
            scale_emb.unsqueeze_(2)
            .unsqueeze_(3)
            .expand([x.shape[0], scale_emb.shape[1], x.shape[2], x.shape[3]])
        )

        y = torch.cat([x, scale_emb], dim=1)
        y = self.conv(y)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out


@UpsamplerRegistry.register()
class Multiscaleupsampler(nn.Module):
    def __init__(self, n_feat, split=4):
        # final
        super().__init__()

        self.distilled_channels = n_feat // split
        self.out = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 4, 3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(n_feat // 4, 3, 3, padding=1),
        )

        up = []
        up.append(
            nn.Conv2d(n_feat // split * 3, (n_feat // split) * 4 * 3, 3, 1, 1, groups=3)
        )
        up.append(nn.PixelShuffle(2))
        self.upsample = nn.Sequential(*up)

        up1 = []
        up1.append(
            nn.Conv2d(n_feat // split * 2, (n_feat // split) * 4 * 2, 3, 1, 1, groups=2)
        )
        up1.append(nn.PixelShuffle(2))
        self.upsample1 = nn.Sequential(*up1)

        up2 = []
        up2.append(nn.Conv2d(n_feat // split, (n_feat // split) * 4, 3, 1, 1))
        up2.append(nn.PixelShuffle(2))
        self.upsample2 = nn.Sequential(*up2)

        self.SAPA = SAPA(n_feat)

    def forward(self, x, out_size):
        scale = torch.tensor([x.shape[2] / out_size[0]], device=x.device)

        out1, remaining_c1 = torch.split(
            x, (self.distilled_channels, self.distilled_channels * 3), dim=1
        )
        out = self.upsample(remaining_c1)

        out2, remaining_c2 = torch.split(
            out, (self.distilled_channels, self.distilled_channels * 2), dim=1
        )
        out = self.upsample1(remaining_c2)

        out3, remaining_c3 = torch.split(
            out, (self.distilled_channels, self.distilled_channels), dim=1
        )
        out = self.upsample2(remaining_c3)

        distilled_c1 = F.interpolate(
            out1, out_size, mode="bilinear", align_corners=False
        )
        distilled_c2 = F.interpolate(
            out2, out_size, mode="bilinear", align_corners=False
        )
        distilled_c3 = F.interpolate(
            out3, out_size, mode="bilinear", align_corners=False
        )
        distilled_c4 = F.interpolate(
            out, out_size, mode="bilinear", align_corners=False
        )

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, distilled_c4], dim=1)

        out = self.out(self.SAPA(out, scale))
        return out
