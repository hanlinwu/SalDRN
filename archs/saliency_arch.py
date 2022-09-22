import torch
import torch.nn as nn
import torch.nn.functional as F
from litsr.utils.registry import ArchRegistry

#######################
# Saliency Detection  #
#######################


class Upsample(nn.Sequential):
    def __init__(self, channel):
        layers = [
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(channel, channel, 3, padding=1),
        ]

        super().__init__(*layers)


class Downsample(nn.Sequential):
    def __init__(self, channel):
        layers = [nn.Conv2d(channel, channel, 3, stride=2, padding=1)]

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dropout):
        super().__init__()

        self.norm1 = nn.GroupNorm(16, in_channel)
        self.activation1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, padding=1)

        self.norm2 = nn.GroupNorm(16, out_channel)
        self.activation2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, padding=1)

        if in_channel != out_channel:
            self.skip = nn.Conv2d(in_channel, out_channel, 1)
        else:
            self.skip = None

    def forward(self, input):

        out = self.conv1(self.activation1(self.norm1(input)))
        out = self.conv2(self.dropout(self.activation2(self.norm2(out))))

        if self.skip is not None:
            input = self.skip(input)

        return out + input


class SaliencyDetector(nn.Module):
    def __init__(
        self,
        in_channel,
        channel,
        channel_multiplier,
        n_res_blocks,
        dropout=0,
        down_scale=2,
    ):
        super().__init__()
        self.down_scale = down_scale

        n_block = len(channel_multiplier)
        down_layers = [
            nn.Conv2d(in_channel, channel, 3, padding=1, stride=self.down_scale)
        ]
        feat_channels = [channel]
        in_channel = channel
        for i in range(n_block):
            for _ in range(n_res_blocks):
                channel_mult = int(channel * channel_multiplier[i])

                down_layers.append(
                    ResBlock(
                        in_channel,
                        channel_mult,
                        dropout,
                    )
                )
                feat_channels.append(channel_mult)
                in_channel = channel_mult

            if i != n_block - 1:
                down_layers.append(Downsample(in_channel))
                feat_channels.append(in_channel)

        self.down = nn.ModuleList(down_layers)

        self.mid = nn.ModuleList(
            [
                ResBlock(in_channel, in_channel, dropout=dropout),
            ]
        )

        up_layers = []
        for i in reversed(range(n_block)):
            for _ in range(n_res_blocks + 1):
                channel_mult = int(channel * channel_multiplier[i])

                up_layers.append(
                    ResBlock(
                        in_channel + feat_channels.pop(),
                        channel_mult,
                        dropout=dropout,
                    )
                )
                in_channel = channel_mult

            if i != 0:
                up_layers.append(Upsample(in_channel))

        self.up = nn.ModuleList(up_layers)

        self.out = nn.Sequential(
            nn.GroupNorm(16, in_channel),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channel, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        # return ones_like(input)
        feats = []
        h_origin, w_origin = input.shape[2:]

        out = input

        for layer in self.down:
            out = layer(out)
            feats.append(out)

        for layer in self.mid:
            out = layer(out)

        for layer in self.up:
            if isinstance(layer, ResBlock):
                h, w = feats[-1].shape[-2:]
                out = layer(torch.cat((out[:, :, 0:h, 0:w], feats.pop()), 1))
            else:
                out = layer(out)

        out = self.out(out)

        out = F.interpolate(
            out, size=(h_origin, w_origin), mode="bilinear", align_corners=False
        )

        return out


class ResBlockLight(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.act = nn.LeakyReLU(0.02, True)
        self.conv1 = nn.Conv2d(channel, channel, 3, padding=1)
        self.norm1 = nn.GroupNorm(16, channel)

        self.conv2 = nn.Conv2d(channel, channel, 3, padding=1)
        self.norm2 = nn.GroupNorm(16, channel)

    def forward(self, input):

        out = self.norm1(self.act(self.conv1(input)))
        out = self.norm2(self.act(self.conv2(out)))
        return out + input


class SaliencyDetectorLight(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.down1 = nn.Conv2d(in_channel, channel, 3, padding=1, stride=2)
        self.res1 = ResBlockLight(channel)

        self.down2 = nn.Conv2d(channel, channel, 3, padding=1, stride=2)
        self.res2 = ResBlockLight(channel)

        self.down3 = nn.Conv2d(channel, channel, 3, padding=1, stride=2)
        self.res3 = ResBlockLight(channel)

        self.compress = nn.Conv2d(channel * 3, channel, 1, padding=0)
        self.out = nn.Sequential(
            nn.Conv2d(channel, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        h_origin, w_origin = input.shape[2:]

        out1 = self.res1(self.down1(input))
        out2 = self.res2(self.down2(out1))
        out3 = self.res3(self.down3(out2))

        out1_resample = F.interpolate(out1, size=(h_origin, w_origin), mode="nearest")
        out2_resample = F.interpolate(out2, size=(h_origin, w_origin), mode="nearest")
        out3_resample = F.interpolate(out3, size=(h_origin, w_origin), mode="nearest")

        out = self.compress(
            torch.cat([out1_resample, out2_resample, out3_resample], dim=1)
        )
        return self.out(out)
