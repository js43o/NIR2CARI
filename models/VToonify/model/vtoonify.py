import torch
import numpy as np
import math
from torch import nn
from .stylegan.model import Generator


class VToonifyResBlock(nn.Module):
    def __init__(self, fin):
        super().__init__()

        self.conv = nn.Conv2d(fin, fin, 3, 1, 1)
        self.conv2 = nn.Conv2d(fin, fin, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        out = self.lrelu(self.conv(x))
        out = self.lrelu(self.conv2(out))
        out = (out + x) / math.sqrt(2)
        return out


class VToonify(nn.Module):
    def __init__(
        self,
        in_size=256,
        out_size=1024,
        img_channels=3,
        style_channels=512,
        num_mlps=8,
        channel_multiplier=2,
        num_res_layers=6,
    ):
        super().__init__()

        # StyleGANv2, with weights being fixed
        self.generator = Generator(
            out_size, style_channels, num_mlps, channel_multiplier
        )

        self.in_size = in_size
        self.style_channels = style_channels
        channels = self.generator.channels

        # encoder
        self.encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(img_channels + 19, 32, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(32, 128, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(128, 256, 3, 2, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(256, 256, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(256, 512, 3, 2, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(512, 512, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(512, 512, 3, 2, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(512, 512, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                ),
                nn.Sequential(
                    VToonifyResBlock(512),
                    VToonifyResBlock(512),
                    VToonifyResBlock(512),
                    VToonifyResBlock(512),
                    VToonifyResBlock(512),
                    VToonifyResBlock(512),
                ),
                nn.Conv2d(512, img_channels, 1, 1, 0, bias=True),
            ]
        )

        # trainable fusion module
        self.fusion_out = nn.ModuleList(
            [
                nn.Conv2d(1024, 512, 3, 1, 1, bias=True),
                nn.Conv2d(1024, 512, 3, 1, 1, bias=True),
                nn.Conv2d(512, 256, 3, 1, 1, bias=True),
                nn.Conv2d(256, 128, 3, 1, 1, bias=True),
            ]
        )
        self.fusion_skip = nn.ModuleList(
            [
                nn.Conv2d(515, 3, 3, 1, 1, bias=True),
                nn.Conv2d(515, 3, 3, 1, 1, bias=True),
                nn.Conv2d(259, 3, 3, 1, 1, bias=True),
                nn.Conv2d(131, 3, 3, 1, 1, bias=True),
            ]
        )

    def forward(self, x, style):
        # map style to W+ space
        if style is not None and style.ndim < 3:
            adastyles = style.unsqueeze(1).repeat(1, self.generator.n_latent, 1)
        elif style is not None:
            adastyles = style

        feat = x
        encoder_features = []

        # encode
        feat = self.encoder[0](feat)
        encoder_features.append(feat)
        feat = self.encoder[1](feat)
        encoder_features.append(feat)
        feat = self.encoder[2](feat)
        encoder_features.append(feat)
        feat = self.encoder[3](feat)
        encoder_features.append(feat)
        feat = self.encoder[4](feat)

        encoder_features = encoder_features[::-1]

        out = feat
        skip = self.encoder[5](feat)

        self.stylegan().convs

        # decode
        f_E = encoder_features[0]
        out = self.fusion_out[0](torch.cat([out, f_E], dim=1))
        skip = self.fusion_skip[0](torch.cat([skip, f_E], dim=1))
        noise = (
            x.new_empty(out.shape[0], 1, out.shape[2] * 2, out.shape[3] * 2)
            .normal_()
            .detach()
            * 0.0
        )
        out = self.stylegan().convs[6](out, adastyles[:, 7], noise=noise)
        out = self.stylegan().convs[7](out, adastyles[:, 8], noise=noise)
        skip = self.stylegan().to_rgbs[3](out, adastyles[:, 9], skip)

        f_E = encoder_features[1]
        out = self.fusion_out[1](torch.cat([out, f_E], dim=1))
        skip = self.fusion_skip[1](torch.cat([skip, f_E], dim=1))
        noise = (
            x.new_empty(out.shape[0], 1, out.shape[2] * 2, out.shape[3] * 2)
            .normal_()
            .detach()
            * 0.0
        )
        out = self.stylegan().convs[8](out, adastyles[:, 9], noise=noise)
        out = self.stylegan().convs[9](out, adastyles[:, 10], noise=noise)
        skip = self.stylegan().to_rgbs[4](out, adastyles[:, 11], skip)

        f_E = encoder_features[2]
        out = self.fusion_out[2](torch.cat([out, f_E], dim=1))
        skip = self.fusion_skip[2](torch.cat([skip, f_E], dim=1))
        noise = (
            x.new_empty(out.shape[0], 1, out.shape[2] * 2, out.shape[3] * 2)
            .normal_()
            .detach()
            * 0.0
        )
        out = self.stylegan().convs[10](out, adastyles[:, 11], noise=noise)
        out = self.stylegan().convs[11](out, adastyles[:, 12], noise=noise)
        skip = self.stylegan().to_rgbs[5](out, adastyles[:, 13], skip)

        f_E = encoder_features[3]
        out = self.fusion_out[3](torch.cat([out, f_E], dim=1))
        skip = self.fusion_skip[3](torch.cat([skip, f_E], dim=1))
        noise = (
            x.new_empty(out.shape[0], 1, out.shape[2] * 2, out.shape[3] * 2)
            .normal_()
            .detach()
            * 0.0
        )
        out = self.stylegan().convs[12](out, adastyles[:, 13], noise=noise)
        out = self.stylegan().convs[13](out, adastyles[:, 14], noise=noise)
        skip = self.stylegan().to_rgbs[6](out, adastyles[:, 15], skip)

        noise = (
            x.new_empty(out.shape[0], 1, out.shape[2] * 2, out.shape[3] * 2)
            .normal_()
            .detach()
            * 0.0
        )
        out = self.stylegan().convs[14](out, adastyles[:, 15], noise=noise)
        out = self.stylegan().convs[15](out, adastyles[:, 16], noise=noise)
        skip = self.stylegan().to_rgbs[7](out, adastyles[:, 17], skip)

        image = skip

        return image

    def stylegan(self):
        return self.generator

    def zplus2wplus(self, zplus):
        return (
            self.stylegan()
            .style(zplus.reshape(zplus.shape[0] * zplus.shape[1], zplus.shape[2]))
            .reshape(zplus.shape)
        )
