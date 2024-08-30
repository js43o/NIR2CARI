import torch
import numpy as np
import math
from torch import nn
from .stylegan.model import Generator
import cv2


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
        num_styles = int(np.log2(out_size)) * 2 - 2
        encoder_res = [2**i for i in range(int(np.log2(in_size)), 4, -1)]
        self.encoder = nn.ModuleList()
        # 첫 레이어는 'CONV 파트' (Conv2d + LeakyReLU + Conv2d + LeakyReLU)
        self.encoder.append(
            nn.Sequential(
                nn.Conv2d(img_channels + 19, 32, 3, 1, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(32, channels[in_size], 3, 1, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )
        )

        # 이어서 (입력 사이즈 ~ 32)까지 똑같이 CONV 파트 추가
        for res in encoder_res:
            in_channels = channels[res]
            if res > 32:
                out_channels = channels[res // 2]  # 반씩 줄어듦
                block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                )
                self.encoder.append(block)
            else:  # 마지막 해상도에서는 RES_BLOCK 여러 층 추가
                layers = []
                for _ in range(num_res_layers):
                    layers.append(VToonifyResBlock(in_channels))
                self.encoder.append(nn.Sequential(*layers))
                block = nn.Conv2d(in_channels, img_channels, 1, 1, 0, bias=True)
                self.encoder.append(block)  # 맨 끝에 Conv2d 하나 (RGB 3채널로 만듦)

        # trainable fusion module
        self.fusion_out = nn.ModuleList()
        self.fusion_skip = nn.ModuleList()
        for res in encoder_res[::-1]:
            num_channels = channels[res]
            self.fusion_out.append(
                nn.Conv2d(num_channels * 2, num_channels, 3, 1, 1, bias=True)
            )

            self.fusion_skip.append(nn.Conv2d(num_channels + 3, 3, 3, 1, 1, bias=True))

    def forward(self, x, style):
        # map style to W+ space
        if style is not None and style.ndim < 3:
            adastyles = style.unsqueeze(1).repeat(1, self.generator.n_latent, 1)
        elif style is not None:
            adastyles = style

        feat = x
        encoder_features = []  # 중간 특징 저장용 배열 (이후 G의 각 레이어에 융합)

        # 인코더의 CONV 파트를 통과시키면서 다운샘플링
        for i, block in enumerate(self.encoder[:-2]):
            feat = block(feat)
            encoder_features.append(feat)
        encoder_features = encoder_features[::-1]  # 거꾸로

        # 인코더의 ResBlocks 파트 통과
        for ii, block in enumerate(self.encoder[-2]):
            feat = block(feat)

        # 인코더의 마지막 레이어 (단일 Conv) 통과
        out = feat
        skip = self.encoder[-1](feat)

        # G 시작, 32x32부터 다시 업스케일링
        _index = 1
        m_Es = []
        for conv1, conv2, to_rgb in zip(
            self.stylegan().convs[6::2],  # conv1 = 짝수 파트 (512, 512, 3부터 시작)
            self.stylegan().convs[7::2],  # conv2 = 홀수 파트
            self.stylegan().to_rgbs[3:],
        ):
            # 인코더의 각 중간 레이어들의 특징을 동일 해상도를 갖는 생성자의 레이어에 전달
            if 2 ** (5 + ((_index - 1) // 2)) <= self.in_size:  # 32, 64, 128, 256
                fusion_index = (_index - 1) // 2  # 0, 1, 2, 3
                f_E = encoder_features[fusion_index] * 1

                # 현재 레이어의 특징과 상응하는 인코더의 특징을 concatenation한 뒤 융합
                out = self.fusion_out[fusion_index](torch.cat([out, f_E], dim=1))
                skip = self.fusion_skip[fusion_index](torch.cat([skip, f_E], dim=1))

            # remove the noise input
            batch, _, height, width = out.shape
            noise = (
                x.new_empty(batch, 1, height * 2, width * 2).normal_().detach() * 0.0
            )

            code = adastyles

            out = conv1(out, code[:, _index + 6], noise=noise)
            out = conv2(out, code[:, _index + 7], noise=noise)
            skip = to_rgb(out, code[:, _index + 8], skip)

            _index += 2

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
