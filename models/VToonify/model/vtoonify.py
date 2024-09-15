import torch
import numpy as np
import math
import cv2
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from .bisenet.model import BiSeNet
from .encoder.align_all_parallel import align_face
from .stylegan.model import Generator
from ..util import *


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
    def __init__(self):
        super().__init__()

        # self.in_size = 256
        self.out_size = 1024
        self.img_channels = 3
        self.style_channels = 512
        self.num_mlps = 8
        self.channel_multiplier = 2
        # self.num_res_layers = 6

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # preprocessing models
        self.parsingpredictor = BiSeNet(n_classes=19)
        self.parsingpredictor.load_state_dict(
            torch.load(
                "models/VToonify/checkpoints/faceparsing.pth",
                map_location=lambda storage, loc: storage,
            )
        )
        self.parsingpredictor.to(self.device).eval()
        self.landmarkpredictor = dlib.shape_predictor(
            "models/VToonify/checkpoints/shape_predictor_68_face_landmarks.dat"
        )
        self.pspencoder = load_psp_standalone(
            "models/VToonify/checkpoints/encoder.pt", self.device
        )

        # StyleGANv2, with weights being fixed
        self.generator = Generator()

        # encoder
        self.encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self.img_channels + 19, 32, 3, 1, 1, bias=True),
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
                nn.Conv2d(512, self.img_channels, 1, 1, 0, bias=True),
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

    def forward(self, x):
        paras = get_video_crop_parameter(x, self.landmarkpredictor)
        if paras is not None:
            scale = 1
            kernel_1d = np.array([[0.125], [0.375], [0.375], [0.125]])

            h, w, top, bottom, left, right, scale = paras
            H, W = int(bottom - top), int(right - left)

            if scale <= 0.75:
                x = cv2.sepFilter2D(x, -1, kernel_1d, kernel_1d)
            if scale <= 0.375:
                x = cv2.sepFilter2D(x, -1, kernel_1d, kernel_1d)
            x = cv2.resize(x, (w, h))[top:bottom, left:right]

        with torch.no_grad():
            I = align_face(x, self.landmarkpredictor)
            I = self.transform(I).unsqueeze(dim=0).to(self.device)
            s_w = self.pspencoder(I)
            s_w = self.zplus2wplus(s_w)

            x = self.transform(x).unsqueeze(dim=0).to(self.device)
            x_p = F.interpolate(
                self.parsingpredictor(
                    F.interpolate(
                        x, scale_factor=2, mode="bilinear", align_corners=False
                    )
                    * 2
                )[0],
                scale_factor=0.5,
                recompute_scale_factor=False,
            ).detach()

            feat = torch.cat((x, x_p / 16.0), dim=1)
            styles = s_w.repeat(feat.size(0), 1, 1)

        ###
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

        out = self.generator.convs[6](out, styles[:, 7], noise=noise)
        out = self.generator.convs[7](out, styles[:, 8], noise=noise)
        skip = self.generator.to_rgbs[3](out, styles[:, 9], skip)

        f_E = encoder_features[1]
        out = self.fusion_out[1](torch.cat([out, f_E], dim=1))
        skip = self.fusion_skip[1](torch.cat([skip, f_E], dim=1))
        noise = (
            x.new_empty(out.shape[0], 1, out.shape[2] * 2, out.shape[3] * 2)
            .normal_()
            .detach()
            * 0.0
        )
        out = self.generator.convs[8](out, styles[:, 9], noise=noise)
        out = self.generator.convs[9](out, styles[:, 10], noise=noise)
        skip = self.generator.to_rgbs[4](out, styles[:, 11], skip)

        f_E = encoder_features[2]
        out = self.fusion_out[2](torch.cat([out, f_E], dim=1))
        skip = self.fusion_skip[2](torch.cat([skip, f_E], dim=1))
        noise = (
            x.new_empty(out.shape[0], 1, out.shape[2] * 2, out.shape[3] * 2)
            .normal_()
            .detach()
            * 0.0
        )
        out = self.generator.convs[10](out, styles[:, 11], noise=noise)
        out = self.generator.convs[11](out, styles[:, 12], noise=noise)
        skip = self.generator.to_rgbs[5](out, styles[:, 13], skip)

        f_E = encoder_features[3]
        out = self.fusion_out[3](torch.cat([out, f_E], dim=1))
        skip = self.fusion_skip[3](torch.cat([skip, f_E], dim=1))
        noise = (
            x.new_empty(out.shape[0], 1, out.shape[2] * 2, out.shape[3] * 2)
            .normal_()
            .detach()
            * 0.0
        )
        out = self.generator.convs[12](out, styles[:, 13], noise=noise)
        out = self.generator.convs[13](out, styles[:, 14], noise=noise)
        skip = self.generator.to_rgbs[6](out, styles[:, 15], skip)

        noise = (
            x.new_empty(out.shape[0], 1, out.shape[2] * 2, out.shape[3] * 2)
            .normal_()
            .detach()
            * 0.0
        )
        out = self.generator.convs[14](out, styles[:, 15], noise=noise)
        out = self.generator.convs[15](out, styles[:, 16], noise=noise)
        skip = self.generator.to_rgbs[7](out, styles[:, 17], skip)

        image = skip

        return image

    def zplus2wplus(self, zplus):
        return self.generator.style(
            zplus.reshape(zplus.shape[0] * zplus.shape[1], zplus.shape[2])
        ).reshape(zplus.shape)
