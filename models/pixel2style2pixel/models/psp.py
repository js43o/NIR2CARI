"""
This file defines the core research contribution
"""

import torch
from torch import nn
from .encoders import psp_encoders
from .stylegan2.model import Generator
from utils import resize_and_pad


class pSp(nn.Module):
    def __init__(self):
        super(pSp, self).__init__()
        self.encoder = psp_encoders.GradualStyleEncoder()
        self.decoder = Generator()
        self.face_pool = torch.nn.AdaptiveAvgPool2d((1024, 1024))

    def forward(self, x):
        x = resize_and_pad(x, 256)
        codes = self.encoder(x)
        images = self.decoder([codes])
        images = self.face_pool(images)

        return images

    def __load_latent_avg(self, ckpt, repeat=None):
        if "latent_avg" in ckpt:
            self.latent_avg = ckpt["latent_avg"].to(self.opts.device)
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None
