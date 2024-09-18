"""
This file defines the core research contribution
"""

import matplotlib

matplotlib.use("Agg")
import math

import torch
from torch import nn
from .encoders import psp_encoders
from .stylegan2.model import Generator


def get_keys(d, name):
    if "state_dict" in d:
        d = d["state_dict"]
    d_filt = {k[len(name) + 1 :]: v for k, v in d.items() if k[: len(name)] == name}
    return d_filt


class pSp(nn.Module):
    def __init__(self):
        super(pSp, self).__init__()
        self.encoder = psp_encoders.GradualStyleEncoder()
        self.decoder = Generator()
        self.face_pool = torch.nn.AdaptiveAvgPool2d((1024, 1024))
        self.load_weights()

    def load_weights(self):
        ckpt = torch.load(
            "models/pixel2style2pixel/checkpoints/best_model.pt", map_location="cpu"
        )
        self.encoder.load_state_dict(get_keys(ckpt, "encoder"), strict=False)
        self.decoder.load_state_dict(get_keys(ckpt, "decoder"), strict=False)
        self.__load_latent_avg(ckpt)

    def forward(self, x):
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
