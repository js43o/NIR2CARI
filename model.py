from models.pix2pixHD.models.pix2pixHD_model import Pix2PixHDModel
from models.pix2pixHD.util import util

from models.VToonify.model.vtoonify import VToonify
from models.pixel2style2pixel.models.psp import pSp

from models.CycleGAN.models import *
from utils import *

import numpy as np
import torch
import cv2
import time
import torchvision.transforms.functional as F
from typing import Dict, Union


class NIR2CARI(nn.Module):
    def __init__(
        self, options={"dataroot": "dataset", "output": "output", "gpu_ids": [0]}
    ):
        super(NIR2CARI, self).__init__()
        self.opt = options
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_models()

    def load_models(self):
        # pix2pixHD
        self.pix2pixHD = Pix2PixHDModel()

        """
        # vtoonify
        self.vtoonify = VToonify()
        self.vtoonify.load_state_dict(
            torch.load(
                "models/VToonify/checkpoints/vtoonify_t.pt",
                map_location=lambda storage, loc: storage,
            )["g_ema"],
            strict=False,
        )
        self.vtoonify.to(self.device)
        """

        # pSp
        self.pSp = pSp()
        self.pSp.eval()
        self.pSp.cuda()

        # cyclegan
        self.cyclegan = GeneratorResNet((3, 1024, 1024), 9)
        if self.device == "cuda":
            self.cyclegan = self.cyclegan.cuda()
        self.cyclegan.load_state_dict(
            torch.load("models/CycleGAN/checkpoints/generator.pth")
        )

        print("All models are successfully loaded")

    def forward(self, image: torch.Tensor, filename: str):
        # pix2pixHD
        colorized = self.pix2pixHD(image)

        """
        # vtoonify
        time_s = time.time()
        colorized = util.tensor2im(colorized.data[0])
        # cv2.imwrite(
        #     "%s/%s_colorized.png" % (self.opt["output"], filename),
        #     colorized[..., ::-1],
        # )
        
        y_tilde = self.vtoonify(colorized)
        y_tilde = torch.clamp(y_tilde, -1, 1)
        print(time.time() - time_s)

        caricatured = cv2.cvtColor(
            (
                (y_tilde[0].detach().cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5
            ).astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        )
        # cv2.imwrite(
        #     "%s/%s_caricatured.png" % (self.opt["output"], filename),
        #     caricatured,
        # )

        """
        # pixel2style2pixel
        colorized = resize_and_pad(colorized, 256)
        caricatured = self.pSp(colorized)[0]
        """
        caricatured = cv2.cvtColor(
            (
                (caricatured.detach().cpu().numpy().squeeze().transpose(1, 2, 0) + 1.0)
                * 127.5
            ).astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        )
        """

        # cyclegan
        Y, I, Q = yiq_from_image(caricatured)
        luminance = extend_to_three_channel(Y)
        real = luminance.unsqueeze(dim=0)

        stylized = self.cyclegan(real)
        stylized = F.rgb_to_grayscale(stylized)
        R, G, B = yiq_to_rgb(stylized, I, Q)

        stylized = (
            torch.stack([R.squeeze(), G.squeeze(), B.squeeze()])
            .permute(1, 2, 0)
            .mul(120)
            .add(120)
            .clip(0, 255)
        )
        cv2.imwrite(
            "%s/%s.png" % (self.opt["output"], filename),
            stylized.detach().cpu().numpy(),
        )
