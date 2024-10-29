from models.pix2pixHD.models.pix2pixHD_model import Pix2PixHDModel

from models.VToonify.model.vtoonify import VToonify
from models.pixel2style2pixel.models.psp import pSp

from models.CycleGAN.models import GeneratorResNet
from utils import *

import torch
import torchvision.transforms.functional as F
from typing import Union

CARICATURE_MODELS = ["vtoonify", "vtoonify_no_align", "psp"]


class NIR2CARI(torch.nn.Module):
    # caricature_model: "vtoonify" or "psp"
    def __init__(self, caricature_model="vtoonify"):
        super(NIR2CARI, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.caricature_model = (
            caricature_model.lower().strip() if caricature_model else "vtoonify"
        )
        if self.caricature_model not in CARICATURE_MODELS:
            print(
                "▶ No caricature models was specified, so only RGB colorization will be performed"
            )
        self.load_models()

    def load_models(self):
        # pix2pixHD
        self.pix2pixHD = Pix2PixHDModel()
        self.pix2pixHD.netG.load_state_dict(
            torch.load("models/pix2pixHD/checkpoints/latest_net_G.pth")
        )

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

        # psp
        self.pSp = pSp()
        pSpCheckpoints = torch.load(
            "models/pixel2style2pixel/checkpoints/best_model.pt", map_location="cpu"
        )
        self.pSp.encoder.load_state_dict(
            get_keys(pSpCheckpoints, "encoder"), strict=False
        )
        self.pSp.decoder.load_state_dict(
            get_keys(pSpCheckpoints, "decoder"), strict=False
        )
        self.pSp.eval()
        self.pSp.cuda()

        # cyclegan
        self.cyclegan = GeneratorResNet((3, 1024, 1024), 9)
        self.cyclegan.cuda()
        self.cyclegan.load_state_dict(
            torch.load("models/CycleGAN/checkpoints/generator.pth")
        )

        print("▶ All models are successfully loaded")

    def forward(self, image: torch.Tensor):
        colorized = self.pix2pixHD(image)
        caricatured = colorized

        if self.caricature_model.startswith("vtoonify"):
            skip_align = "no_align" in self.caricature_model
            caricatured = self.vtoonify(colorized, skip_align).squeeze()
        elif self.caricature_model == "psp":
            caricatured = self.pSp(colorized)[0]
        else:
            colorized = (colorized.squeeze().permute(1, 2, 0) * 255.0 + 1).clip(0, 255)
            return colorized

        Y, I, Q = yiq_from_image(caricatured)
        luminance = extend_to_three_channel(Y)
        real = luminance.unsqueeze(dim=0)

        stylized = self.cyclegan(real)
        stylized = F.rgb_to_grayscale(stylized)
        R, G, B = yiq_to_rgb(stylized, I, Q)

        stylized = (
            torch.stack([B.squeeze(), G.squeeze(), R.squeeze()])
            .permute(1, 2, 0)
            .mul(127)
            .add(127)
            .clip(0, 255)
        )

        return stylized
