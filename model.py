from models.pix2pixHD.models.pix2pixHD_model import Pix2PixHDModel
from models.pix2pixHD.util.util import tensor2im

from models.VToonify.model.vtoonify import VToonify
from models.pixel2style2pixel.models.psp import pSp

from models.CycleGAN.models import GeneratorResNet
from utils import *

import torch
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image


class NIR2CARI(torch.nn.Module):
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

        # vtoonify
        colorized = tensor2im(colorized.data[0])
        caricatured = self.vtoonify(colorized).squeeze()

        # pixel2style2pixel
        """
        colorized = resize_and_pad(colorized, 256)
        caricatured = self.pSp(colorized)[0]
        """

        # cyclegan
        Y, I, Q = yiq_from_image(caricatured)
        luminance = extend_to_three_channel(Y)
        real = luminance.unsqueeze(dim=0)

        stylized = self.cyclegan(real)
        stylized = F.rgb_to_grayscale(stylized)
        R, G, B = yiq_to_rgb(stylized, I, Q)

        stylized = (
            torch.stack([B.squeeze(), G.squeeze(), R.squeeze()])
            .permute(1, 2, 0)
            .mul(120)
            .add(120)
            .clip(0, 255)
        )

        result = Image.fromarray(stylized.detach().cpu().numpy().astype(np.uint8))
        result.save("%s/%s.png" % (self.opt["output"], filename))
