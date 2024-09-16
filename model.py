from models.pix2pixHD.models.pix2pixHD_model import Pix2PixHDModel
from models.pix2pixHD.util import util

from models.VToonify.model.vtoonify import VToonify
from models.pixel2style2pixel.models.psp import pSp

from models.CycleGAN.models import *
from models.CycleGAN.luminance import *
from models.CycleGAN.inference import *

import numpy as np
import torch
import time
import torchvision.transforms.functional as F


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
        self.pix2pixHD = Pix2PixHDModel(self.opt)

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

    def forward(self, data):
        # pix2pixHD
        colorized = self.pix2pixHD(data["label"])
        # colorized = util.tensor2im(colorized.data[0])
        # cv2.imwrite(
        #     "%s/%s_colorized.png" % (self.opt["output"], data["filename"]),
        #     colorized[..., ::-1],
        # )

        # vtoonify
        """
        time_s = time.time()
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
        #     "%s/%s_caricatured.png" % (self.opt["output"], data["filename"]),
        #     caricatured,
        # )
        """

        # pixel2style2pixel
        colorized = resize_and_pad(colorized.data[0], 256).unsqueeze(0)
        caricatured = self.pSp(colorized)[0]
        caricatured = cv2.cvtColor(
            (
                (caricatured.detach().cpu().numpy().squeeze().transpose(1, 2, 0) + 1.0)
                * 127.5
            ).astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        )

        # cyclegan
        synthesized = sample_images(caricatured, self.cyclegan)
        cv2.imwrite("%s/%s.png" % (self.opt["output"], data["filename"]), synthesized)

        return synthesized


def resize_and_pad(img, size: int):
    c, h, w = img.shape

    if h > w:
        w = int(w * (size / h))
        h = size
    else:
        h = int(h * (size / w))
        w = size

    py = (size - h) // 2
    px = (size - w) // 2

    img = F.resize(img, (h, w), antialias=True)

    # img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)
    img = F.pad(
        img,
        [px, py, px + (size - (w + px * 2)), py + (size - (h + py * 2))],
        padding_mode="symmetric",
    )

    return img
