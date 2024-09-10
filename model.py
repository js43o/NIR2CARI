from models.pix2pixHD.models.pix2pixHD_model import Pix2PixHDModel
from models.pix2pixHD.util import util

from models.VToonify.model.vtoonify import VToonify
from models.VToonify.model.bisenet.model import BiSeNet
from models.VToonify.model.encoder.align_all_parallel import align_face
from models.VToonify.util import *

from models.CycleGAN.models import *
from models.CycleGAN.luminance import *
from models.CycleGAN.inference import *

import numpy as np
import torch
import torch.nn.functional as F
import dlib


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
        colorized = util.tensor2im(colorized.data[0])
        cv2.imwrite(
            "%s/%s_colorized.png" % (self.opt["output"], data["filename"]),
            colorized[..., ::-1],
        )

        # vtoonify
        paras = get_video_crop_parameter(colorized, self.landmarkpredictor)
        if paras is not None:
            scale = 1
            kernel_1d = np.array([[0.125], [0.375], [0.375], [0.125]])

            h, w, top, bottom, left, right, scale = paras
            H, W = int(bottom - top), int(right - left)

            if scale <= 0.75:
                colorized = cv2.sepFilter2D(colorized, -1, kernel_1d, kernel_1d)
            if scale <= 0.375:
                colorized = cv2.sepFilter2D(colorized, -1, kernel_1d, kernel_1d)
            colorized = cv2.resize(colorized, (w, h))[top:bottom, left:right]

        with torch.no_grad():
            I = align_face(colorized, self.landmarkpredictor)
            I = transform(I).unsqueeze(dim=0).to(self.device)
            s_w = self.pspencoder(I)
            s_w = self.vtoonify.zplus2wplus(s_w)

            x = transform(colorized).unsqueeze(dim=0).to(self.device)
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

            inputs = torch.cat((x, x_p / 16.0), dim=1)
            y_tilde = self.vtoonify(
                inputs,
                s_w.repeat(inputs.size(0), 1, 1),
            )
            y_tilde = torch.clamp(y_tilde, -1, 1)

        caricatured = cv2.cvtColor(
            (
                (y_tilde[0].detach().cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5
            ).astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        )
        cv2.imwrite(
            "%s/%s_caricatured.png" % (self.opt["output"], data["filename"]),
            caricatured,
        )

        # cyclegan
        synthesized = sample_images(caricatured, self.cyclegan)
        cv2.imwrite("%s/%s.png" % (self.opt["output"], data["filename"]), synthesized)

        return synthesized
