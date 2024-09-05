from models.pix2pixHD.data.data_loader import CreateDataLoader
from models.pix2pixHD.models.models import create_model
from models.pix2pixHD.util import util

from models.VToonify.model.vtoonify import VToonify
from models.VToonify.model.bisenet.model import BiSeNet
from models.VToonify.model.encoder.align_all_parallel import align_face
from models.VToonify.util import *

from models.CycleGAN.models import *
from models.CycleGAN.luminance import *
from models.CycleGAN.inference import *

import os
import numpy as np
import torch
import torch.nn.functional as F
import dlib
import argparse
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--gpu_ids", type=str, help="which gpu when going inference", default="0"
)
parser.add_argument(
    "--dataroot", type=str, help="path where input images exist", default="dataset"
)
parser.add_argument(
    "--output", type=str, help="path where output images will be", default="output"
)
options = parser.parse_args()
options.gpu_ids = list(map(lambda x: int(x), options.gpu_ids.split(",")))


def load_models():
    # pix2pixHD
    pix2pixHD = create_model(options)

    # vtoonify
    vtoonify = VToonify()
    vtoonify.load_state_dict(
        torch.load(
            "models/VToonify/checkpoints/vtoonify_t.pt",
            map_location=lambda storage, loc: storage,
        )["g_ema"],
        strict=False,
    )
    vtoonify.to(device)

    parsingpredictor = BiSeNet(n_classes=19)
    parsingpredictor.load_state_dict(
        torch.load(
            "models/VToonify/checkpoints/faceparsing.pth",
            map_location=lambda storage, loc: storage,
        )
    )
    parsingpredictor.to(device).eval()

    face_landmarker_model = (
        "models/VToonify/checkpoints/shape_predictor_68_face_landmarks.dat"
    )
    if not os.path.exists(face_landmarker_model):
        import wget, bz2

        wget.download(
            "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
            face_landmarker_model + ".bz2",
        )
        zipfile = bz2.BZ2File(face_landmarker_model + ".bz2")
        data = zipfile.read()
        open(face_landmarker_model, "wb").write(data)
    landmarkpredictor = dlib.shape_predictor(face_landmarker_model)

    pspencoder = load_psp_standalone("models/VToonify/checkpoints/encoder.pt", device)

    # cyclegan
    input_shape = (3, 1024, 1024)
    n_residual_blocks = 9

    cyclegan = GeneratorResNet(input_shape, n_residual_blocks)
    if device == "cuda":
        cyclegan = cyclegan.cuda()

    cyclegan.load_state_dict(torch.load("models/CycleGAN/checkpoints/generator.pth"))

    return (
        pix2pixHD,
        vtoonify,
        parsingpredictor,
        landmarkpredictor,
        pspencoder,
        cyclegan,
    )


if __name__ == "__main__":
    data_loader = CreateDataLoader(options)
    dataset = data_loader.load_data()
    os.makedirs(options.output, exist_ok=True)

    pix2pixHD, vtoonify, parsingpredictor, landmarkpredictor, pspencoder, cyclegan = (
        load_models()
    )

    print("All models are successfully loaded")
    times = []

    for i, data in enumerate(dataset):
        time_s = time.time()
        filename = os.path.basename(data["path"][0]).split(".")[0]

        # pix2pixHD
        colorized = pix2pixHD.inference(data["label"], data["inst"], data["image"])
        colorized = util.tensor2im(colorized.data[0])
        cv2.imwrite(
            "%s/%s_colorized.png" % (options.output, filename), colorized[..., ::-1]
        )

        # vtoonify
        paras = get_video_crop_parameter(colorized, landmarkpredictor)
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
            I = align_face(colorized, landmarkpredictor)
            I = transform(I).unsqueeze(dim=0).to(device)
            s_w = pspencoder(I)
            s_w = vtoonify.zplus2wplus(s_w)

            x = transform(colorized).unsqueeze(dim=0).to(device)
            x_p = F.interpolate(
                parsingpredictor(
                    F.interpolate(
                        x, scale_factor=2, mode="bilinear", align_corners=False
                    )
                    * 2
                )[0],
                scale_factor=0.5,
                recompute_scale_factor=False,
            ).detach()

            inputs = torch.cat((x, x_p / 16.0), dim=1)
            y_tilde = vtoonify(
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
        cv2.imwrite("%s/%s_caricatured.png" % (options.output, filename), caricatured)

        # cyclegan
        synthesized = sample_images(caricatured, cyclegan)
        cv2.imwrite("%s/%s.png" % (options.output, filename), synthesized)

        times.append(time.time() - time_s)
        # print(times[-1])

    # print("avg =", sum(times) / len(times))
