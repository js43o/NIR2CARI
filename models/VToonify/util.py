import numpy as np
import dlib
import math
import argparse
import torch

from .model.encoder.encoders.psp_encoders import GradualStyleEncoder
from .model.encoder.align_all_parallel import get_landmark


def load_psp_standalone(checkpoint_path, device="cuda"):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    opts = ckpt["opts"]
    if "output_size" not in opts:
        opts["output_size"] = 1024
    opts["n_styles"] = int(math.log(opts["output_size"], 2)) * 2 - 2
    opts = argparse.Namespace(**opts)
    psp = GradualStyleEncoder(50, "ir_se", opts)
    psp_dict = {
        k.replace("encoder.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("encoder.")
    }
    psp.load_state_dict(psp_dict)
    psp.eval()
    psp = psp.to(device)
    latent_avg = ckpt["latent_avg"].to(device)

    def add_latent_avg(model, inputs, outputs):
        return outputs + latent_avg.repeat(outputs.shape[0], 1, 1)

    psp.register_forward_hook(add_latent_avg)
    return psp


def get_video_crop_parameter(filepath, predictor, padding=[200, 200, 200, 200]):
    if type(filepath) == str:
        img = dlib.load_rgb_image(filepath)
    else:
        img = filepath
    lm = get_landmark(img, predictor)
    if lm is None:
        return None
    lm_chin = lm[0:17]  # left-right
    lm_eyebrow_left = lm[17:22]  # left-right
    lm_eyebrow_right = lm[22:27]  # left-right
    lm_nose = lm[27:31]  # top-down
    lm_nostrils = lm[31:36]  # top-down
    lm_eye_left = lm[36:42]  # left-clockwise
    lm_eye_right = lm[42:48]  # left-clockwise
    lm_mouth_outer = lm[48:60]  # left-clockwise
    lm_mouth_inner = lm[60:68]  # left-clockwise

    scale = 64.0 / (np.mean(lm_eye_right[:, 0]) - np.mean(lm_eye_left[:, 0]))
    center = (
        (np.mean(lm_eye_right, axis=0) + np.mean(lm_eye_left, axis=0)) / 2
    ) * scale
    h, w = round(img.shape[0] * scale), round(img.shape[1] * scale)
    left = max(round(center[0] - padding[0]), 0) // 8 * 8
    right = min(round(center[0] + padding[1]), w) // 8 * 8
    top = max(round(center[1] - padding[2]), 0) // 8 * 8
    bottom = min(round(center[1] + padding[3]), h) // 8 * 8
    return h, w, top, bottom, left, right, scale
