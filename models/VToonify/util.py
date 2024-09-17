import math
import argparse
import torch
from torchvision.transforms import functional as F

from .model.encoder.encoders.psp_encoders import GradualStyleEncoder


def load_psp_standalone(checkpoint_path, device="cuda"):
    psp = GradualStyleEncoder()
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    psp_dict = {
        k.replace("encoder.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("encoder.")
    }

    psp.load_state_dict(psp_dict)
    psp.eval().to(device)

    return psp


def resize_and_pad(img: torch.Tensor, size: int):
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
