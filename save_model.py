import numpy as np
import torch
import cv2
from PIL import Image

from model import NIR2CARI


def save():
    model = torch.jit.script(NIR2CARI())
    model.save("nir2cari.pt")


def load():
    model = torch.jit.load("nir2cari.pt")

    filename = "Aaron_Eckhart_0001.png"
    image = (
        torch.tensor(cv2.imread("dataset/%s" % filename))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
    )

    result = model(image)

    result = Image.fromarray(result.detach().cpu().numpy().astype(np.uint8))
    result.save(filename)


save()
load()
