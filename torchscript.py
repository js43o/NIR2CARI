import numpy as np
import torch
import cv2
from PIL import Image

from model import NIR2CARI


def save():
    model = torch.jit.script(NIR2CARI())
    model.save("torchscripts/nir2cari_psp.pt")
    print("SAVE COMPLETED")


def load():
    model = torch.jit.load("torchscripts/nir2cari_psp.pt")

    # 테스트용 샘플 이미지
    filename = "Anthony_Garotinho_0001.jpg"
    image = (
        torch.tensor(cv2.imread("dataset/%s" % filename))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
    )

    result = model(image)

    result = Image.fromarray(result.detach().cpu().numpy().astype(np.uint8))
    result.save(filename)

    print("LOAD & INFERENCE COMPLETED")


save()
load()
