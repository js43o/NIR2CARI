import numpy as np
import torch
import cv2
from PIL import Image

from model import NIR2CARI


def save(caricature_model: str = "vtoonify"):
    model = torch.jit.script(NIR2CARI(caricature_model))
    model.save("torchscripts/nir2cari_%s.pt" % caricature_model)
    print("SAVE COMPLETED")


def load(caricature_model: str = "vtoonify"):
    model = torch.jit.load("torchscripts/nir2cari_%s.pt" % caricature_model)

    # 테스트용 샘플 이미지
    filename = "Anthony_Garotinho_0001.jpg"

    image = cv2.imread("dataset_/%s" % filename)
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float()

    result = model(image)

    result = Image.fromarray(result.detach().cpu().numpy().astype(np.uint8))
    result.save("%s" % filename)

    print("LOAD & INFERENCE COMPLETED")


save("vtoonify")
load("vtoonify")
