import torch

from model import NIR2CARI
from models.pix2pixHD.models.pix2pixHD_model import Pix2PixHDModel
from models.VToonify.model.vtoonify import VToonify
from models.pixel2style2pixel.models.psp import pSp


def save():
    module = torch.jit.script(pSp())
    module.save("nir2cari.pt")


def load():
    loaded = torch.jit.load("nir2cari.pt")
    print(loaded.code)


save()
load()
