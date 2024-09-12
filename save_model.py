from model import NIR2CARI
from models.pix2pixHD.models.pix2pixHD_model import Pix2PixHDModel
from models.VToonify.model.vtoonify import VToonify
from models.CycleGAN.models import *
import torch

module = torch.jit.script(VToonify())
module.save("nir2cari.pt")

""" loaded = torch.jit.load("nir2cari.pt")

print(loaded)
print(loaded.code)
 """
