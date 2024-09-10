from model import NIR2CARI
from models.pix2pixHD.models.pix2pixHD_model import Pix2PixHDModel
from models.VToonify.model.vtoonify import VToonify
from models.CycleGAN.models import *
import torch

# module = torch.jit.script(NIR2CARI())
# module = torch.jit.script(
#    create_model({"dataroot": "dataset", "output": "output", "gpu_ids": [0]})
# )
# module = torch.jit.script(VToonify())
# module = torch.jit.script(GeneratorResNet((3, 1024, 1024), 9))

opt = {"dataroot": "dataset", "output": "output", "gpu_ids": [0]}

model = Pix2PixHDModel(opt)
module = torch.jit.script(model)
module.save("nir2cari.pt")

""" loaded = torch.jit.load("nir2cari.pt")

print(loaded)
print(loaded.code)
 """
