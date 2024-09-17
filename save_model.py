import torch

from models.VToonify.model.vtoonify import VToonify


def save():
    example = torch.rand((3, 256, 256))
    module = torch.jit.script(VToonify())
    module.save("nir2cari.pt")


def load():
    loaded = torch.jit.load("nir2cari.pt")
    print(loaded.code)


save()
