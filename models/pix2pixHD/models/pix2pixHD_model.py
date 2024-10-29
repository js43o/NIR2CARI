from torch import nn
from typing import *
from .networks import define_G


class Pix2PixHDModel(nn.Module):
    def __init__(self):
        super(Pix2PixHDModel, self).__init__()
        self.netG = define_G(3, 3, 64, 4, 9, "instance")

    def forward(self, image):
        image = image.cuda()
        fake_image = self.netG.forward(image)

        return fake_image
