import torch
from torch import nn
from typing import *
from .networks import define_G


class Pix2PixHDModel(nn.Module):
    def __init__(self):
        super(Pix2PixHDModel, self).__init__()
        self.netG = define_G(3, 3, 64, 4, 9, "instance", gpu_ids=[0])

        save_path = "models/pix2pixHD/checkpoints/latest_net_G.pth"
        self.netG.load_state_dict(torch.load(save_path))

    def forward(self, label):
        label = label.data.cuda()
        fake_image = self.netG.forward(label)

        return fake_image
