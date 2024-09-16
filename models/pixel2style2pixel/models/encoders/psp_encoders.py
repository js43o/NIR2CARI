import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, PReLU, Sequential, Module

from .helpers import Bottleneck_IR_SE
from ..stylegan2.model import EqualLinear


class GradualStyleBlock(Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [
            Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
        ]
        for i in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(),
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x


class GradualStyleEncoder(Module):
    def __init__(self):
        super(GradualStyleEncoder, self).__init__()

        self.body_modules = [
            Bottleneck_IR_SE(64, 64, 2),
            Bottleneck_IR_SE(64, 64, 1),
            Bottleneck_IR_SE(64, 64, 1),
            Bottleneck_IR_SE(64, 128, 2),
            Bottleneck_IR_SE(128, 128, 1),
            Bottleneck_IR_SE(128, 128, 1),
            Bottleneck_IR_SE(128, 128, 1),
            Bottleneck_IR_SE(128, 256, 2),
            Bottleneck_IR_SE(256, 256, 1),
            Bottleneck_IR_SE(256, 256, 1),
            Bottleneck_IR_SE(256, 256, 1),
            Bottleneck_IR_SE(256, 256, 1),
            Bottleneck_IR_SE(256, 256, 1),
            Bottleneck_IR_SE(256, 256, 1),
            Bottleneck_IR_SE(256, 256, 1),
            Bottleneck_IR_SE(256, 256, 1),
            Bottleneck_IR_SE(256, 256, 1),
            Bottleneck_IR_SE(256, 256, 1),
            Bottleneck_IR_SE(256, 256, 1),
            Bottleneck_IR_SE(256, 256, 1),
            Bottleneck_IR_SE(256, 256, 1),
            Bottleneck_IR_SE(256, 512, 2),
            Bottleneck_IR_SE(512, 512, 1),
            Bottleneck_IR_SE(512, 512, 1),
        ]

        self.input_layer = Sequential(
            Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            BatchNorm2d(64),
            PReLU(64),
        )
        self.body = Sequential(*self.body_modules)

        self.styles = nn.ModuleList()
        self.style_count = 18
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        """Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        """
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True) + y

    def forward(self, x: torch.Tensor):
        x = self.input_layer(x)

        latents = []
        c1, c2, c3 = x, x, x
        for i, l in enumerate(self.body):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        p1, p2 = x, x
        for idx, layer in enumerate(self.styles):
            if idx < self.coarse_ind:
                latents.append(layer(c3))

            elif self.coarse_ind <= idx < self.middle_ind:
                if idx == self.coarse_ind:
                    p2 = self._upsample_add(c3, self.latlayer1(c2))
                latents.append(layer(p2))

            else:
                if idx == self.middle_ind:
                    p1 = self._upsample_add(p2, self.latlayer2(c1))
                latents.append(layer(p1))

        out = torch.stack(latents, dim=1)
        return out
