import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional, List

from .op import (
    FusedLeakyReLU,
    fused_leaky_relu,
    upfirdn2d,
)


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input**2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor**2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor**2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        dilation=1,  ## modified
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size**2)

        self.stride = stride
        self.padding = padding
        self.dilation = dilation  ## modified

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,  ## modified
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding}, dilation={self.dilation})"  ## modified
        )


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation is not None:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        style_dim: int,
        demodulate: bool = True,
        upsample: bool = False,
        downsample: bool = False,
        fused: bool = True,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        blur_kernel = [1, 3, 3, 1]

        factor = 2
        p = (len(blur_kernel) - factor) - (kernel_size - 1)
        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2 + 1
        self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        fan_in = in_channel * kernel_size**2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate
        self.fused = fused

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, style):

        batch, in_channel, height, width = input.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)
        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise: Optional[torch.Tensor] = None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        style_dim: int,
        upsample: bool = False,
        demodulate: bool = True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise: Optional[torch.Tensor] = None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.upsample = Upsample([1, 3, 3, 1])
        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip: Optional[torch.Tensor] = None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip

        return out


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.input = ConstantInput(512)
        self.conv1 = StyledConv(512, 512, 3, 512)
        self.to_rgb1 = ToRGB(512, 512)

        self.style = nn.Sequential(
            PixelNorm(),
            EqualLinear(
                512,
                512,
                lr_mul=0.01,
                activation="fused_lrelu",
            ),
            EqualLinear(
                512,
                512,
                lr_mul=0.01,
                activation="fused_lrelu",
            ),
            EqualLinear(
                512,
                512,
                lr_mul=0.01,
                activation="fused_lrelu",
            ),
            EqualLinear(
                512,
                512,
                lr_mul=0.01,
                activation="fused_lrelu",
            ),
            EqualLinear(
                512,
                512,
                lr_mul=0.01,
                activation="fused_lrelu",
            ),
            EqualLinear(
                512,
                512,
                lr_mul=0.01,
                activation="fused_lrelu",
            ),
            EqualLinear(
                512,
                512,
                lr_mul=0.01,
                activation="fused_lrelu",
            ),
            EqualLinear(
                512,
                512,
                lr_mul=0.01,
                activation="fused_lrelu",
            ),
        )

        self.convs = nn.ModuleList(
            [
                StyledConv(512, 512, 3, 512, upsample=True),
                StyledConv(512, 512, 3, 512),
                StyledConv(512, 512, 3, 512, upsample=True),
                StyledConv(512, 512, 3, 512),
                StyledConv(512, 512, 3, 512, upsample=True),
                StyledConv(512, 512, 3, 512),
                StyledConv(512, 512, 3, 512, upsample=True),
                StyledConv(512, 512, 3, 512),
                StyledConv(512, 256, 3, 512, upsample=True),
                StyledConv(256, 256, 3, 512),
                StyledConv(256, 128, 3, 512, upsample=True),
                StyledConv(128, 128, 3, 512),
                StyledConv(128, 64, 3, 512, upsample=True),
                StyledConv(64, 64, 3, 512),
                StyledConv(64, 32, 3, 512, upsample=True),
                StyledConv(32, 32, 3, 512),
            ]
        )

        self.to_rgbs = nn.ModuleList(
            [
                ToRGB(512, 512),
                ToRGB(512, 512),
                ToRGB(512, 512),
                ToRGB(512, 512),
                ToRGB(256, 512),
                ToRGB(128, 512),
                ToRGB(64, 512),
                ToRGB(32, 512),
            ]
        )

    def forward(self, styles: List[torch.Tensor]):
        latent = styles[0]

        out = self.input(latent)
        out = self.conv1(out, latent[:, 0])

        skip = self.to_rgb1(out, latent[:, 1])

        out = self.convs[0](out, latent[:, 1])
        out = self.convs[1](out, latent[:, 2])
        skip = self.to_rgbs[0](out, latent[:, 3], skip)

        out = self.convs[2](out, latent[:, 3])
        out = self.convs[3](out, latent[:, 4])
        skip = self.to_rgbs[1](out, latent[:, 5], skip)

        out = self.convs[4](out, latent[:, 5])
        out = self.convs[5](out, latent[:, 6])
        skip = self.to_rgbs[2](out, latent[:, 7], skip)

        out = self.convs[6](out, latent[:, 7])
        out = self.convs[7](out, latent[:, 8])
        skip = self.to_rgbs[3](out, latent[:, 9], skip)

        out = self.convs[8](out, latent[:, 9])
        out = self.convs[9](out, latent[:, 10])
        skip = self.to_rgbs[4](out, latent[:, 11], skip)

        out = self.convs[10](out, latent[:, 11])
        out = self.convs[11](out, latent[:, 12])
        skip = self.to_rgbs[5](out, latent[:, 13], skip)

        out = self.convs[12](out, latent[:, 13])
        out = self.convs[13](out, latent[:, 14])
        skip = self.to_rgbs[6](out, latent[:, 15], skip)

        out = self.convs[14](out, latent[:, 15])
        out = self.convs[15](out, latent[:, 16])
        skip = self.to_rgbs[7](out, latent[:, 17], skip)

        image = skip

        return image
