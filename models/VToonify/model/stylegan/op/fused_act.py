import torch
from torch import nn
from torch.nn import functional as F


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, bias=True, negative_slope=0.2, scale=2**0.5):
        super().__init__()

        if bias:
            self.bias = nn.Parameter(torch.zeros(channel))

        else:
            self.bias = None

        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(
    input,
    bias,
    negative_slope: float = 0.2,
    scale: float = 2**0.5,
):
    if input.dim() == 4:
        bias = bias.unsqueeze(1).unsqueeze(2)
    return scale * F.leaky_relu((input + bias), negative_slope)
    # return FusedLeakyReLUFunction.apply(input.contiguous(), bias, negative_slope, scale)
