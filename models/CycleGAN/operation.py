import torch
import torch.nn as nn
import torch.nn.functional as F


class DemodulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        stride=1,
        padding=0,
        bias=False,
        dilation=1,
        batch_size=1,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.weight = nn.Parameter(
            torch.randn(batch_size, out_channel, in_channel, kernel_size, kernel_size)
        )
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channel))

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, input):
        batch, in_channel, height, width = input.shape

        demod = torch.rsqrt(self.weight.pow(2).sum([2, 3, 4]) + self.eps)
        weight = self.weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        input = input.view(1, batch * in_channel, height, width)
        if self.bias is None:
            out = F.conv2d(
                input,
                weight,
                padding=self.padding,
                groups=batch,
                dilation=self.dilation,
                stride=self.stride,
            )
        else:
            out = F.conv2d(
                input,
                weight,
                bias=self.bias,
                padding=self.padding,
                groups=batch,
                dilation=self.dilation,
                stride=self.stride,
            )
        _, _, height, width = out.shape
        out = out.view(batch, self.out_channel, height, width)

        return out
