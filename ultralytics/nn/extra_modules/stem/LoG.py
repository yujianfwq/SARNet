# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""LoGStem: Laplacian-of-Gaussian edge-enhanced stem for SARNet.

The stem keeps the high-frequency structural information (edges, contours) that a plain
strided convolution discards. It is composed of four parts:

1. :class:`LoGFilter` -- a 7x7 projection followed by a fixed Laplacian-of-Gaussian kernel.
2. a strided depthwise convolution for 2x spatial downsampling.
3. :class:`Gaussian` -- a fixed Gaussian smoothing branch fused with the downsampled features.
4. :class:`DRFD_LoG` -- a dual-path (conv / max-pool) structure-aware downsampling block.

Reference for the Laplacian-of-Gaussian filter design: https://arxiv.org/abs/2503.14012
"""

import math

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class Conv_Extra(nn.Module):
    """1x1 -> 3x3 -> 1x1 bottleneck used to refine the Gaussian branch."""

    def __init__(self, channel):
        super().__init__()
        self.block = nn.Sequential(
            Conv(channel, 64, 1),
            Conv(64, 64, 3),
            Conv(64, channel, 1, act=False),
        )

    def forward(self, x):
        return self.block(x)


class Gaussian(nn.Module):
    """Fixed Gaussian smoothing branch, optionally followed by a residual refinement."""

    def __init__(self, dim, size, sigma, feature_extra=True):
        super().__init__()
        self.feature_extra = feature_extra
        gaussian = self.gaussian_kernel(size, sigma)
        gaussian = nn.Parameter(data=gaussian, requires_grad=False).clone()
        self.gaussian = nn.Conv2d(dim, dim, kernel_size=size, stride=1, padding=int(size // 2), groups=dim, bias=False)
        self.gaussian.weight.data = gaussian.repeat(dim, 1, 1, 1)
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.SiLU()
        if feature_extra is True:
            self.conv_extra = Conv_Extra(dim)

    def forward(self, x):
        edges_o = self.gaussian(x)
        gaussian = self.act(self.norm(edges_o))
        if self.feature_extra is True:
            out = self.conv_extra(x + gaussian)
        else:
            out = gaussian
        return out

    def gaussian_kernel(self, size: int, sigma: float):
        """Build a normalized ``size`` x ``size`` 2D Gaussian kernel."""
        kernel = torch.FloatTensor(
            [
                [
                    (1 / (2 * math.pi * sigma**2)) * math.exp(-(x**2 + y**2) / (2 * sigma**2))
                    for x in range(-size // 2 + 1, size // 2 + 1)
                ]
                for y in range(-size // 2 + 1, size // 2 + 1)
            ]
        ).unsqueeze(0).unsqueeze(0)
        return kernel / kernel.sum()


class DRFD_LoG(nn.Module):
    """Dual-path downsampling: a strided depthwise conv and a max-pool path are concatenated and fused."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.outdim = dim * 2
        self.conv = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim)
        self.conv_c = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=2, padding=1, groups=dim * 2)
        self.act_c = nn.SiLU()
        self.norm_c = nn.BatchNorm2d(dim * 2)
        self.max_m = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.norm_m = nn.BatchNorm2d(dim * 2)
        self.fusion = nn.Conv2d(dim * 4, self.outdim, kernel_size=1, stride=1)
        # gaussian
        self.gaussian = Gaussian(self.outdim, 5, 0.5, feature_extra=False)
        self.norm_g = nn.BatchNorm2d(self.outdim)

    def forward(self, x):  # x = [B, C, H, W]
        x = self.conv(x)  # x = [B, 2C, H, W]
        gaussian = self.gaussian(x)
        x = self.norm_g(x + gaussian)
        max = self.norm_m(self.max_m(x))  # [B, 2C, H/2, W/2]
        conv = self.norm_c(self.act_c(self.conv_c(x)))  # [B, 2C, H/2, W/2]
        x = torch.cat([conv, max], dim=1)  # [B, 4C, H/2, W/2]
        x = self.fusion(x)  # [B, 2C, H/2, W/2]
        return x


class LoGFilter(nn.Module):
    """Edge enhancement: 7x7 channel projection followed by a fixed Laplacian-of-Gaussian kernel."""

    def __init__(self, in_c, out_c, kernel_size, sigma):
        super().__init__()
        # 7x7 convolution with stride 1 for feature reinforcement, channels from in_c to out_c
        self.conv_init = nn.Conv2d(in_c, out_c, kernel_size=7, stride=1, padding=3)
        # build the Laplacian-of-Gaussian kernel
        ax = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = (xx**2 + yy**2 - 2 * sigma**2) / (2 * math.pi * sigma**4) * torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel - kernel.mean()
        kernel = kernel / kernel.sum()
        log_kernel = kernel.unsqueeze(0).unsqueeze(0)  # add batch and channel dims
        self.LoG = nn.Conv2d(
            out_c, out_c, kernel_size=kernel_size, stride=1, padding=int(kernel_size // 2), groups=out_c, bias=False
        )
        self.LoG.weight.data = log_kernel.repeat(out_c, 1, 1, 1)
        self.act = nn.SiLU()
        self.norm1 = nn.BatchNorm2d(out_c)
        self.norm2 = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv_init(x)
        LoG = self.LoG(x)
        LoG_edge = self.act(self.norm1(LoG))
        x = self.norm2(x + LoG_edge)
        return x


class LoGStem(nn.Module):
    """SARNet stem: LoG edge enhancement + Gaussian smoothing + dual-path downsampling (H/4, W/4).

    Args:
        in_chans (int): Number of input image channels.
        stem_dim (int): Base channel width of the stem; the output has ``stem_dim`` channels.
    """

    def __init__(self, in_chans, stem_dim):
        super().__init__()
        out_c14 = int(stem_dim / 4)
        out_c12 = int(stem_dim / 2)
        # original size to 2x downsampling layer
        self.Conv_D = nn.Sequential(
            nn.Conv2d(out_c14, out_c12, kernel_size=3, stride=1, padding=1, groups=out_c14),
            Conv(out_c12, out_c12, 3, 2, g=out_c12),
        )
        self.LoG = LoGFilter(in_chans, out_c14, 7, 1.0)
        self.gaussian = Gaussian(out_c12, 9, 0.5)
        self.norm = nn.BatchNorm2d(out_c12)
        self.drfd = DRFD_LoG(out_c12)

    def forward(self, x):
        x = self.LoG(x)
        x = self.Conv_D(x)  # original size to 2x downsampling layer
        x = self.norm(x + self.gaussian(x))
        x = self.drfd(x)
        return x  # x = [B, C, H/4, W/4]
