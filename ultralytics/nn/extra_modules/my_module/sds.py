# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""SDS: Semantic-guided Dual-path Structure downsampling for SARNet.

SDS replaces the strided convolution used for stage downsampling with two parallel paths:

- a *semantic* path (strided 3x3 conv) that extracts abstract, high-level context;
- a *structure* path (average pooling + 1x1 projection + 3x3 / horizontal / vertical
  depthwise convolutions) that preserves thin, direction-sensitive structures.

A channel-wise :class:`SemanticGate` computed from the semantic path re-weights the
structure path, and the two paths are then concatenated and fused by a 1x1 convolution.
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


# ---------------------------------------------
# Semantic Gate
# ---------------------------------------------
class SemanticGate(nn.Module):

    def __init__(self, channels, reduction=4):
        super().__init__()

        hidden = max(channels // reduction, 16)

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(channels, hidden, 1, bias=False),

            nn.SiLU(inplace=True),

            nn.Conv2d(hidden, channels, 1, bias=False),

            nn.Sigmoid()
        )

    def forward(self, x):
        return self.gate(x)


# ---------------------------------------------
# Structure Retention Branch
# ---------------------------------------------
class StructureBranch(nn.Module):

    def __init__(self, c1, c_out):
        super().__init__()

        # gentle downsampling
        self.pool = nn.AvgPool2d(
            kernel_size=2,
            stride=2
        )

        # channel projection
        self.reduce = Conv(
            c1,
            c_out,
            k=1,
            s=1
        )

        # direction perception
        self.dw3 = Conv(
            c_out,
            c_out,
            k=3,
            s=1,
            g=c_out
        )

        self.dw_h = Conv(
            c_out,
            c_out,
            k=(1, 5),
            s=1,
            p=(0, 2),
            g=c_out
        )

        self.dw_v = Conv(
            c_out,
            c_out,
            k=(5, 1),
            s=1,
            p=(2, 0),
            g=c_out
        )

        self.fuse = Conv(
            c_out,
            c_out,
            k=1,
            s=1
        )

    def forward(self, x):

        x = self.pool(x)

        x = self.reduce(x)

        f1 = self.dw3(x)

        f2 = self.dw_h(x)

        f3 = self.dw_v(x)

        out = f1 + f2 + f3

        out = self.fuse(out)

        return out


# ---------------------------------------------
# SDS
# Semantic-guided Dual-path Structure Downsampling
# ---------------------------------------------
class SDS(nn.Module):

    def __init__(self, c1, c2):
        super().__init__()

        c_half = c2 // 2

        # -----------------------------
        # Semantic Branch
        # -----------------------------
        self.semantic = nn.Sequential(

            Conv(
                c1,
                c_half,
                k=3,
                s=2
            ),

            Conv(
                c_half,
                c_half,
                k=1,
                s=1
            )

        )

        # -----------------------------
        # Structure Branch
        # -----------------------------
        self.structure = StructureBranch(
            c1,
            c_half
        )

        # -----------------------------
        # Semantic Gate
        # -----------------------------
        self.gate = SemanticGate(
            c_half
        )

        # -----------------------------
        # Final Fusion
        # -----------------------------
        self.fusion = Conv(
            c2,
            c2,
            k=1,
            s=1
        )

    def forward(self, x):

        # semantic branch
        semantic = self.semantic(x)

        # structure branch
        structure = self.structure(x)

        # semantic guidance
        gate = self.gate(semantic)

        structure = structure * gate

        # fusion
        out = torch.cat(
            [semantic, structure],
            dim=1
        )

        out = self.fusion(out)

        return out