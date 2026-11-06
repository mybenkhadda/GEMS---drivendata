"""U-Net decoder over the fused skips (fused level features, deepest first)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .encoders import ResBlock


class Decoder(nn.Module):
    def __init__(self, widths: list[int]):
        super().__init__()
        self.ups = nn.ModuleList()
        for i in range(len(widths) - 1, 0, -1):
            self.ups.append(nn.ModuleDict({"reduce": nn.Conv2d(widths[i], widths[i - 1], 1),
                                           "block": ResBlock(2 * widths[i - 1], widths[i - 1])}))

    def forward(self, bottom: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        x = bottom
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = up["block"](torch.cat([up["reduce"](x), skip], 1))
        return x
