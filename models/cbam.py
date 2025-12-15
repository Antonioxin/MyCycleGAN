# models/cbam.py

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Channel Attention (CBAM)
    - 使用 AvgPool + MaxPool 两路
    - 共享 MLP（这里用 1x1 Conv 实现）
    """
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        assert in_planes > 0, "in_planes must be positive"
        assert ratio > 0, "ratio must be positive"

        hidden = max(1, in_planes // ratio)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_planes, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention (CBAM)
    - 对通道做 mean/max，拼成 2 通道后做卷积
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 5, 7, 9, 11) or (kernel_size > 1 and kernel_size % 2 == 1), \
            "kernel_size should be odd and > 1"

        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    顺序：Channel -> Spatial
    """
    def __init__(self, in_planes: int, ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio=ratio)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out
