import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetBlock(nn.Module):
    """基础残差块，后续可以插入 CBAM 模块"""
    def __init__(self, dim):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim),
        )

        # CBAM 插槽（空）
        self.cbam = None

    def forward(self, x):
        out = self.conv_block(x)
        if self.cbam:
            out = self.cbam(out)
        return x + out


class ResnetGenerator(nn.Module):
    """
    编码器 -> 9 个残差块 -> 解码器
    默认输入 256x256
    """

    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9):
        super().__init__()

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        ]

        # 下采样
        in_dim = ngf
        out_dim = in_dim * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_dim),
                nn.ReLU(True)
            ]
            in_dim = out_dim
            out_dim *= 2

        # 残差块
        for _ in range(n_blocks):
            model += [ResnetBlock(in_dim)]

        # 上采样
        out_dim = in_dim // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_dim, out_dim, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_dim),
                nn.ReLU(True)
            ]
            in_dim = out_dim
            out_dim //= 2

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
