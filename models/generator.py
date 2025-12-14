# models/generator.py

import torch
import torch.nn as nn
from typing import Optional, Dict


class ResnetBlock(nn.Module):
    """
    基础残差块（CycleGAN-ResNet）
    - 默认行为与原版一致
    - cbam 默认 nn.Identity()，避免 forward 中动态分支（更利于 torch.compile）
    """
    def __init__(
        self,
        dim: int,
        norm_layer=nn.InstanceNorm2d,
        use_dropout: bool = False,
        dropout_p: float = 0.0,
        use_bias: bool = True,
        cbam: Optional[nn.Module] = None,   # ✅ 兼容 Py<3.10
    ):
        super().__init__()

        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(inplace=True),
        ]

        if use_dropout and dropout_p > 0:
            layers.append(nn.Dropout(dropout_p))

        layers += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias),
            norm_layer(dim),
        ]

        self.conv_block = nn.Sequential(*layers)
        self.cbam = cbam if cbam is not None else nn.Identity()

    def forward(self, x):
        out = self.conv_block(x)
        out = self.cbam(out)
        return x + out


class ResnetGenerator(nn.Module):
    """
    编码器 -> N 个残差块 -> 解码器
    默认输入 256x256
    """

    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        ngf: int = 64,
        n_blocks: int = 9,
        norm_layer=nn.InstanceNorm2d,
        use_dropout: bool = False,
        dropout_p: float = 0.0,
        upsample_mode: str = "transpose",  # "transpose" | "nearest"
        use_cbam: bool = False,
        cbam_kwargs: Optional[Dict] = None,  # ✅ 兼容 Py<3.10
    ):
        super().__init__()
        assert n_blocks >= 0, "n_blocks must be >= 0"
        assert upsample_mode in ["transpose", "nearest"], "upsample_mode must be 'transpose' or 'nearest'"

        # InstanceNorm2d 默认 affine=False，CycleGAN 经典实现常用 conv bias=True
        use_bias = True

        cbam_kwargs = cbam_kwargs or {}
        cbam_factory = None
        if use_cbam:
            try:
                from .cbam import CBAM
                cbam_factory = lambda ch: CBAM(ch, **cbam_kwargs)
            except Exception as e:
                raise ImportError(
                    "use_cbam=True 但无法导入 CBAM。请确认 models/cbam.py 中存在 CBAM 类。"
                ) from e

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(inplace=True),
        ]

        # 下采样 x2
        in_dim = ngf
        out_dim = in_dim * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(out_dim),
                nn.ReLU(inplace=True),
            ]
            in_dim = out_dim
            out_dim *= 2

        # 残差块
        for _ in range(n_blocks):
            cbam = cbam_factory(in_dim) if cbam_factory is not None else None
            model += [
                ResnetBlock(
                    in_dim,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    dropout_p=dropout_p,
                    use_bias=use_bias,
                    cbam=cbam,
                )
            ]

        # 上采样 x2
        out_dim = in_dim // 2
        for _ in range(2):
            if upsample_mode == "transpose":
                model += [
                    nn.ConvTranspose2d(
                        in_dim, out_dim, kernel_size=3, stride=2,
                        padding=1, output_padding=1, bias=use_bias
                    ),
                    norm_layer(out_dim),
                    nn.ReLU(inplace=True),
                ]
            else:
                # 更稳的上采样：nearest + conv（更少棋盘格）
                model += [
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=0, bias=use_bias),
                    norm_layer(out_dim),
                    nn.ReLU(inplace=True),
                ]

            in_dim = out_dim
            out_dim //= 2

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, bias=True),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
