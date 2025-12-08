import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN"""

    def __init__(self, input_nc, ndf=64):
        super().__init__()

        def conv_layer(in_c, out_c, norm=True):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, True))
            return layers

        layers = []
        layers += conv_layer(input_nc, ndf, norm=False)
        layers += conv_layer(ndf, ndf * 2)
        layers += conv_layer(ndf * 2, ndf * 4)
        layers += conv_layer(ndf * 4, ndf * 8)

        layers += [nn.Conv2d(ndf * 8, 1, kernel_size=4, padding=1)]  # PatchGAN 输出

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
