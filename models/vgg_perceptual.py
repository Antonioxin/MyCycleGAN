# models/vgg_perceptual.py

import torch
import torch.nn as nn
import torchvision.models as models


class VGG16FeatureExtractor(nn.Module):
    """
    提取 VGG16 某几层的特征，用于感知损失
    """
    def __init__(self, requires_grad=False):
        super().__init__()
        try:
            vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        except Exception:
            # 兼容老版本 torchvision
            vgg16 = models.vgg16(pretrained=True)

        features = list(vgg16.features.children())
        # 取前面的卷积层即可（0-16 大致对应 relu3_3）
        self.features = nn.Sequential(*features[:16])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.features(x)


class PerceptualLoss(nn.Module):
    """
    L2( VGG(fake), VGG(real) )
    """
    def __init__(self):
        super().__init__()
        self.vgg = VGG16FeatureExtractor().eval()
        self.criterion = nn.MSELoss()

    def forward(self, fake, real):
        fake_f = self.vgg(fake)
        real_f = self.vgg(real)
        return self.criterion(fake_f, real_f)
