# utils/image_pool.py

import random
import torch


class ImagePool:
    """
    CycleGAN 论文中的 Image Buffer 机制
    - pool_size = 0 时等价于不使用 ImagePool
    """

    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.images = []
        else:
            self.images = None

    def query(self, images):
        """
        images: [B, C, H, W]
        返回用于训练判别器的样本
        """
        if self.pool_size == 0:
            return images

        return_images = []
        for image in images:
            image = image.detach()
            if len(self.images) < self.pool_size:
                self.images.append(image)
                return_images.append(image)
            else:
                if random.random() > 0.5:
                    # 使用历史图像替换
                    idx = random.randint(0, self.pool_size - 1)
                    tmp = self.images[idx].clone()
                    self.images[idx] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return torch.stack(return_images, dim=0)
