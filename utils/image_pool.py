# utils/image_pool.py

import random
import torch


class ImagePool:
    """
    CycleGAN 论文中的 Image Buffer 机制（历史假样本池）

    关键点：
    - 入池时必须 clone()，避免保存 “切片视图” 导致引用整个 batch 的底层 storage，
      进而造成显存/内存占用持续增长、训练越来越慢甚至 OOM。
    - 支持 images 为 Tensor([B,C,H,W]) 或 Tensor([C,H,W])。
    - pool_size = 0 时等价于不使用 ImagePool（直接返回输入）。
    """

    def __init__(self, pool_size=50):
        self.pool_size = int(pool_size)
        self.images = [] if self.pool_size > 0 else None

    def query(self, images: torch.Tensor) -> torch.Tensor:
        """
        输入：
            images: Tensor [B, C, H, W] 或 [C, H, W]
        输出：
            Tensor [B, C, H, W]（用于训练判别器）
        """
        if self.pool_size == 0:
            return images

        if images is None:
            return None

        if not torch.is_tensor(images):
            raise TypeError(f"ImagePool.query expects a torch.Tensor, got {type(images)}")

        # 兼容单张 [C,H,W]
        if images.dim() == 3:
            images = images.unsqueeze(0)

        if images.dim() != 4:
            raise ValueError(f"images should have 3 or 4 dims, got {images.dim()} dims")

        # 注意：这里不做 .cpu()，保持在原 device（通常是 GPU），避免来回搬运拖慢训练
        batch_size = images.size(0)
        return_images = []

        for i in range(batch_size):
            img = images[i].detach()

            # 池子没满：直接入池并返回当前图（入池要 clone，断开对 batch storage 的引用）
            if len(self.images) < self.pool_size:
                self.images.append(img.clone())
                return_images.append(img)
            else:
                # 50% 概率用历史图像替换
                if random.random() > 0.5:
                    idx = random.randint(0, self.pool_size - 1)
                    old = self.images[idx]
                    # 用当前图替换池中图（同样要 clone）
                    self.images[idx] = img.clone()
                    return_images.append(old)
                else:
                    return_images.append(img)

        # 返回 [B,C,H,W]
        return torch.stack(return_images, dim=0)
