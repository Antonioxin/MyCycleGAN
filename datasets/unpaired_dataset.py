# datasets/unpaired_dataset.py

import os
import io
import random
from typing import List, Optional, Tuple

from PIL import Image, ImageFile
from torch.utils.data import Dataset

# 避免少量损坏图片直接把训练打断（真实数据集里很常见）
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _is_image_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in IMG_EXTENSIONS


def _list_image_paths(dir_path: str) -> List[str]:
    """
    更稳更快的文件扫描：
    - 过滤非图片文件（.DS_Store / txt / json 等）
    - 只收集普通文件
    """
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    paths = []
    # scandir 比 listdir 更快一点点，并且能直接判断 file/dir
    with os.scandir(dir_path) as it:
        for entry in it:
            if entry.is_file():
                name = entry.name
                if _is_image_file(name):
                    paths.append(entry.path)

    paths.sort()
    return paths


class UnpairedImageDataset(Dataset):
    """
    无配对数据集 A<->B

    默认行为（更符合 CycleGAN 训练习惯）：
    - A 使用 index % len(A)
    - B 每次随机采样（unpaired 的“随机配对”）

    你仍然可以用 serial_batches=True 退回到 “B 按 index%len(B)” 的固定映射，
    方便 debug 或复现实验。
    """

    def __init__(
        self,
        root: str,
        transform=None,
        phase: str = "train",
        serial_batches: bool = False,
        max_dataset_size: int = 10**18,
        cache_images: bool = False,
        cache_format: str = "bytes",  # "bytes" 或 "none"
    ):
        self.transform = transform
        self.phase = phase
        self.serial_batches = bool(serial_batches)
        self.max_dataset_size = int(max_dataset_size)

        self.dir_A = os.path.join(root, f"{phase}A")
        self.dir_B = os.path.join(root, f"{phase}B")

        self.A_paths = _list_image_paths(self.dir_A)
        self.B_paths = _list_image_paths(self.dir_B)

        # 限制最大数据量（有时你做快速实验会很有用）
        if self.max_dataset_size > 0:
            self.A_paths = self.A_paths[: self.max_dataset_size]
            self.B_paths = self.B_paths[: self.max_dataset_size]

        self.len_A = len(self.A_paths)
        self.len_B = len(self.B_paths)

        if self.len_A == 0:
            raise RuntimeError(f"No images found in {self.dir_A}")
        if self.len_B == 0:
            raise RuntimeError(f"No images found in {self.dir_B}")

        # 可选：缓存（默认关闭，不影响你现在的训练）
        # 注意：多进程 DataLoader（Linux fork）下，这些 bytes 往往能“写时复制共享”，性价比很高
        self.cache_images = bool(cache_images)
        self.cache_format = cache_format.lower().strip()

        self.A_cache: Optional[List[bytes]] = None
        self.B_cache: Optional[List[bytes]] = None

        if self.cache_images:
            if self.cache_format != "bytes":
                raise ValueError("cache_format only supports 'bytes' in this implementation.")

            self.A_cache = [self._read_file_bytes(p) for p in self.A_paths]
            self.B_cache = [self._read_file_bytes(p) for p in self.B_paths]

    @staticmethod
    def _read_file_bytes(path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    @staticmethod
    def _load_pil_rgb_from_path(path: str) -> Image.Image:
        # 用 with 确保文件句柄及时释放
        with Image.open(path) as im:
            im = im.convert("RGB")
            # convert 会触发解码，但这里显式 load 一下更稳
            im.load()
            return im

    @staticmethod
    def _load_pil_rgb_from_bytes(b: bytes) -> Image.Image:
        with Image.open(io.BytesIO(b)) as im:
            im = im.convert("RGB")
            im.load()
            return im

    def _get_B_index(self, index: int) -> int:
        if self.serial_batches:
            return index % self.len_B
        # unpaired 的经典做法：随机抽一个 B
        return random.randrange(self.len_B)

    def __getitem__(self, index: int):
        # A：按 index 循环取
        A_index = index % self.len_A
        B_index = self._get_B_index(index)

        A_path = self.A_paths[A_index]
        B_path = self.B_paths[B_index]

        # 读取图像（可选缓存）
        if self.cache_images and self.A_cache is not None and self.B_cache is not None:
            img_A = self._load_pil_rgb_from_bytes(self.A_cache[A_index])
            img_B = self._load_pil_rgb_from_bytes(self.B_cache[B_index])
        else:
            img_A = self._load_pil_rgb_from_path(A_path)
            img_B = self._load_pil_rgb_from_path(B_path)

        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self) -> int:
        # CycleGAN 通常用 max(lenA, lenB) 作为一个 epoch 的长度
        return max(self.len_A, self.len_B)

    def __repr__(self) -> str:
        return (
            f"UnpairedImageDataset(phase={self.phase}, "
            f"len_A={self.len_A}, len_B={self.len_B}, "
            f"serial_batches={self.serial_batches}, "
            f"cache_images={self.cache_images})"
        )
