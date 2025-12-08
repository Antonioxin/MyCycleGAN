import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class UnpairedImageDataset(Dataset):
    """
    用于加载无配对数据 A->B 的数据集
    支持后续加入更多增强方式（随机裁剪、随机翻转）
    """

    def __init__(self, root, transform=None, phase="train"):
        self.transform = transform
        self.phase = phase

        self.dir_A = os.path.join(root, f"{phase}A")
        self.dir_B = os.path.join(root, f"{phase}B")

        self.A_paths = sorted([os.path.join(self.dir_A, f) for f in os.listdir(self.dir_A)])
        self.B_paths = sorted([os.path.join(self.dir_B, f) for f in os.listdir(self.dir_B)])

        self.len_A = len(self.A_paths)
        self.len_B = len(self.B_paths)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.len_A]
        B_path = self.B_paths[index % self.len_B]

        img_A = Image.open(A_path).convert("RGB")
        img_B = Image.open(B_path).convert("RGB")

        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return max(self.len_A, self.len_B)
