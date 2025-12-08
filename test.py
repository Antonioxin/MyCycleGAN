# test.py

import os
import argparse

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.utils import save_image

from datasets.unpaired_dataset import UnpairedImageDataset
from models.cycleGAN_model import CycleGANModel


def get_transform(size=256):
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def denorm(tensor):
    """
    [-1,1] -> [0,1]
    """
    return (tensor + 1.0) / 2.0


def parse_args():
    parser = argparse.ArgumentParser(description="Test CycleGAN")
    parser.add_argument("--data-root", type=str, required=True,
                        help="包含 testA/testB 的根目录")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="训练好的权重路径 .pth")
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--phase", type=str, default="test", choices=["test", "train"])
    parser.add_argument("--direction", type=str, default="AtoB", choices=["AtoB", "BtoA", "both"])
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    transform = get_transform()
    dataset = UnpairedImageDataset(root=args.data_root, transform=transform, phase=args.phase)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 只需要生成器即可
    model = CycleGANModel(
        input_nc=3,
        output_nc=3,
        device=device,
    )
    model.load(args.checkpoint, load_optim=False)

    model.netG_A.eval()
    model.netG_B.eval()

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)

            if args.direction in ["AtoB", "both"]:
                fake_B = model.netG_A(real_A)
                save_image(
                    denorm(fake_B),
                    os.path.join(args.save_dir, f"{idx:04d}_fake_B.png"),
                )
                save_image(
                    denorm(real_A),
                    os.path.join(args.save_dir, f"{idx:04d}_real_A.png"),
                )

            if args.direction in ["BtoA", "both"]:
                fake_A = model.netG_B(real_B)
                save_image(
                    denorm(fake_A),
                    os.path.join(args.save_dir, f"{idx:04d}_fake_A.png"),
                )
                save_image(
                    denorm(real_B),
                    os.path.join(args.save_dir, f"{idx:04d}_real_B.png"),
                )

    print(f"Results saved to {args.save_dir}")


if __name__ == "__main__":
    main()
