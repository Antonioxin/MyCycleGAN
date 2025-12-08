# train.py

import os
import argparse

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

from datasets.unpaired_dataset import UnpairedImageDataset
from models.cycleGAN_model import CycleGANModel
from utils.logger import TrainLogger


def get_transform(load_size=286, crop_size=256):
    return T.Compose([
        T.Resize((load_size, load_size), InterpolationMode.BICUBIC),
        T.RandomCrop((crop_size, crop_size)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def parse_args():
    parser = argparse.ArgumentParser(description="Train CycleGAN from scratch")
    parser.add_argument("--data-root", type=str, required=True,
                        help="根目录，包含 trainA/trainB/testA/testB")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-cycle", type=float, default=10.0)
    parser.add_argument("--lambda-idt", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--print-freq", type=int, default=100)
    parser.add_argument("--save-freq", type=int, default=5)
    parser.add_argument("--checkpoints-dir", type=str, default="./checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    os.makedirs(args.checkpoints_dir, exist_ok=True)

    # 数据集 & DataLoader
    transform = get_transform()
    train_dataset = UnpairedImageDataset(root=args.data_root, transform=transform, phase="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 模型
    model = CycleGANModel(
        input_nc=3,
        output_nc=3,
        ngf=64,
        ndf=64,
        lambda_cycle=args.lambda_cycle,
        lambda_idt=args.lambda_idt,
        lr=args.lr,
        device=device,
        use_perceptual_loss=False,  # 基础版先关掉
    )

    logger = TrainLogger()

    total_iters_per_epoch = len(train_loader)

    for epoch in range(1, args.epochs + 1):
        for i, batch in enumerate(train_loader, start=1):
            model.set_input(batch)
            model.optimize_parameters()

            if i % args.print_freq == 0 or i == 1 or i == total_iters_per_epoch:
                losses = model.get_current_losses()
                logger.log(epoch, i, total_iters_per_epoch, losses)

        # 保存 ckpt
        if epoch % args.save_freq == 0 or epoch == args.epochs:
            save_path = os.path.join(args.checkpoints_dir, f"epoch_{epoch}.pth")
            model.save(save_path)
            print(f"Saved checkpoint: {save_path}")


if __name__ == "__main__":
    main()
