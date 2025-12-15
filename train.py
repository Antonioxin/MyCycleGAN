# train.py
import os
import argparse
import random

import numpy as np
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


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str = ""):
    """
    兼容 Python 3.8/3.9- 的布尔开关写法：
      --name / --no-name
    """
    dest = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
    group.add_argument(f"--no-{name}", dest=dest, action="store_false", help=help_text)
    parser.set_defaults(**{dest: default})


def parse_args():
    parser = argparse.ArgumentParser(description="Train CycleGAN from scratch (GPU-optimized)")

    # --------- 基础训练参数 ---------
    parser.add_argument("--data-root", type=str, required=True,
                        help="根目录，包含 trainA/trainB/testA/testB")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-cycle", type=float, default=10.0)
    parser.add_argument("--lambda-idt", type=float, default=0.5)
    parser.add_argument("--pool-size", type=int, default=50)

    # --------- 输入尺寸 ---------
    parser.add_argument("--load-size", type=int, default=286)
    parser.add_argument("--crop-size", type=int, default=256)

    # --------- 设备与性能开关 ---------
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    add_bool_arg(parser, "tf32", default=True, help_text="Enable TF32 on CUDA (Ampere+).")

    add_bool_arg(parser, "amp", default=True, help_text="Enable AMP mixed precision.")
    parser.add_argument("--amp-dtype", type=str, default="auto",
                        choices=["auto", "bf16", "fp16", "fp32"],
                        help="auto=prefer bf16 else fp16; fp32 disables amp")

    add_bool_arg(parser, "channels-last", default=True, help_text="Use channels_last memory format (CUDA).")
    add_bool_arg(parser, "fused-adam", default=True, help_text="Use fused Adam if supported by torch/cuda.")

    add_bool_arg(parser, "compile", default=False, help_text="Enable torch.compile (CUDA).")
    parser.add_argument("--compile-mode", type=str, default="max-autotune",
                        choices=["default", "reduce-overhead", "max-autotune"])

    # --------- 数据加载 ---------
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4,
                        help="batches prefetched per worker (num_workers>0)")
    add_bool_arg(parser, "pin-memory", default=True, help_text="DataLoader pin_memory (CUDA).")
    add_bool_arg(parser, "drop-last", default=True, help_text="Drop last incomplete batch.")

    add_bool_arg(parser, "limit-cpu-threads", default=True,
                 help_text="Limit torch CPU threads when using multi-worker loader.")

    # --------- 日志与保存 ---------
    parser.add_argument("--print-freq", type=int, default=100)
    parser.add_argument("--save-freq", type=int, default=5)
    parser.add_argument("--checkpoints-dir", type=str, default="./checkpoints")
    add_bool_arg(parser, "save-optim", default=True, help_text="Save optimizer & scaler states.")

    # --------- 可选：感知损失 ---------
    add_bool_arg(parser, "perceptual", default=False, help_text="Enable perceptual loss.")
    parser.add_argument("--perceptual-weight", type=float, default=0.0)

    # --------- ✅ 可选：CBAM（只加在最后 N 个残差块）---------
    add_bool_arg(parser, "cbam", default=False, help_text="Enable CBAM in generators (only last N ResNet blocks).")
    parser.add_argument("--cbam-last-n", type=int, default=3,
                        help="Apply CBAM to the last N residual blocks. (0 disables even if --cbam)")
    parser.add_argument("--cbam-ratio", type=int, default=16, help="CBAM channel reduction ratio.")
    parser.add_argument("--cbam-kernel-size", type=int, default=7, help="CBAM spatial attention kernel size (odd).")

    # --------- 复现 / 续训 ---------
    parser.add_argument("--seed", type=int, default=-1, help=">=0 set seed; -1 means no seed (faster).")
    parser.add_argument("--resume", type=str, default="", help="checkpoint path to resume from")
    add_bool_arg(parser, "resume-optim", default=True, help_text="Load optimizer & scaler when resuming.")

    return parser.parse_args()


def main():
    args = parse_args()

    # ---- CBAM 参数合法性处理（尽量不让你踩坑）----
    if args.cbam_last_n < 0:
        raise ValueError("--cbam-last-n must be >= 0")
    if args.cbam_kernel_size <= 1 or args.cbam_kernel_size % 2 == 0:
        raise ValueError("--cbam-kernel-size must be an odd integer > 1")
    if args.cbam_ratio <= 0:
        raise ValueError("--cbam-ratio must be > 0")

    use_cuda = (args.device == "cuda" and torch.cuda.is_available())
    device = "cuda" if use_cuda else "cpu"

    os.makedirs(args.checkpoints_dir, exist_ok=True)

    if args.seed is not None and args.seed >= 0:
        seed_everything(args.seed)

    # CUDA 性能设置
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        if args.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

    # 避免 CPU 线程超卖（多 worker 时更稳）
    if args.limit_cpu_threads and args.num_workers > 0:
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

    # Data
    transform = get_transform(load_size=args.load_size, crop_size=args.crop_size)
    train_dataset = UnpairedImageDataset(root=args.data_root, transform=transform, phase="train")

    dl_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda" and args.pin_memory),
        drop_last=args.drop_last,
    )
    if args.num_workers > 0:
        dl_kwargs.update(
            persistent_workers=True,
            prefetch_factor=args.prefetch_factor,
        )
    train_loader = DataLoader(train_dataset, **dl_kwargs)

    # AMP dtype
    amp_dtype = args.amp_dtype
    if amp_dtype == "auto":
        amp_dtype = "fp16"
    if amp_dtype == "fp32":
        args.amp = False

    # ---- CBAM kwargs ----
    cbam_kwargs = {
        "ratio": args.cbam_ratio,
        "kernel_size": args.cbam_kernel_size,
    }

    # Model
    model = CycleGANModel(
        input_nc=3,
        output_nc=3,
        ngf=64,
        ndf=64,
        lambda_cycle=args.lambda_cycle,
        lambda_idt=args.lambda_idt,
        lr=args.lr,
        pool_size=args.pool_size,
        device=device,

        use_amp=args.amp,
        amp_dtype=amp_dtype,
        channels_last=args.channels_last,
        use_fused_adam=args.fused_adam,
        use_compile=args.compile,
        compile_mode=args.compile_mode,

        use_perceptual_loss=args.perceptual,
        perceptual_weight=args.perceptual_weight,

        # ✅ CBAM（只加最后 N 个残差块）
        use_cbam=args.cbam,
        cbam_last_n=args.cbam_last_n,
        cbam_kwargs=cbam_kwargs,
    )

    # Resume
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"[Resume] Loading checkpoint: {args.resume} (optim={args.resume_optim})")
            model.load(args.resume, load_optim=args.resume_optim)
        else:
            print(f"[WARN] resume path not found: {args.resume}")

    logger = TrainLogger()
    iters_per_epoch = len(train_loader)

    for epoch in range(1, args.epochs + 1):
        for i, batch in enumerate(train_loader, start=1):
            model.set_input(batch)
            model.optimize_parameters()

            if i % args.print_freq == 0 or i == 1 or i == iters_per_epoch:
                losses = model.get_current_losses()
                logger.log(epoch, i, iters_per_epoch, losses)

        if epoch % args.save_freq == 0 or epoch == args.epochs:
            save_path = os.path.join(args.checkpoints_dir, f"epoch_{epoch}.pth")
            model.save(save_path, save_optim=args.save_optim)
            print(f"Saved checkpoint: {save_path}")


if __name__ == "__main__":
    main()
