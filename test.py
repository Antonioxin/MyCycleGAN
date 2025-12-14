# test.py

import os
import argparse
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image

from datasets.unpaired_dataset import UnpairedImageDataset
from models.cycleGAN_model import CycleGANModel


def get_transform(size=256):
    return T.Compose([
        T.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def denorm(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1]"""
    return (x + 1.0) / 2.0


def resolve_amp_dtype(amp_dtype: str):
    """
    auto: 优先 bf16（如果支持），否则 fp16
    fp32: 等价于关闭 amp
    """
    s = str(amp_dtype).lower().strip()
    if s == "auto":
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if s in ("bf16", "bfloat16"):
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if s in ("fp16", "float16", "16"):
        return torch.float16
    return None  # fp32 或未知 -> 不启用 autocast


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
    parser = argparse.ArgumentParser(description="Test CycleGAN (GPU-optimized, Py3.8+)")

    parser.add_argument("--data-root", type=str, required=True,
                        help="包含 testA/testB 的根目录")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="训练好的权重路径 .pth")
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--phase", type=str, default="test", choices=["test", "train"])
    parser.add_argument("--direction", type=str, default="AtoB", choices=["AtoB", "BtoA", "both"])
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    # 输入尺寸
    parser.add_argument("--size", type=int, default=256)

    # 推理吞吐
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    add_bool_arg(parser, "pin-memory", default=True, help_text="DataLoader pin_memory (CUDA).")

    # AMP 推理
    add_bool_arg(parser, "amp", default=True, help_text="Enable AMP for inference on CUDA.")
    parser.add_argument("--amp-dtype", type=str, default="auto",
                        choices=["auto", "bf16", "fp16", "fp32"])

    # channels_last（与训练保持一致）
    add_bool_arg(parser, "channels-last", default=True, help_text="Use channels_last on CUDA.")

    # test 阶段通常希望“顺序一致”
    add_bool_arg(parser, "serial-batches", default=True, help_text="Use deterministic B selection (if supported).")

    # 保存方式
    parser.add_argument("--save-mode", type=str, default="pair", choices=["pair", "separate"])
    parser.add_argument("--max-images", type=int, default=-1, help=">0 则最多保存这么多张")

    return parser.parse_args()


def main():
    args = parse_args()

    use_cuda = (args.device == "cuda" and torch.cuda.is_available())
    device = "cuda" if use_cuda else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    # 输出子目录
    dir_AtoB = os.path.join(args.save_dir, "AtoB")
    dir_BtoA = os.path.join(args.save_dir, "BtoA")
    if args.direction in ("AtoB", "both"):
        os.makedirs(dir_AtoB, exist_ok=True)
    if args.direction in ("BtoA", "both"):
        os.makedirs(dir_BtoA, exist_ok=True)

    transform = get_transform(size=args.size)

    # 兼容新/旧版 UnpairedImageDataset：新版本有 serial_batches 参数，旧版没有
    try:
        dataset = UnpairedImageDataset(
            root=args.data_root,
            transform=transform,
            phase=args.phase,
            serial_batches=args.serial_batches,
        )
    except TypeError:
        dataset = UnpairedImageDataset(
            root=args.data_root,
            transform=transform,
            phase=args.phase,
        )

    dl_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(use_cuda and args.pin_memory),
        drop_last=False,
    )
    if args.num_workers > 0:
        dl_kwargs.update(
            persistent_workers=True,
            prefetch_factor=args.prefetch_factor,
        )
    loader = DataLoader(dataset, **dl_kwargs)

    # 用 CycleGANModel 来 load（兼容你们 checkpoint 结构）
    # 兼容新/旧版 CycleGANModel：新版本支持 use_amp/channels_last 等参数，旧版没有
    try:
        model = CycleGANModel(
            input_nc=3,
            output_nc=3,
            device=device,
            use_amp=args.amp,
            amp_dtype=("bf16" if args.amp_dtype == "auto" else args.amp_dtype),
            channels_last=args.channels_last,
            use_fused_adam=False,
            use_compile=False,
        )
    except TypeError:
        model = CycleGANModel(
            input_nc=3,
            output_nc=3,
            device=device,
        )

    model.load(args.checkpoint, load_optim=False)
    model.netG_A.eval()
    model.netG_B.eval()

    # AMP autocast 上下文（只在 CUDA 且 amp 开启时启用）
    amp_dtype = resolve_amp_dtype(args.amp_dtype) if (use_cuda and args.amp) else None
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
        if (use_cuda and amp_dtype is not None)
        else nullcontext()
    )

    saved = 0
    global_idx = 0

    # inference_mode() 更适合推理
    with torch.inference_mode():
        for batch in loader:
            real_A = batch["A"].to(device, non_blocking=True)
            real_B = batch["B"].to(device, non_blocking=True)

            if args.channels_last and use_cuda:
                real_A = real_A.contiguous(memory_format=torch.channels_last)
                real_B = real_B.contiguous(memory_format=torch.channels_last)

            bs = real_A.size(0)

            with autocast_ctx:
                fake_B = model.netG_A(real_A) if args.direction in ("AtoB", "both") else None
                fake_A = model.netG_B(real_B) if args.direction in ("BtoA", "both") else None

            # 保存 A->B
            if args.direction in ("AtoB", "both"):
                real_A_vis = denorm(real_A).clamp(0, 1).cpu()
                fake_B_vis = denorm(fake_B).clamp(0, 1).cpu()

                for b in range(bs):
                    idx = global_idx + b
                    if args.save_mode == "pair":
                        pair = torch.cat([real_A_vis[b], fake_B_vis[b]], dim=2)  # (C,H,W*2)
                        save_image(pair, os.path.join(dir_AtoB, "%06d_AtoB_pair.png" % idx))
                    else:
                        save_image(real_A_vis[b], os.path.join(dir_AtoB, "%06d_real_A.png" % idx))
                        save_image(fake_B_vis[b], os.path.join(dir_AtoB, "%06d_fake_B.png" % idx))

                    saved += 1
                    if args.max_images > 0 and saved >= args.max_images:
                        print("[Done] Saved %d images to %s" % (saved, args.save_dir))
                        return

            # 保存 B->A
            if args.direction in ("BtoA", "both"):
                real_B_vis = denorm(real_B).clamp(0, 1).cpu()
                fake_A_vis = denorm(fake_A).clamp(0, 1).cpu()

                for b in range(bs):
                    idx = global_idx + b
                    if args.save_mode == "pair":
                        pair = torch.cat([real_B_vis[b], fake_A_vis[b]], dim=2)  # (C,H,W*2)
                        save_image(pair, os.path.join(dir_BtoA, "%06d_BtoA_pair.png" % idx))
                    else:
                        save_image(real_B_vis[b], os.path.join(dir_BtoA, "%06d_real_B.png" % idx))
                        save_image(fake_A_vis[b], os.path.join(dir_BtoA, "%06d_fake_A.png" % idx))

                    saved += 1
                    if args.max_images > 0 and saved >= args.max_images:
                        print("[Done] Saved %d images to %s" % (saved, args.save_dir))
                        return

            global_idx += bs

    print("[Done] Saved %d images to %s" % (saved, args.save_dir))


if __name__ == "__main__":
    main()
