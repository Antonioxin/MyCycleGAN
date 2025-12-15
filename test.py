# # test.py

# import os
# import argparse
# from contextlib import nullcontext

# import torch
# from torch.utils.data import DataLoader
# import torchvision.transforms as T
# from torchvision.transforms import InterpolationMode
# from torchvision.utils import save_image

# from datasets.unpaired_dataset import UnpairedImageDataset
# from models.cycleGAN_model import CycleGANModel


# def get_transform(size=256):
#     return T.Compose([
#         T.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
#         T.ToTensor(),
#         T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ])


# def denorm(x: torch.Tensor) -> torch.Tensor:
#     """[-1, 1] -> [0, 1]"""
#     return (x + 1.0) / 2.0


# def resolve_amp_dtype(amp_dtype: str):
#     """
#     auto: 优先 bf16（如果支持），否则 fp16
#     fp32: 等价于关闭 amp
#     """
#     s = str(amp_dtype).lower().strip()
#     if s == "auto":
#         if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
#             return torch.bfloat16
#         return torch.float16
#     if s in ("bf16", "bfloat16"):
#         if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
#             return torch.bfloat16
#         return torch.float16
#     if s in ("fp16", "float16", "16"):
#         return torch.float16
#     return None  # fp32 或未知 -> 不启用 autocast


# def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str = ""):
#     """
#     兼容 Python 3.8/3.9- 的布尔开关写法：
#       --name / --no-name
#     """
#     dest = name.replace("-", "_")
#     group = parser.add_mutually_exclusive_group(required=False)
#     group.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
#     group.add_argument(f"--no-{name}", dest=dest, action="store_false", help=help_text)
#     parser.set_defaults(**{dest: default})


# def parse_args():
#     parser = argparse.ArgumentParser(description="Test CycleGAN (GPU-optimized, Py3.8+)")

#     parser.add_argument("--data-root", type=str, required=True,
#                         help="包含 testA/testB 的根目录")
#     parser.add_argument("--checkpoint", type=str, required=True,
#                         help="训练好的权重路径 .pth")
#     parser.add_argument("--save-dir", type=str, default="./results")
#     parser.add_argument("--phase", type=str, default="test", choices=["test", "train"])
#     parser.add_argument("--direction", type=str, default="AtoB", choices=["AtoB", "BtoA", "both"])
#     parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

#     # 输入尺寸
#     parser.add_argument("--size", type=int, default=256)

#     # 推理吞吐
#     parser.add_argument("--batch-size", type=int, default=8)
#     parser.add_argument("--num-workers", type=int, default=8)
#     parser.add_argument("--prefetch-factor", type=int, default=4)
#     add_bool_arg(parser, "pin-memory", default=True, help_text="DataLoader pin_memory (CUDA).")

#     # AMP 推理
#     add_bool_arg(parser, "amp", default=True, help_text="Enable AMP for inference on CUDA.")
#     parser.add_argument("--amp-dtype", type=str, default="auto",
#                         choices=["auto", "bf16", "fp16", "fp32"])

#     # channels_last（与训练保持一致）
#     add_bool_arg(parser, "channels-last", default=True, help_text="Use channels_last on CUDA.")

#     # test 阶段通常希望“顺序一致”
#     add_bool_arg(parser, "serial-batches", default=True, help_text="Use deterministic B selection (if supported).")

#     # 保存方式
#     parser.add_argument("--save-mode", type=str, default="pair", choices=["pair", "separate"])
#     parser.add_argument("--max-images", type=int, default=-1, help=">0 则最多保存这么多张")

#     return parser.parse_args()


# def main():
#     args = parse_args()

#     use_cuda = (args.device == "cuda" and torch.cuda.is_available())
#     device = "cuda" if use_cuda else "cpu"
#     os.makedirs(args.save_dir, exist_ok=True)

#     # 输出子目录
#     dir_AtoB = os.path.join(args.save_dir, "AtoB")
#     dir_BtoA = os.path.join(args.save_dir, "BtoA")
#     if args.direction in ("AtoB", "both"):
#         os.makedirs(dir_AtoB, exist_ok=True)
#     if args.direction in ("BtoA", "both"):
#         os.makedirs(dir_BtoA, exist_ok=True)

#     transform = get_transform(size=args.size)

#     # 兼容新/旧版 UnpairedImageDataset：新版本有 serial_batches 参数，旧版没有
#     try:
#         dataset = UnpairedImageDataset(
#             root=args.data_root,
#             transform=transform,
#             phase=args.phase,
#             serial_batches=args.serial_batches,
#         )
#     except TypeError:
#         dataset = UnpairedImageDataset(
#             root=args.data_root,
#             transform=transform,
#             phase=args.phase,
#         )

#     dl_kwargs = dict(
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers,
#         pin_memory=(use_cuda and args.pin_memory),
#         drop_last=False,
#     )
#     if args.num_workers > 0:
#         dl_kwargs.update(
#             persistent_workers=True,
#             prefetch_factor=args.prefetch_factor,
#         )
#     loader = DataLoader(dataset, **dl_kwargs)

#     # 用 CycleGANModel 来 load（兼容你们 checkpoint 结构）
#     # 兼容新/旧版 CycleGANModel：新版本支持 use_amp/channels_last 等参数，旧版没有
#     try:
#         model = CycleGANModel(
#             input_nc=3,
#             output_nc=3,
#             device=device,
#             use_amp=args.amp,
#             amp_dtype=("bf16" if args.amp_dtype == "auto" else args.amp_dtype),
#             channels_last=args.channels_last,
#             use_fused_adam=False,
#             use_compile=False,
#         )
#     except TypeError:
#         model = CycleGANModel(
#             input_nc=3,
#             output_nc=3,
#             device=device,
#         )

#     model.load(args.checkpoint, load_optim=False)
#     model.netG_A.eval()
#     model.netG_B.eval()

#     # AMP autocast 上下文（只在 CUDA 且 amp 开启时启用）
#     amp_dtype = resolve_amp_dtype(args.amp_dtype) if (use_cuda and args.amp) else None
#     autocast_ctx = (
#         torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
#         if (use_cuda and amp_dtype is not None)
#         else nullcontext()
#     )

#     saved = 0
#     global_idx = 0

#     # inference_mode() 更适合推理
#     with torch.inference_mode():
#         for batch in loader:
#             real_A = batch["A"].to(device, non_blocking=True)
#             real_B = batch["B"].to(device, non_blocking=True)

#             if args.channels_last and use_cuda:
#                 real_A = real_A.contiguous(memory_format=torch.channels_last)
#                 real_B = real_B.contiguous(memory_format=torch.channels_last)

#             bs = real_A.size(0)

#             with autocast_ctx:
#                 fake_B = model.netG_A(real_A) if args.direction in ("AtoB", "both") else None
#                 fake_A = model.netG_B(real_B) if args.direction in ("BtoA", "both") else None

#             # 保存 A->B
#             if args.direction in ("AtoB", "both"):
#                 real_A_vis = denorm(real_A).clamp(0, 1).cpu()
#                 fake_B_vis = denorm(fake_B).clamp(0, 1).cpu()

#                 for b in range(bs):
#                     idx = global_idx + b
#                     if args.save_mode == "pair":
#                         pair = torch.cat([real_A_vis[b], fake_B_vis[b]], dim=2)  # (C,H,W*2)
#                         save_image(pair, os.path.join(dir_AtoB, "%06d_AtoB_pair.png" % idx))
#                     else:
#                         save_image(real_A_vis[b], os.path.join(dir_AtoB, "%06d_real_A.png" % idx))
#                         save_image(fake_B_vis[b], os.path.join(dir_AtoB, "%06d_fake_B.png" % idx))

#                     saved += 1
#                     if args.max_images > 0 and saved >= args.max_images:
#                         print("[Done] Saved %d images to %s" % (saved, args.save_dir))
#                         return

#             # 保存 B->A
#             if args.direction in ("BtoA", "both"):
#                 real_B_vis = denorm(real_B).clamp(0, 1).cpu()
#                 fake_A_vis = denorm(fake_A).clamp(0, 1).cpu()

#                 for b in range(bs):
#                     idx = global_idx + b
#                     if args.save_mode == "pair":
#                         pair = torch.cat([real_B_vis[b], fake_A_vis[b]], dim=2)  # (C,H,W*2)
#                         save_image(pair, os.path.join(dir_BtoA, "%06d_BtoA_pair.png" % idx))
#                     else:
#                         save_image(real_B_vis[b], os.path.join(dir_BtoA, "%06d_real_B.png" % idx))
#                         save_image(fake_A_vis[b], os.path.join(dir_BtoA, "%06d_fake_A.png" % idx))

#                     saved += 1
#                     if args.max_images > 0 and saved >= args.max_images:
#                         print("[Done] Saved %d images to %s" % (saved, args.save_dir))
#                         return

#             global_idx += bs

#     print("[Done] Saved %d images to %s" % (saved, args.save_dir))


# if __name__ == "__main__":
#     main()


# test.py

import os
import re
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


def infer_cbam_from_checkpoint(ckpt_path: str):
    """
    从 checkpoint(netG_A 的 state_dict) 推断：
    - 是否启用 CBAM
    - cbam_last_n = 有 cbam 参数的 ResnetBlock 数量
    - cbam_ratio / cbam_kernel_size（根据参数 shape 推断）
    适配你们保存格式：{"netG_A": state_dict, ...}
    """
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "netG_A" in state:
        sd = state["netG_A"]
    elif isinstance(state, dict):
        # 万一用户传的是纯 state_dict
        sd = state
    else:
        return dict(use_cbam=False, cbam_last_n=0, cbam_ratio=16, cbam_kernel_size=7)

    cbam_param_keys = [k for k in sd.keys() if ".cbam." in k]
    if not cbam_param_keys:
        return dict(use_cbam=False, cbam_last_n=0, cbam_ratio=16, cbam_kernel_size=7)

    # 1) 推断 cbam_last_n：统计包含 cbam 的 ResnetBlock（Sequential 的 index）
    # 生成器 key 形如：model.10.cbam.ca.mlp.0.weight
    idx_set = set()
    pat = re.compile(r"^model\.(\d+)\.cbam\.")
    for k in cbam_param_keys:
        m = pat.match(k)
        if m:
            idx_set.add(int(m.group(1)))
    cbam_last_n = len(idx_set) if idx_set else 0

    # 2) 推断 kernel_size：从 spatial attention conv 的 weight shape: (1, 2, k, k)
    cbam_kernel_size = 7
    sa_w_key = None
    for k in cbam_param_keys:
        if k.endswith("cbam.sa.conv.weight"):
            sa_w_key = k
            break
    if sa_w_key is not None:
        w = sd[sa_w_key]
        if hasattr(w, "shape") and len(w.shape) == 4:
            cbam_kernel_size = int(w.shape[-1])

    # 3) 推断 ratio：从 channel attention 第一个 1x1 conv weight shape: (hidden, in_planes, 1, 1)
    cbam_ratio = 16
    ca_w_key = None
    for k in cbam_param_keys:
        if k.endswith("cbam.ca.mlp.0.weight"):
            ca_w_key = k
            break
    if ca_w_key is not None:
        w = sd[ca_w_key]
        if hasattr(w, "shape") and len(w.shape) == 4:
            hidden = int(w.shape[0])
            in_planes = int(w.shape[1])
            if hidden > 0:
                cbam_ratio = max(1, in_planes // hidden)

    return dict(
        use_cbam=True,
        cbam_last_n=cbam_last_n,
        cbam_ratio=cbam_ratio,
        cbam_kernel_size=cbam_kernel_size,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Test CycleGAN (CBAM-aware, GPU-optimized, Py3.8+)")

    parser.add_argument("--data-root", type=str, required=True, help="包含 testA/testB 的根目录")
    parser.add_argument("--checkpoint", type=str, required=True, help="训练好的权重路径 .pth")
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
    parser.add_argument("--amp-dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])

    # channels_last（与训练保持一致）
    add_bool_arg(parser, "channels-last", default=True, help_text="Use channels_last on CUDA.")

    # test 阶段通常希望“顺序一致”
    add_bool_arg(parser, "serial-batches", default=True, help_text="Use deterministic B selection (if supported).")

    # 保存方式
    parser.add_argument("--save-mode", type=str, default="pair", choices=["pair", "separate"])
    parser.add_argument("--max-images", type=int, default=-1, help=">0 则最多保存这么多张")

    # ✅ CBAM 推理配置（默认 auto：从 checkpoint 推断）
    parser.add_argument("--cbam-mode", type=str, default="auto", choices=["auto", "on", "off"],
                        help="auto: infer from ckpt; on/off: force enable/disable")
    parser.add_argument("--cbam-last-n", type=int, default=-1,
                        help=">=0 强制指定最后 N 个残差块用 CBAM；-1 表示从 ckpt 推断")
    parser.add_argument("--cbam-ratio", type=int, default=-1,
                        help=">0 强制指定 CBAM ratio；-1 表示从 ckpt 推断")
    parser.add_argument("--cbam-kernel-size", type=int, default=-1,
                        help=">1 且为奇数时强制指定 kernel；-1 表示从 ckpt 推断")

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

    # --------- ✅ 从 checkpoint 推断 CBAM 配置（或按用户强制设置）---------
    inferred = infer_cbam_from_checkpoint(args.checkpoint)

    if args.cbam_mode == "off":
        use_cbam = False
        cbam_last_n = 0
        cbam_ratio = 16
        cbam_kernel = 7
    else:
        # on / auto 都会启用（auto 会按 ckpt 决定是否启用）
        use_cbam = inferred["use_cbam"] if args.cbam_mode == "auto" else True

        # last_n / ratio / kernel 允许用户强制覆盖；否则用推断值
        cbam_last_n = inferred["cbam_last_n"] if args.cbam_last_n < 0 else int(args.cbam_last_n)
        cbam_ratio = inferred["cbam_ratio"] if args.cbam_ratio < 0 else int(args.cbam_ratio)
        cbam_kernel = inferred["cbam_kernel_size"] if args.cbam_kernel_size < 0 else int(args.cbam_kernel_size)

        # 基本合法性
        if cbam_last_n < 0:
            cbam_last_n = 0
        if cbam_ratio <= 0:
            cbam_ratio = 16
        if cbam_kernel <= 1 or cbam_kernel % 2 == 0:
            cbam_kernel = 7

    cbam_kwargs = {"ratio": cbam_ratio, "kernel_size": cbam_kernel}

    print("[CBAM] mode=%s | inferred=%s" % (args.cbam_mode, inferred))
    print("[CBAM] using: use_cbam=%s, cbam_last_n=%d, ratio=%d, kernel=%d"
          % (use_cbam, cbam_last_n, cbam_ratio, cbam_kernel))

    # --------- 构建模型并 load ---------
    # 兼容旧版 CycleGANModel：旧版可能不支持这些参数
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

            # ✅ CBAM（关键：让结构与 ckpt 一致）
            use_cbam=use_cbam,
            cbam_last_n=cbam_last_n,
            cbam_kwargs=cbam_kwargs,
        )
    except TypeError:
        # 旧版不支持 cbam 参数 -> 只能按“无 cbam”建模（加载 cbam ckpt 会失败）
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
