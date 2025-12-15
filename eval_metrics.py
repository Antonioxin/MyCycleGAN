#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_metrics.py
CycleGAN (horse2zebra) evaluation script for:
- FID / KID (InceptionV3 features)
- PRDC (Precision/Recall/Density/Coverage) in feature space
- Cycle consistency L1
- Identity L1
- Background Edge-F1 (structure preservation on background)
- Leakage Energy Ratio (change energy outside main subject component)

Default test paths (as you provided):
/root/autodl-tmp/MyCycleGAN/data/horse2zebra/testA
/root/autodl-tmp/MyCycleGAN/data/horse2zebra/testB

Example:
python eval_metrics.py \
  --checkpoint /root/autodl-tmp/MyCycleGAN/checkpoints/latest.pth \
  --data-root /root/autodl-tmp/MyCycleGAN/data/horse2zebra \
  --device cuda \
  --batch-size 16 \
  --save-dir ./eval_out
"""

import os
import sys
import json
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
# from torchvision.models import inception_v3, Inception_V3_Weights
# from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import inception_v3
from tqdm import tqdm

import re


# --------------------------
# Basic utils
# --------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(dir_path: str) -> List[str]:
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    paths = []
    with os.scandir(dir_path) as it:
        for e in it:
            if e.is_file():
                ext = os.path.splitext(e.name)[1].lower()
                if ext in IMG_EXTS:
                    paths.append(e.path)
    paths.sort()
    return paths


def denorm(x: torch.Tensor) -> torch.Tensor:
    """[-1,1] -> [0,1]"""
    return (x + 1.0) / 2.0


class ImageDirDataset(Dataset):
    def __init__(self, dir_path: str, transform):
        self.paths = list_images(dir_path)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        with Image.open(p) as im:
            im = im.convert("RGB")
            im.load()
        x = self.transform(im)
        return x, os.path.basename(p)


def seed_all(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------
# Inception feature extractor (for FID/KID/PRDC)
# --------------------------
@dataclass
class InceptionExtractor:
    model: nn.Module
    mean: torch.Tensor
    std: torch.Tensor
    device: torch.device

    @torch.inference_mode()
    def extract(self, x_01: torch.Tensor) -> torch.Tensor:
        """
        x_01: float tensor in [0,1], shape [N,3,H,W]
        returns: features [N,2048]
        """
        # Inception expects 299x299
        if x_01.shape[-1] != 299 or x_01.shape[-2] != 299:
            x_01 = F.interpolate(x_01, size=(299, 299), mode="bilinear", align_corners=False)

        x = (x_01 - self.mean) / self.std
        out = self.model(x)["feat"]  # [N,2048,1,1]
        out = out.flatten(1)
        return out


# def build_inception(device: torch.device) -> InceptionExtractor:
#     weights = Inception_V3_Weights.DEFAULT
#     inc = inception_v3(weights=weights, aux_logits=False, transform_input=False)
#     inc.eval().to(device)

#     # feature node: avgpool output
#     feat_model = create_feature_extractor(inc, return_nodes={"avgpool": "feat"}).eval()

#     mean = torch.tensor(weights.meta["mean"], dtype=torch.float32, device=device).view(1, 3, 1, 1)
#     std = torch.tensor(weights.meta["std"], dtype=torch.float32, device=device).view(1, 3, 1, 1)

#     return InceptionExtractor(model=feat_model, mean=mean, std=std, device=device)

def build_inception(device: torch.device) -> InceptionExtractor:
    """
    Backward-compatible InceptionV3 feature extractor:
    - Works on older torchvision (no Inception_V3_Weights / no create_feature_extractor)
    - Uses forward hook on avgpool to fetch 2048-d features
    """
    # ImageNet mean/std (torchvision pretrained convention)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)

    # Older torchvision: use pretrained=True
    inc = inception_v3(pretrained=True, transform_input=False, aux_logits=False)
    inc.eval().to(device)

    class _InceptionAvgPoolWrapper(nn.Module):
        def __init__(self, net: nn.Module):
            super().__init__()
            self.net = net
            self._feat = None

            def _hook(module, inp, out):
                self._feat = out

            # avgpool output: [N, 2048, 1, 1]
            self.net.avgpool.register_forward_hook(_hook)

        def forward(self, x):
            _ = self.net(x)
            return {"feat": self._feat}

    feat_model = _InceptionAvgPoolWrapper(inc).eval().to(device)
    return InceptionExtractor(model=feat_model, mean=mean, std=std, device=device)



# --------------------------
# FID (no scipy) via eigen decomposition
# --------------------------
def _compute_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def _sqrtm_psd(mat: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Matrix square root for symmetric PSD matrix using eigen decomposition.
    """
    # Make symmetric
    mat = (mat + mat.T) * 0.5
    w, v = np.linalg.eigh(mat)
    w = np.clip(w, eps, None)
    return (v * np.sqrt(w)) @ v.T


def fid_from_feats(real: np.ndarray, fake: np.ndarray) -> float:
    mu1, sigma1 = _compute_stats(real)
    mu2, sigma2 = _compute_stats(fake)

    diff = mu1 - mu2
    diff_sq = diff.dot(diff)

    # sqrt of sigma1*sigma2 via symmetric trick:
    # sqrtm(sigma1*sigma2) is not necessarily symmetric; we approximate using sqrtm( sqrt(s1) * s2 * sqrt(s1) )
    sqrt_sigma1 = _sqrtm_psd(sigma1)
    middle = sqrt_sigma1 @ sigma2 @ sqrt_sigma1
    sqrt_middle = _sqrtm_psd(middle)

    trace = np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(sqrt_middle)
    fid = float(diff_sq + trace)
    # numerical guard
    if fid < 0:
        fid = 0.0
    return fid


# --------------------------
# KID (polynomial MMD) on features
# --------------------------
def _poly_kernel(x: np.ndarray, y: np.ndarray, degree: int = 3, gamma: Optional[float] = None, coef0: float = 1.0):
    # x: [N,D], y: [M,D]
    d = x.shape[1]
    if gamma is None:
        gamma = 1.0 / d
    return (gamma * (x @ y.T) + coef0) ** degree


def kid_from_feats(real: np.ndarray, fake: np.ndarray, subset_size: int = 100, n_subsets: int = 100, seed: int = 0) -> float:
    """
    KID = average unbiased MMD^2 over random subsets.
    More stable than FID for small sample sizes.
    """
    rng = np.random.RandomState(seed)
    n_r = real.shape[0]
    n_f = fake.shape[0]
    m = min(subset_size, n_r, n_f)
    if m < 2:
        return float("nan")

    vals = []
    for _ in range(n_subsets):
        idx_r = rng.choice(n_r, size=m, replace=False)
        idx_f = rng.choice(n_f, size=m, replace=False)
        x = real[idx_r]
        y = fake[idx_f]

        k_xx = _poly_kernel(x, x)
        k_yy = _poly_kernel(y, y)
        k_xy = _poly_kernel(x, y)

        # unbiased MMD^2
        np.fill_diagonal(k_xx, 0.0)
        np.fill_diagonal(k_yy, 0.0)
        mmd = (k_xx.sum() / (m * (m - 1))) + (k_yy.sum() / (m * (m - 1))) - (2.0 * k_xy.mean())
        vals.append(mmd)

    return float(np.mean(vals))


# --------------------------
# PRDC (Precision/Recall/Density/Coverage)
# Ref: "Improved Precision and Recall Metric for Assessing Generative Models"
# Implementation adapted for small N (O(N^2) ok)
# --------------------------
def _pairwise_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # ||x-y||^2 then sqrt
    x2 = np.sum(x * x, axis=1, keepdims=True)
    y2 = np.sum(y * y, axis=1, keepdims=True).T
    d2 = np.maximum(x2 + y2 - 2.0 * (x @ y.T), 0.0)
    return np.sqrt(d2 + 1e-12)


def prdc_from_feats(real: np.ndarray, fake: np.ndarray, k: int = 5) -> Dict[str, float]:
    """
    Returns dict: precision, recall, density, coverage
    """
    real_real = _pairwise_distances(real, real)
    fake_fake = _pairwise_distances(fake, fake)
    real_fake = _pairwise_distances(real, fake)  # [Nr, Nf]
    fake_real = real_fake.T                      # [Nf, Nr]

    # radii = k-th nearest neighbor distance (exclude self)
    np.fill_diagonal(real_real, np.inf)
    np.fill_diagonal(fake_fake, np.inf)

    k = max(1, min(k, real.shape[0] - 1, fake.shape[0] - 1))

    real_radii = np.partition(real_real, k - 1, axis=1)[:, k - 1]  # [Nr]
    fake_radii = np.partition(fake_fake, k - 1, axis=1)[:, k - 1]  # [Nf]

    # precision: fake sample is within some real sample's radius
    precision = np.mean((fake_real <= real_radii[None, :]).any(axis=1))

    # recall: real sample is within some fake sample's radius
    recall = np.mean((real_fake <= fake_radii[None, :]).any(axis=1))

    # density: average number of real neighbors within real radii, normalized by k
    density = np.mean((fake_real <= real_radii[None, :]).sum(axis=1) / float(k))

    # coverage: fraction of real samples within radius of nearest fake
    nearest_fake_dist = real_fake.min(axis=1)
    coverage = np.mean(nearest_fake_dist <= real_radii)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "density": float(density),
        "coverage": float(coverage),
        "k": int(k),
    }


# --------------------------
# Background structure preservation + leakage
# (no external segmentation: use "largest changed connected component" as main subject)
# --------------------------
def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """
    mask: HxW bool
    return: HxW bool (largest 4-neighborhood component)
    """
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=np.uint8)

    best_coords = None
    best_size = 0

    # 4-neighborhood
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for y in range(h):
        row = mask[y]
        for x in range(w):
            if row[x] and not visited[y, x]:
                # BFS
                q = [(y, x)]
                visited[y, x] = 1
                coords = [(y, x)]
                while q:
                    cy, cx = q.pop()
                    for dy, dx in nbrs:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = 1
                            q.append((ny, nx))
                            coords.append((ny, nx))

                if len(coords) > best_size:
                    best_size = len(coords)
                    best_coords = coords

    out = np.zeros((h, w), dtype=bool)
    if best_coords is None:
        return out
    for (yy, xx) in best_coords:
        out[yy, xx] = True
    return out

def _tensor01_to_uint8_rgb(x01: torch.Tensor) -> np.ndarray:
    """
    x01: [3,H,W] in [0,1] -> uint8 HxWx3
    """
    x = (x01.clamp(0, 1) * 255.0).to(torch.uint8)
    x = x.permute(1, 2, 0).contiguous().cpu().numpy()
    return x

def _np_mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    """
    mask: HxW bool -> uint8 HxW (0 or 255)
    """
    return (mask.astype(np.uint8) * 255)

def _save_png(path: str, arr: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if arr.ndim == 2:
        Image.fromarray(arr, mode="L").save(path)
    else:
        Image.fromarray(arr, mode="RGB").save(path)

def _save_debug_artifacts(
    out_dir: str,
    fname: str,
    real01: torch.Tensor,   # [3,H,W]
    fake01: torch.Tensor,   # [3,H,W]
    diff_map: np.ndarray,   # HxW float
    tau_to_masks: Dict[float, Dict[str, np.ndarray]],  # {"change","main","bg"}
):
    """
    Save:
      real.png, fake.png, diff.png,
      change_tauX.png, main_tauX.png, bg_tauX.png
    """
    base = os.path.splitext(fname)[0]
    sample_dir = os.path.join(out_dir, base)
    os.makedirs(sample_dir, exist_ok=True)

    _save_png(os.path.join(sample_dir, "real.png"), _tensor01_to_uint8_rgb(real01))
    _save_png(os.path.join(sample_dir, "fake.png"), _tensor01_to_uint8_rgb(fake01))

    # diff visualization (grayscale, normalize by max for display)
    d = diff_map.astype(np.float32)
    if d.max() > 1e-8:
        d_vis = (d / d.max() * 255.0).clip(0, 255).astype(np.uint8)
    else:
        d_vis = np.zeros_like(d, dtype=np.uint8)
    _save_png(os.path.join(sample_dir, "diff.png"), d_vis)

    for tau, m in tau_to_masks.items():
        tag = f"{tau:.2f}".replace(".", "_")
        _save_png(os.path.join(sample_dir, f"change_tau{tag}.png"), _np_mask_to_uint8(m["change"]))
        _save_png(os.path.join(sample_dir, f"main_tau{tag}.png"), _np_mask_to_uint8(m["main"]))
        _save_png(os.path.join(sample_dir, f"bg_tau{tag}.png"), _np_mask_to_uint8(m["bg"]))


def _sobel_edges(gray: torch.Tensor) -> torch.Tensor:
    """
    gray: [N,1,H,W] in [0,1]
    return: edge magnitude [N,1,H,W]
    """
    # Sobel kernels
    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)

    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
    return mag


def _laplacian_energy(gray: torch.Tensor) -> torch.Tensor:
    """
    gray: [N,1,H,W]
    return: abs(laplacian) [N,1,H,W]
    """
    k = torch.tensor([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    y = F.conv2d(gray, k, padding=1)
    return y.abs()


def _rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    """
    x: [N,3,H,W] in [0,1]
    return: [N,1,H,W]
    """
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


@torch.inference_mode()
def background_edge_f1_and_leakage_multi(
    real_01: torch.Tensor,
    fake_01: torch.Tensor,
    diff_taus: List[float],
) -> Dict[float, Dict[str, float]]:
    """
    real_01, fake_01: [N,3,H,W] in [0,1]
    returns: {tau: {"bg_edge_f1": float(mean), "leakage": float(mean)}}
    """

    n, _, h, w = real_01.shape

    gray_real = _rgb_to_gray(real_01)
    gray_fake = _rgb_to_gray(fake_01)

    # change magnitude map in [0,1] approx
    diff = (fake_01 - real_01).abs().mean(dim=1, keepdim=True)  # [N,1,H,W]

    # edges (independent of tau)
    edge_real = _sobel_edges(gray_real)  # [N,1,H,W]
    edge_fake = _sobel_edges(gray_fake)

    # adaptive threshold per image (90% quantile)
    edge_real_flat = edge_real.view(n, -1)
    edge_fake_flat = edge_fake.view(n, -1)
    thr_r = torch.quantile(edge_real_flat, 0.90, dim=1).view(n, 1, 1, 1)
    thr_f = torch.quantile(edge_fake_flat, 0.90, dim=1).view(n, 1, 1, 1)

    edge_real_bin = (edge_real > thr_r).squeeze(1).detach().cpu().numpy()  # [N,H,W] bool
    edge_fake_bin = (edge_fake > thr_f).squeeze(1).detach().cpu().numpy()

    # leakage energy: laplacian energy of diff (high-frequency change)
    diff_gray = (gray_fake - gray_real).abs()  # [N,1,H,W]
    energy = _laplacian_energy(diff_gray).squeeze(1).detach().cpu().numpy()  # [N,H,W]

    diff_cpu = diff.squeeze(1).detach().cpu().numpy()  # [N,H,W] float

    out: Dict[float, Dict[str, float]] = {}

    for tau in diff_taus:
        bg_f1_list = []
        leak_list = []

        for i in range(n):
            # binarize change mask for this tau
            cm = (diff_cpu[i] > float(tau))  # HxW bool
            main = _largest_connected_component(cm)
            bg = ~main

            # Background edge F1
            er = edge_real_bin[i] & bg
            ef = edge_fake_bin[i] & bg
            tp = np.logical_and(er, ef).sum()
            fp = np.logical_and(~er, ef).sum()
            fn = np.logical_and(er, ~ef).sum()
            denom = (2 * tp + fp + fn)
            f1 = (2 * tp / denom) if denom > 0 else 1.0
            bg_f1_list.append(float(f1))

            # Leakage: energy outside main / total
            e = energy[i]
            total = float(e.sum()) + 1e-12
            out_e = float((e * bg).sum())
            leak_list.append(out_e / total)

        out[float(tau)] = {
            "bg_edge_f1": float(np.mean(bg_f1_list)),
            "leakage": float(np.mean(leak_list)),
        }

    return out


# --------------------------
# Main evaluation
# --------------------------
def build_gen_transform(size: int) -> T.Compose:
    return T.Compose([
        T.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # -> [-1,1]
    ])


def make_loader(dir_path: str, size: int, batch_size: int, num_workers: int, pin_memory: bool) -> DataLoader:
    ds = ImageDirDataset(dir_path, transform=build_gen_transform(size))
    dl_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    if num_workers > 0:
        dl_kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(ds, **dl_kwargs)


def load_model(checkpoint, device, amp=True, amp_dtype="bf16", channels_last=True,
               use_cbam=False, cbam_last_n=0, cbam_kwargs=None):
    """
    Use your CycleGANModel loader to stay compatible with checkpoint format.
    """
    from models.cycleGAN_model import CycleGANModel

    use_cuda = (device == "cuda" and torch.cuda.is_available())
    dev = "cuda" if use_cuda else "cpu"

    try:
        model = CycleGANModel(
            input_nc=3,
            output_nc=3,
            device=dev,
            use_amp=amp,
            amp_dtype=amp_dtype,
            channels_last=channels_last,
            use_fused_adam=False,
            use_compile=False,
            use_cbam=use_cbam,
            cbam_last_n=cbam_last_n,
            cbam_kwargs=(cbam_kwargs or {}),
        )
    except TypeError:
        model = CycleGANModel(input_nc=3, output_nc=3, device=dev)

    model.load(checkpoint, load_optim=False)
    model.netG_A.eval()
    model.netG_B.eval()
    return model, torch.device(dev)


@torch.inference_mode()
def extract_features_from_loader(
    loader: DataLoader,
    extractor: InceptionExtractor,
    device: torch.device,
) -> np.ndarray:
    feats = []
    for x, _ in tqdm(loader, desc="Extract real feats", leave=False):
        x = x.to(device, non_blocking=True)  # [-1,1]
        x01 = denorm(x).clamp(0, 1).float()
        f = extractor.extract(x01).detach().to("cpu").numpy()
        feats.append(f)
    return np.concatenate(feats, axis=0)


@torch.inference_mode()
def run_direction(
    name: str,
    loader_src: DataLoader,        # source domain (input to generator)
    loader_tgt_real: DataLoader,   # target domain real images (for FID/KID/PRDC)
    G_src2tgt: nn.Module,
    G_tgt2src: nn.Module,
    extractor: InceptionExtractor,
    device: torch.device,
    diff_taus: List[float],
    primary_tau: float,
    debug_dir: str,
    debug_samples: int,
) -> Dict[str, float]:

    # 1) real target features
    real_feats = extract_features_from_loader(loader_tgt_real, extractor, device)

    # 2) fake target features + cycle stats + bg/leak stats (multi tau)
    fake_feats = []
    cycle_l1_list = []

    # weighted sums for tau-dependent metrics
    bg_f1_sum = {float(t): 0.0 for t in diff_taus}
    leak_sum  = {float(t): 0.0 for t in diff_taus}
    total_n = 0

    # debug counter
    debug_left = int(max(0, debug_samples))

    for x_src, names in tqdm(loader_src, desc=f"Infer+Metrics {name}", leave=False):
        x_src = x_src.to(device, non_blocking=True)  # [-1,1]
        fake_tgt = G_src2tgt(x_src)
        rec_src = G_tgt2src(fake_tgt)

        bs = x_src.shape[0]
        total_n += bs

        # cycle L1 per image
        l1 = (rec_src - x_src).abs().mean(dim=(1, 2, 3))  # [N]
        cycle_l1_list.append(l1.detach().to("cpu").numpy())

        # background & leakage (multi tau)
        real01 = denorm(x_src).clamp(0, 1).float()
        fake01 = denorm(fake_tgt).clamp(0, 1).float()

        multi = background_edge_f1_and_leakage_multi(real01, fake01, diff_taus)
        for t in diff_taus:
            t = float(t)
            bg_f1_sum[t] += multi[t]["bg_edge_f1"] * bs
            leak_sum[t]  += multi[t]["leakage"] * bs

        # inception features for fake target
        f = extractor.extract(fake01).detach().to("cpu").numpy()
        fake_feats.append(f)

        # -------- DEBUG DUMP --------
        if debug_left > 0:
            # compute per-image diff map on CPU (mean abs RGB diff)
            diff_batch = (fake01 - real01).abs().mean(dim=1)  # [N,H,W]
            diff_batch_np = diff_batch.detach().cpu().numpy()

            for j in range(min(debug_left, bs)):
                # prepare masks for each tau
                tau_to_masks = {}
                for t in diff_taus:
                    cm = (diff_batch_np[j] > float(t))
                    main = _largest_connected_component(cm)
                    bg = ~main
                    tau_to_masks[float(t)] = {"change": cm, "main": main, "bg": bg}

                _save_debug_artifacts(
                    out_dir=os.path.join(debug_dir, name),
                    fname=names[j],
                    real01=real01[j].detach().cpu(),
                    fake01=fake01[j].detach().cpu(),
                    diff_map=diff_batch_np[j],
                    tau_to_masks=tau_to_masks,
                )
            debug_left -= min(debug_left, bs)

    fake_feats = np.concatenate(fake_feats, axis=0)
    cycle_l1 = float(np.mean(np.concatenate(cycle_l1_list, axis=0)))

    # tau-dependent averages
    bg_edge_f1_by_tau = {t: float(bg_f1_sum[t] / max(1, total_n)) for t in diff_taus}
    leakage_by_tau    = {t: float(leak_sum[t]  / max(1, total_n)) for t in diff_taus}

    # 3) distribution metrics
    fid = fid_from_feats(real_feats, fake_feats)
    kid = kid_from_feats(real_feats, fake_feats, subset_size=100, n_subsets=100, seed=0)
    prdc = prdc_from_feats(real_feats, fake_feats, k=5)

    # primary values (for single-number summary)
    # if primary_tau not in list, fallback to first
    pt = float(primary_tau)
    if pt not in leakage_by_tau:
        pt = float(diff_taus[0])

    return {
        "n_real": int(real_feats.shape[0]),
        "n_fake": int(fake_feats.shape[0]),
        "fid": float(fid),
        "kid": float(kid),
        "precision": prdc["precision"],
        "recall": prdc["recall"],
        "density": prdc["density"],
        "coverage": prdc["coverage"],
        "prdc_k": prdc["k"],
        "cycle_l1": float(cycle_l1),

        # tau-dependent
        "bg_edge_f1_by_tau": {f"{t:.3f}": bg_edge_f1_by_tau[t] for t in diff_taus},
        "leakage_energy_ratio_by_tau": {f"{t:.3f}": leakage_by_tau[t] for t in diff_taus},

        # primary single values
        "bg_edge_f1": float(bg_edge_f1_by_tau[pt]),
        "leakage_energy_ratio": float(leakage_by_tau[pt]),
        "primary_diff_tau": float(pt),
    }


@torch.inference_mode()
def identity_l1(loader_tgt: DataLoader, G_src2tgt: nn.Module, device: torch.device) -> float:
    """
    Identity: target images passed through src->tgt generator should remain similar.
    Returns mean L1 in [-1,1] space (consistent with training normalization).
    """
    vals = []
    for y, _ in tqdm(loader_tgt, desc="Identity L1", leave=False):
        y = y.to(device, non_blocking=True)
        y_id = G_src2tgt(y)
        l1 = (y_id - y).abs().mean(dim=(1, 2, 3))
        vals.append(l1.detach().to("cpu").numpy())
    return float(np.mean(np.concatenate(vals, axis=0)))

def add_bool_arg(parser, name, default, help_text=""):
    """
    兼容 Python3.8 的布尔开关：
      --name / --no-name
    """
    dest = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
    group.add_argument(f"--no-{name}", dest=dest, action="store_false", help=help_text)
    parser.set_defaults(**{dest: default})


def infer_cbam_from_checkpoint(ckpt_path):
    """
    从 checkpoint(netG_A 的 state_dict) 推断：
    - 是否启用 CBAM
    - cbam_last_n
    - cbam_ratio / cbam_kernel_size
    兼容保存格式：{"netG_A": state_dict, ...} 或纯 state_dict
    """
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "netG_A" in state:
        sd = state["netG_A"]
    elif isinstance(state, dict):
        sd = state
    else:
        return dict(use_cbam=False, cbam_last_n=0, cbam_ratio=16, cbam_kernel_size=7)

    cbam_param_keys = [k for k in sd.keys() if ".cbam." in k]
    if not cbam_param_keys:
        return dict(use_cbam=False, cbam_last_n=0, cbam_ratio=16, cbam_kernel_size=7)

    # 1) last_n：统计含 cbam 的 ResnetBlock index 数量（key: model.<idx>.cbam.xxx）
    idx_set = set()
    pat = re.compile(r"^model\.(\d+)\.cbam\.")
    for k in cbam_param_keys:
        m = pat.match(k)
        if m:
            idx_set.add(int(m.group(1)))
    cbam_last_n = len(idx_set) if idx_set else 0

    # 2) kernel_size：从 sa.conv.weight shape: (1,2,k,k)
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

    # 3) ratio：从 ca.mlp.0.weight shape: (hidden, in_planes, 1, 1)
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
    p = argparse.ArgumentParser("CycleGAN evaluation (FID/KID/PRDC/Cycle/Identity/Leakage)")

    p.add_argument("--data-root", type=str, default="/root/autodl-tmp/MyCycleGAN/data/horse2zebra",
                   help="root containing testA/testB")
    p.add_argument("--testA", type=str, default="",
                   help="override testA dir (horse). If empty, use data-root/testA")
    p.add_argument("--testB", type=str, default="",
                   help="override testB dir (zebra). If empty, use data-root/testB")

    p.add_argument("--checkpoint", type=str, required=True, help="path to .pth checkpoint")
    p.add_argument("--save-dir", type=str, default="./eval_out", help="where to save JSON report")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=0)

    # dataloader / size
    p.add_argument("--img-size", type=int, default=256, help="generator input size")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-pin-memory", action="store_true", help="disable pin_memory")

    # -------- tau sweep + debug dump --------
    p.add_argument("--diff-tau-list", type=str, default="0.15,0.20,0.25",
                   help="comma-separated thresholds for change-mask, e.g. 0.15,0.20,0.25")
    p.add_argument("--primary-diff-tau", type=float, default=0.15,
                   help="which tau to use for the single-value bg_edge_f1/leakage in console summary")

    # ✅ CBAM 兼容：默认 auto（从 checkpoint 推断）
    p.add_argument("--cbam-mode", type=str, default="auto", choices=["auto", "on", "off"],
                   help="auto: infer from ckpt; on/off: force enable/disable")
    p.add_argument("--cbam-last-n", type=int, default=-1,
                   help=">=0 强制指定最后 N 个残差块用 CBAM；-1 表示从 ckpt 推断")
    p.add_argument("--cbam-ratio", type=int, default=-1,
                   help=">0 强制指定 CBAM ratio；-1 表示从 ckpt 推断")
    p.add_argument("--cbam-kernel-size", type=int, default=-1,
                   help=">1 且为奇数时强制指定 kernel；-1 表示从 ckpt 推断")

    # backward compatible (deprecated)
    p.add_argument("--diff-tau", type=float, default=None,
                   help="(deprecated) alias of --primary-diff-tau; kept for backward compatibility")

    p.add_argument("--debug-samples", type=int, default=10,
                   help="dump debug masks for first N samples per direction (0 to disable)")

    # ✅ 关键修复：把 debug_dir 参数加回来
    p.add_argument("--debug-dir", type=str, default="",
                   help="where to dump debug images. default: <save-dir>/debug_masks")

    return p.parse_args()


def main():
    args = parse_args()

    # ✅ backward compatibility: --diff-tau overrides primary-diff-tau
    if getattr(args, "diff_tau", None) is not None:
        args.primary_diff_tau = float(args.diff_tau)

    # parse diff tau list
    diff_taus = [float(x.strip()) for x in args.diff_tau_list.split(",") if x.strip()]
    if len(diff_taus) == 0:
        diff_taus = [0.15, 0.20, 0.25]
    diff_taus = sorted(list(dict.fromkeys(diff_taus)))  # unique + sorted

    # ✅ debug_dir：只有 debug_samples > 0 才创建目录（更干净）
    debug_dir_arg = (getattr(args, "debug_dir", "") or "").strip()
    if int(args.debug_samples) > 0:
        debug_dir = debug_dir_arg if debug_dir_arg else os.path.join(args.save_dir, "debug_masks")
        os.makedirs(debug_dir, exist_ok=True)
    else:
        debug_dir = debug_dir_arg  # 可能为空；反正不会写文件

    seed_all(args.seed)

    testA = args.testA.strip() if args.testA.strip() else os.path.join(args.data_root, "testA")
    testB = args.testB.strip() if args.testB.strip() else os.path.join(args.data_root, "testB")

    os.makedirs(args.save_dir, exist_ok=True)

    use_cuda = (args.device == "cuda" and torch.cuda.is_available())
    device_str = "cuda" if use_cuda else "cpu"
    pin_memory = (use_cuda and (not args.no_pin_memory))

    # loaders: A(horse), B(zebra)
    loader_A = make_loader(testA, args.img_size, args.batch_size, args.num_workers, pin_memory)
    loader_B = make_loader(testB, args.img_size, args.batch_size, args.num_workers, pin_memory)

    # model
    # --------- ✅ 从 checkpoint 推断 / 或按用户强制设置 CBAM 配置 ---------
    inferred = infer_cbam_from_checkpoint(args.checkpoint)

    if args.cbam_mode == "off":
        use_cbam = False
        cbam_last_n = 0
        cbam_ratio = 16
        cbam_kernel = 7
    else:
        use_cbam = inferred["use_cbam"] if args.cbam_mode == "auto" else True

        cbam_last_n = inferred["cbam_last_n"] if args.cbam_last_n < 0 else int(args.cbam_last_n)
        cbam_ratio  = inferred["cbam_ratio"] if args.cbam_ratio < 0 else int(args.cbam_ratio)
        cbam_kernel = inferred["cbam_kernel_size"] if args.cbam_kernel_size < 0 else int(args.cbam_kernel_size)

        # 合法性兜底
        if cbam_last_n < 0:
            cbam_last_n = 0
        if cbam_ratio <= 0:
            cbam_ratio = 16
        if cbam_kernel <= 1 or (cbam_kernel % 2 == 0):
            cbam_kernel = 7

    cbam_kwargs = {"ratio": cbam_ratio, "kernel_size": cbam_kernel}

    print("[CBAM] mode=%s | inferred=%s" % (args.cbam_mode, inferred))
    print("[CBAM] using: use_cbam=%s, cbam_last_n=%d, ratio=%d, kernel=%d"
          % (use_cbam, cbam_last_n, cbam_ratio, cbam_kernel))

    # model（关键：结构与 ckpt 匹配）
    model, device = load_model(
        args.checkpoint, device_str,
        amp=True, amp_dtype="bf16", channels_last=True,
        use_cbam=use_cbam, cbam_last_n=cbam_last_n, cbam_kwargs=cbam_kwargs
    )
    G_A = model.netG_A.to(device)  # A->B
    G_B = model.netG_B.to(device)  # B->A


    # inception extractor
    extractor = build_inception(device)

    # --- Direction A->B ---
    metrics_AtoB = run_direction(
        name="AtoB",
        loader_src=loader_A,
        loader_tgt_real=loader_B,
        G_src2tgt=G_A,
        G_tgt2src=G_B,
        extractor=extractor,
        device=device,
        diff_taus=diff_taus,
        primary_tau=args.primary_diff_tau,
        debug_dir=debug_dir,
        debug_samples=args.debug_samples,
    )
    idt_A = identity_l1(loader_B, G_A, device)
    metrics_AtoB["identity_l1"] = float(idt_A)

    # --- Direction B->A ---
    metrics_BtoA = run_direction(
        name="BtoA",
        loader_src=loader_B,
        loader_tgt_real=loader_A,
        G_src2tgt=G_B,
        G_tgt2src=G_A,
        extractor=extractor,
        device=device,
        diff_taus=diff_taus,
        primary_tau=args.primary_diff_tau,
        debug_dir=debug_dir,
        debug_samples=args.debug_samples,
    )
    idt_B = identity_l1(loader_A, G_B, device)
    metrics_BtoA["identity_l1"] = float(idt_B)

    # average (simple mean)
    avg = {}
    for k in metrics_AtoB.keys():
        if isinstance(metrics_AtoB[k], (int, float)) and isinstance(metrics_BtoA.get(k, None), (int, float)):
            avg[k] = float((metrics_AtoB[k] + metrics_BtoA[k]) / 2.0)

    report = {
        "meta": {
            "checkpoint": args.checkpoint,
            "data_root": args.data_root,
            "testA": testA,
            "testB": testB,
            "count_testA": len(loader_A.dataset),
            "count_testB": len(loader_B.dataset),
            "device": device_str,
            "img_size": args.img_size,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "diff_tau_list": diff_taus,
            "primary_diff_tau": float(args.primary_diff_tau),
            "debug_samples": int(args.debug_samples),
            "debug_dir": debug_dir,
            "seed": int(args.seed),
        },
        "AtoB": metrics_AtoB,
        "BtoA": metrics_BtoA,
        "avg": avg,
    }

    out_json = os.path.join(args.save_dir, "eval_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # print summary
    def _fmt(d):
        keys = ["fid", "kid", "precision", "recall", "density", "coverage",
                "cycle_l1", "identity_l1", "bg_edge_f1", "leakage_energy_ratio"]
        s = []
        for kk in keys:
            if kk in d:
                s.append(f"{kk}={d[kk]:.6f}")
        return ", ".join(s)

    print("\n========== Evaluation Done ==========")
    print(f"[AtoB] {_fmt(metrics_AtoB)}")
    print(f"[BtoA] {_fmt(metrics_BtoA)}")
    print(f"[AVG ] {_fmt(avg)}")
    print(f"Saved: {out_json}")
    print("====================================\n")

if __name__ == "__main__":
    main()
