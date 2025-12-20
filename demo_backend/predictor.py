"""
demo_backend/predictor.py

A thin, stable inference wrapper for your CycleGAN generators.
- Loads netG_A / netG_B from a CycleGAN checkpoint (.pth)
- Infers whether CBAM was used (and its config) from checkpoint keys
- Accepts a PIL image, returns a PIL image

Design goal:
  Keep the web/demo side *independent* from training details.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import to_pil_image
from PIL import Image


# Make sure repo root is importable when running as:
#   python demo_backend/infer_one.py ...
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.generator import ResnetGenerator  # noqa


def _denorm(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1]"""
    return (x + 1.0) / 2.0


def _resolve_amp_dtype(amp_dtype: str):
    """
    auto: prefer bf16 if supported, else fp16 (CUDA only)
    fp32: disable autocast
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
    return None  # fp32


def infer_cbam_from_checkpoint(ckpt_path: str) -> Dict:
    """
    Infer CBAM config from netG_A state_dict keys.
    Expected checkpoint format:
      {"netG_A": state_dict, "netG_B": state_dict, ...}

    Returns:
      {
        "use_cbam": bool,
        "cbam_last_n": int,
        "cbam_ratio": int,
        "cbam_kernel_size": int,
      }
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

    # 1) Infer cbam_last_n by counting distinct ResnetBlock indices containing cbam.*
    # Keys look like: model.10.cbam.ca.mlp.0.weight
    idx_set = set()
    pat = re.compile(r"^model\.(\d+)\.cbam\.")
    for k in cbam_param_keys:
        m = pat.match(k)
        if m:
            idx_set.add(int(m.group(1)))
    cbam_last_n = len(idx_set) if idx_set else 0

    # 2) Infer kernel_size from spatial attention conv weight: (1, 2, k, k)
    cbam_kernel_size = 7
    sa_w_key = next((k for k in cbam_param_keys if k.endswith("cbam.sa.conv.weight")), None)
    if sa_w_key is not None:
        w = sd[sa_w_key]
        if hasattr(w, "shape") and len(w.shape) == 4:
            cbam_kernel_size = int(w.shape[-1])

    # 3) Infer ratio from channel attention first conv: (hidden, in_planes, 1, 1)
    cbam_ratio = 16
    ca_w_key = next((k for k in cbam_param_keys if k.endswith("cbam.ca.mlp.0.weight")), None)
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


@dataclass
class PredictorConfig:
    checkpoint: str
    device: str = "cuda"  # "cuda" or "cpu"
    amp: bool = True
    amp_dtype: str = "auto"  # auto|bf16|fp16|fp32
    channels_last: bool = True
    default_size: int = 256


class Predictor:
    """
    A stable inference wrapper:
      - Loads both generators (G_A, G_B)
      - Exposes predict(image, direction, size, strength)

    Notes:
      - Inputs are normalized to [-1, 1] with mean/std = 0.5
      - Outputs are denormed to [0, 1] and converted to PIL (RGB)
    """

    def __init__(self, cfg: PredictorConfig):
        self.cfg = cfg

        use_cuda = (cfg.device == "cuda" and torch.cuda.is_available())
        self.device = torch.device("cuda" if use_cuda else "cpu")

        # AMP
        self.amp = bool(cfg.amp and self.device.type == "cuda")
        self.amp_dtype = _resolve_amp_dtype(cfg.amp_dtype) if self.amp else None

        # Infer CBAM config from checkpoint (auto)
        inferred = infer_cbam_from_checkpoint(cfg.checkpoint)
        self.use_cbam = inferred["use_cbam"]
        self.cbam_last_n = inferred["cbam_last_n"]
        self.cbam_kwargs = dict(ratio=inferred["cbam_ratio"], kernel_size=inferred["cbam_kernel_size"])

        # Build generators
        self.netG_A = ResnetGenerator(
            3, 3, 64,
            use_cbam=self.use_cbam,
            cbam_last_n=self.cbam_last_n,
            cbam_kwargs=self.cbam_kwargs if self.use_cbam else None,
        )
        self.netG_B = ResnetGenerator(
            3, 3, 64,
            use_cbam=self.use_cbam,
            cbam_last_n=self.cbam_last_n,
            cbam_kwargs=self.cbam_kwargs if self.use_cbam else None,
        )

        # Load checkpoint
        state = torch.load(cfg.checkpoint, map_location="cpu")
        if isinstance(state, dict) and ("netG_A" in state and "netG_B" in state):
            self.netG_A.load_state_dict(state["netG_A"], strict=True)
            self.netG_B.load_state_dict(state["netG_B"], strict=True)
        else:
            raise ValueError(
                f"Unsupported checkpoint format: expected keys netG_A/netG_B in {cfg.checkpoint}"
            )

        # Move to device
        self.netG_A.to(self.device)
        self.netG_B.to(self.device)

        # channels_last (optional, CUDA only)
        self.channels_last = bool(cfg.channels_last and self.device.type == "cuda")
        if self.channels_last:
            self.netG_A.to(memory_format=torch.channels_last)
            self.netG_B.to(memory_format=torch.channels_last)

        self.netG_A.eval()
        self.netG_B.eval()

        # Preprocess transform (size is set per-call; we build dynamically)
        self._base_norm = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def _make_transform(self, size: int):
        return T.Compose([
            T.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            self._base_norm,
        ])

    @torch.inference_mode()
    def predict(
        self,
        img: Image.Image,
        direction: str = "AtoB",
        size: Optional[int] = None,
        strength: float = 1.0,
    ) -> Image.Image:
        """
        Args:
          img: PIL Image (any mode). Will be converted to RGB.
          direction: "AtoB" uses G_A; "BtoA" uses G_B
          size: resize to (size, size). If None uses cfg.default_size.
          strength: 0~1. out = strength*fake + (1-strength)*input (in [0,1] space)

        Returns:
          PIL.Image (RGB)
        """
        if direction not in ("AtoB", "BtoA"):
            raise ValueError("direction must be 'AtoB' or 'BtoA'")
        size = int(size or self.cfg.default_size)
        if size <= 0:
            raise ValueError("size must be > 0")
        # downsample twice -> better use multiples of 4
        if size % 4 != 0:
            # keep it permissive but warn via exception message if needed
            pass

        strength = float(strength)
        if not (0.0 <= strength <= 1.0):
            raise ValueError("strength must be in [0,1]")

        img_rgb = img.convert("RGB")
        x_in = self._make_transform(size)(img_rgb).unsqueeze(0)  # [1,3,H,W], in [-1,1]
        x_in = x_in.to(self.device, non_blocking=True)
        if self.channels_last:
            x_in = x_in.contiguous(memory_format=torch.channels_last)

        net = self.netG_A if direction == "AtoB" else self.netG_B

        if self.amp and self.amp_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=True):
                fake = net(x_in)
        else:
            fake = net(x_in)

        # Convert both input and output to [0,1]
        fake01 = _denorm(fake).clamp(0, 1)
        in01 = _denorm(x_in).clamp(0, 1)

        out01 = fake01 if strength >= 1.0 else (strength * fake01 + (1.0 - strength) * in01)
        out = out01.squeeze(0).cpu()
        return to_pil_image(out)
