"""
demo_backend/server_fastapi.py

FastAPI inference service for CycleGAN generators.

Endpoints:
  GET  /health
  GET  /models
  POST /predict  (multipart/form-data)

Run:
  pip install -r demo_backend/requirements_demo.txt
  python demo_backend/server_fastapi.py --checkpoint-dir demo_backend/checkpoints --device cuda
"""

import argparse
import io
import os
from typing import Dict, Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image

from demo_backend.predictor import Predictor, PredictorConfig


app = FastAPI(title="CycleGAN Demo Inference API", version="1.0")

_PREDICTOR_CACHE: Dict[str, Predictor] = {}
_CFG = None  # filled by main()


def _list_models(checkpoint_dir: str):
    if not os.path.isdir(checkpoint_dir):
        return []
    names = []
    for fn in os.listdir(checkpoint_dir):
        if fn.endswith(".pth") or fn.endswith(".pt"):
            names.append(fn)
    names.sort()
    return names


def _get_predictor(model_name: str) -> Predictor:
    assert _CFG is not None
    ckpt_path = os.path.join(_CFG.checkpoint_dir, model_name)
    if model_name not in _PREDICTOR_CACHE:
        _PREDICTOR_CACHE[model_name] = Predictor(PredictorConfig(
            checkpoint=ckpt_path,
            device=_CFG.device,
            amp=_CFG.amp,
            amp_dtype=_CFG.amp_dtype,
            channels_last=_CFG.channels_last,
            default_size=_CFG.default_size,
        ))
    return _PREDICTOR_CACHE[model_name]


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/models")
def models():
    assert _CFG is not None
    return {"ok": True, "models": _list_models(_CFG.checkpoint_dir), "default": _CFG.default_model}


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    direction: str = Form("AtoB"),
    size: int = Form(256),
    strength: float = Form(1.0),
    model: Optional[str] = Form(None),
):
    assert _CFG is not None

    if direction not in ("AtoB", "BtoA"):
        return JSONResponse({"ok": False, "error": "INVALID_PARAM", "detail": "direction must be AtoB/BtoA"}, status_code=400)
    if size <= 0:
        return JSONResponse({"ok": False, "error": "INVALID_PARAM", "detail": "size must be > 0"}, status_code=400)
    if not (0.0 <= float(strength) <= 1.0):
        return JSONResponse({"ok": False, "error": "INVALID_PARAM", "detail": "strength must be in [0,1]"}, status_code=400)

    model_name = model or _CFG.default_model
    if not model_name:
        return JSONResponse({"ok": False, "error": "MODEL_NOT_FOUND", "detail": "no default model configured"}, status_code=400)

    ckpt_path = os.path.join(_CFG.checkpoint_dir, model_name)
    if not os.path.isfile(ckpt_path):
        return JSONResponse({"ok": False, "error": "MODEL_NOT_FOUND", "detail": f"{model_name} not found"}, status_code=404)

    # Read and decode image
    try:
        raw = await image.read()
        pil = Image.open(io.BytesIO(raw))
    except Exception as e:
        return JSONResponse({"ok": False, "error": "INVALID_IMAGE", "detail": str(e)}, status_code=400)

    try:
        pred = _get_predictor(model_name)
        out = pred.predict(pil, direction=direction, size=size, strength=strength)
    except Exception as e:
        return JSONResponse({"ok": False, "error": "INFERENCE_FAILED", "detail": str(e)}, status_code=500)

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", type=str, default="demo_backend/checkpoints")
    p.add_argument("--default-model", type=str, default="")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.set_defaults(amp=True)
    p.add_argument("--amp-dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--no-channels-last", dest="channels_last", action="store_false")
    p.set_defaults(channels_last=True)
    p.add_argument("--default-size", type=int, default=256)

    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args()


def main():
    global _CFG
    args = parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    models = _list_models(args.checkpoint_dir)
    if args.default_model:
        default_model = args.default_model
    else:
        default_model = models[0] if models else ""

    class _Obj:  # simple namespace
        pass

    cfg = _Obj()
    cfg.checkpoint_dir = args.checkpoint_dir
    cfg.default_model = default_model
    cfg.device = args.device
    cfg.amp = args.amp
    cfg.amp_dtype = args.amp_dtype
    cfg.channels_last = args.channels_last
    cfg.default_size = args.default_size
    _CFG = cfg

    print(f"[INFO] checkpoint_dir: {cfg.checkpoint_dir}")
    print(f"[INFO] models: {models}")
    print(f"[INFO] default_model: {cfg.default_model}")
    print(f"[INFO] device: {cfg.device}, amp: {cfg.amp} ({cfg.amp_dtype}), channels_last: {cfg.channels_last}")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
