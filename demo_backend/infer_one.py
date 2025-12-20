"""
demo_backend/infer_one.py

CLI tool for single-image inference.
Example:
  python demo_backend/infer_one.py \
    --checkpoint demo_backend/checkpoints/epoch_200_cbam.pth \
    --input assets/example_inputs/1.jpg \
    --output outputs/out.png \
    --direction AtoB \
    --size 256 \
    --strength 1.0
"""

import argparse
import os

from PIL import Image

from demo_backend.predictor import Predictor, PredictorConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.set_defaults(amp=True)
    p.add_argument("--amp-dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument("--no-channels-last", dest="channels_last", action="store_false")
    p.set_defaults(channels_last=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    pred = Predictor(PredictorConfig(
        checkpoint=args.checkpoint,
        device=args.device,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        channels_last=args.channels_last,
        default_size=args.size,
    ))

    img = Image.open(args.input)
    out = pred.predict(img, direction=args.direction, size=args.size, strength=args.strength)
    out.save(args.output)
    print(f"[OK] saved: {args.output}")


if __name__ == "__main__":
    main()
