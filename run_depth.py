#!/usr/bin/env python3
"""Run Depth-Anything-V2 on a single frame and save the raw depth map.

Outputs written to --out-dir:
  depth_raw.npy   – raw float32 disparity map (H×W); larger = closer
  depth_vis.png   – Spectral_r colorised visualisation

Usage (conda env: sam3):
    conda run -n sam3 python run_depth.py \\
        --image  frame.jpg \\
        --out-dir /path/to/work_dir
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

_DEPTH_ANYTHING_DIR = Path("/usr/prakt/s0021/EH/Depth-Anything-V2")
sys.path.insert(0, str(_DEPTH_ANYTHING_DIR))

from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402


_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image",      required=True, help="Input frame image (JPEG / PNG)")
    ap.add_argument("--out-dir",    required=True, help="Directory to write output files")
    ap.add_argument(
        "--encoder", default="vitl",
        choices=list(_MODEL_CONFIGS),
        help="Model encoder size (default: vitl)",
    )
    ap.add_argument("--input-size", type=int, default=518, help="Inference input resolution")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    DEVICE = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[depth] Device      : {DEVICE}")

    ckpt = _DEPTH_ANYTHING_DIR / "checkpoints" / f"depth_anything_v2_{args.encoder}.pth"
    print(f"[depth] Checkpoint  : {ckpt}")

    model = DepthAnythingV2(**_MODEL_CONFIGS[args.encoder])
    model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
    model = model.to(DEVICE).eval()

    print(f"[depth] Image       : {args.image}")
    raw_img = cv2.imread(args.image)
    if raw_img is None:
        sys.exit(f"[depth] ERROR: cannot read image {args.image}")

    # raw disparity – float32, shape (H, W), larger values = closer to camera
    depth: np.ndarray = model.infer_image(raw_img, args.input_size)

    # ── save raw depth ──────────────────────────────────────────────────────────
    npy_path = out_dir / "depth_raw.npy"
    np.save(str(npy_path), depth)
    print(f"[depth] Saved raw   → {npy_path}  "
          f"shape={depth.shape}  range=[{depth.min():.4f}, {depth.max():.4f}]")

    # ── save Spectral_r colourised visualisation ────────────────────────────────
    d_min, d_max = depth.min(), depth.max()
    depth_norm = ((depth - d_min) / (d_max - d_min + 1e-8) * 255).astype(np.uint8)
    cmap = matplotlib.colormaps.get_cmap("Spectral_r")
    depth_color = (cmap(depth_norm)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

    vis_path = out_dir / "depth_vis.png"
    cv2.imwrite(str(vis_path), depth_color)
    print(f"[depth] Saved vis   → {vis_path}")
    print("[depth] Done.")


if __name__ == "__main__":
    main()
