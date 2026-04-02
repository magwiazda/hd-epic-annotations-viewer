#!/usr/bin/env python3
"""Run SAM3 on a single frame image with a text prompt.

Outputs written to --out-dir:
  mask.png    – binary mask of the best-scoring instance  (0 / 255 PNG)
  box.npy     – 2-D bounding box [x1, y1, x2, y2] in pixels
  scores.npy  – confidence scores for all detected instances

Usage (conda env: sam3):
    conda run -n sam3 python run_sam3.py \\
        --image  frame.jpg \\
        --prompt "kettle" \\
        --out-dir /path/to/work_dir
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ── Patch the fused bfloat16 kernel before importing SAM3 ────────────────────
# sam3.perflib.fused.addmm_act unconditionally casts to bfloat16 for a fused
# CUDA op.  On pre-Ampere GPUs this causes a dtype mismatch when fc2's float32
# weight receives a bfloat16 activation.  Replace with a plain float32 path.
import sam3.perflib.fused as _fused


def _addmm_act_float32(activation, linear, mat1):
    """Float32 drop-in for addmm_act that skips the bfloat16 fused kernel."""
    x = F.linear(mat1, linear.weight, linear.bias)
    if activation in (torch.nn.functional.relu, torch.nn.ReLU):
        return F.relu(x)
    if activation in (torch.nn.functional.gelu, torch.nn.GELU):
        return F.gelu(x)
    raise ValueError(f"Unexpected activation {activation}")


_fused.addmm_act = _addmm_act_float32

# Now import SAM3 (its vitdet module already did `from sam3.perflib.fused import addmm_act`,
# so we also patch the reference in vitdet's module namespace)
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import sam3.model.vitdet as _vitdet
_vitdet.addmm_act = _addmm_act_float32
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image",    required=True, help="Input frame image (JPEG / PNG)")
    ap.add_argument("--prompt",   required=True, help="Text prompt describing the object")
    ap.add_argument("--out-dir",  required=True, help="Directory to write output files")
    ap.add_argument(
        "--instance", type=int, default=0,
        help="Which detected instance to keep: 0 = best score (default)",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[sam3] Loading image : {args.image}")
    image = Image.open(args.image).convert("RGB")

    print("[sam3] Building model ...")
    model     = build_sam3_image_model()
    processor = Sam3Processor(model)

    print(f"[sam3] Running with prompt: {args.prompt!r}")
    state  = processor.set_image(image)
    output = processor.set_text_prompt(state=state, prompt=args.prompt)

    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

    def to_np(t):
        return t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)

    scores_np = to_np(scores)
    print(f"[sam3] Detected {len(masks)} instance(s). Scores: {scores_np}")

    if len(masks) == 0:
        sys.exit(f"[sam3] ERROR: no objects found for prompt '{args.prompt}'")

    idx = int(scores_np.argmax()) if args.instance == 0 else min(args.instance, len(masks) - 1)

    # ── mask ──────────────────────────────────────────────────────────────────
    mask = to_np(masks[idx]).astype(bool)
    if mask.ndim == 3:          # (1, H, W) → (H, W)
        mask = mask[0]
    mask_path = out_dir / "mask.png"
    Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
    print(f"[sam3] Saved mask   → {mask_path}  shape={mask.shape}")

    # ── bounding box ──────────────────────────────────────────────────────────
    box = to_np(boxes[idx]).flatten()       # [x1, y1, x2, y2]
    np.save(out_dir / "box.npy", box)
    np.save(out_dir / "scores.npy", scores_np)
    print(f"[sam3] Saved box    → {out_dir / 'box.npy'}  {box}")
    print(f"[sam3] Saved scores → {out_dir / 'scores.npy'}")
    print("[sam3] Done.")


if __name__ == "__main__":
    main()
