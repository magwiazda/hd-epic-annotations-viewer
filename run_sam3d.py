#!/usr/bin/env python3
"""Run SAM-3D-Objects on a frame + SAM3 mask to produce a posed 3D reconstruction.

Outputs written to --out-dir:
  splat.ply          – Gaussian splat (3D-GS format)
  means_cam.npy      – Gaussian means in camera frame  (N, 3)
  pointmap_obj.npy   – dense object point cloud in camera frame  (M, 3)
  twin.pkl           – lightweight metadata dict

Usage (conda env: sam3d-objects):
    conda run -n sam3d-objects python run_sam3d.py \\
        --image   frame.jpg \\
        --mask    mask.png \\
        --out-dir /path/to/work_dir
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

# SAM-3D-Objects: repo root must be on sys.path so `import sam3d_objects` works;
# notebook/ sub-directory must be on sys.path so `from inference import ...` works.
_SAM3D_ROOT = Path(__file__).resolve().parent.parent / "sam-3d-objects"
sys.path.insert(0, str(_SAM3D_ROOT))            # for sam3d_objects package
sys.path.insert(0, str(_SAM3D_ROOT / "notebook"))  # for inference.py

from inference import Inference, load_mask  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image",   required=True, help="Input frame image (RGB JPEG / PNG)")
    ap.add_argument("--mask",    required=True, help="Binary mask PNG (0/255) from run_sam3.py")
    ap.add_argument("--out-dir", required=True, help="Directory to write output files")
    ap.add_argument(
        "--config",
        default=str(_SAM3D_ROOT / "checkpoints/hf/pipeline.yaml"),
        help="Path to SAM-3D-Objects pipeline.yaml",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load inputs ──────────────────────────────────────────────────────────
    print(f"[sam3d] Loading image  : {args.image}")
    image = np.array(PILImage.open(args.image).convert("RGB"), dtype=np.uint8)

    print(f"[sam3d] Loading mask   : {args.mask}")
    mask = load_mask(args.mask)     # boolean (H, W)

    # ── run reconstruction ────────────────────────────────────────────────────
    print(f"[sam3d] Loading model  : {args.config}")
    inference = Inference(args.config, compile=False)

    print("[sam3d] Running SAM-3D-Objects ...")
    output = inference(image, mask, seed=args.seed)

    # ── save Gaussian splat ───────────────────────────────────────────────────
    ply_path = out_dir / "splat.ply"
    output["gs"].save_ply(str(ply_path))
    print(f"[sam3d] Saved splat    → {ply_path}")

    # ── save Gaussian means (camera frame) ───────────────────────────────────
    xyz_cam = output["gs"].get_xyz.detach().cpu().numpy()   # (N, 3)
    np.save(out_dir / "means_cam.npy", xyz_cam)
    print(f"[sam3d] Saved means    → {out_dir}/means_cam.npy  (N={len(xyz_cam)})")

    # ── save object point cloud from dense pointmap (camera frame) ───────────
    if "pointmap" in output:
        pm = output["pointmap"]
        pm = pm.cpu().numpy() if hasattr(pm, "cpu") else np.asarray(pm)  # (H, W, 3)

        # resize mask to pointmap resolution if they differ
        h_pm, w_pm = pm.shape[:2]
        if mask.shape != (h_pm, w_pm):
            mask_rs = np.array(
                PILImage.fromarray(mask.astype(np.uint8) * 255).resize(
                    (w_pm, h_pm), PILImage.NEAREST
                )
            ) > 0
        else:
            mask_rs = mask

        pts_obj = pm[mask_rs]       # (M, 3)
        np.save(out_dir / "pointmap_obj.npy", pts_obj)
        print(f"[sam3d] Saved ptmap    → {out_dir}/pointmap_obj.npy  (M={len(pts_obj)})")
    else:
        print("[sam3d] No pointmap in output; skipping pointmap_obj.npy")

    # ── save metadata pickle ──────────────────────────────────────────────────
    meta = {
        "ply_path":      str(ply_path),
        "num_gaussians": int(len(xyz_cam)),
    }
    with open(out_dir / "twin.pkl", "wb") as fh:
        pickle.dump(meta, fh)
    print(f"[sam3d] Saved meta     → {out_dir}/twin.pkl")
    print("[sam3d] Done.")


if __name__ == "__main__":
    main()
