#!/usr/bin/env python3
"""Run SAM-3D-Objects on a frame + SAM3 mask to produce a posed 3D reconstruction.

Outputs written to --out-dir:
  splat.ply          – Gaussian splat (3D-GS format), raw decoder frame (unchanged)
  means_cam.npy      – Gaussian means in HD-EPIC RGB camera frame  (N, 3)
  pointmap_obj.npy   – masked dense point cloud in HD-EPIC RGB camera frame  (M, 3)
  twin.pkl           – metadata dict (includes sam3d_frame_version, pose tensors)

means_cam / pointmap_obj are produced by:
  1) layout pose (rotation, translation, scale) via SceneVisualizer.object_pointcloud
  2) inverse of SAM3D's R3→PyTorch3D rotation so coordinates match device_calibration

Usage (conda env: sam3d-objects):
    conda run -n sam3d-objects python run_sam3d.py \\
        --image   frame.jpg \\
        --mask    mask.png \\
        --out-dir /path/to/work_dir
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
import torch
from pytorch3d.transforms import Transform3d

# SAM-3D-Objects: repo root must be on sys.path so `import sam3d_objects` works;
# notebook/ sub-directory must be on sys.path so `from inference import ...` works.
_SAM3D_ROOT = Path(__file__).resolve().parent.parent / "sam-3d-objects"
sys.path.insert(0, str(_SAM3D_ROOT))            # for sam3d_objects package
sys.path.insert(0, str(_SAM3D_ROOT / "notebook"))  # for inference.py

from inference import Inference, load_mask  # noqa: E402
from sam3d_objects.pipeline.inference_pipeline_pointmap import (  # noqa: E402
    camera_to_pytorch3d_camera,
)
from sam3d_objects.utils.visualization.scene_visualizer import SceneVisualizer  # noqa: E402

SAM3D_FRAME_VERSION = 2


def _p3d_to_r3_points(
    pts: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Map PyTorch3D camera coords → MoGe R3 / HD-EPIC RGB camera coords."""
    tform = Transform3d(dtype=dtype, device=device).rotate(
        camera_to_pytorch3d_camera(device=device).rotation
    )
    if pts.dim() == 2:
        pts = pts.unsqueeze(0)
        return tform.inverse().transform_points(pts).squeeze(0)
    return tform.inverse().transform_points(pts)


def _layout_gaussian_xyz_p3d(output: dict) -> torch.Tensor:
    """Apply SAM3D layout pose: local decoder space → PyTorch3D camera space."""
    g0 = output["gaussian"][0]
    dev = g0.get_xyz.device
    PC = SceneVisualizer.object_pointcloud(
        points_local=g0.get_xyz.unsqueeze(0),
        quat_l2c=output["rotation"].to(dev),
        trans_l2c=output["translation"].to(dev),
        scale_l2c=output["scale"].to(dev),
    )
    return PC.points_list()[0]


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

    for key, value in output.items():
        print(f"{key}: {value}")
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")

    # ── save Gaussian splat (raw decoder frame, same as upstream demo cell) ───
    ply_path = out_dir / "splat.ply"
    output["gs"].save_ply(str(ply_path))
    print(f"[sam3d] Saved splat    → {ply_path}")

    # ── layout → PyTorch3D cam → HD-EPIC RGB cam ─────────────────────────────
    xyz_p3d = _layout_gaussian_xyz_p3d(output)
    dev = xyz_p3d.device
    dtype = xyz_p3d.dtype
    xyz_cam = _p3d_to_r3_points(xyz_p3d, dev, dtype).detach().cpu().numpy()
    np.save(out_dir / "means_cam.npy", xyz_cam)
    print(
        f"[sam3d] Saved means    → {out_dir}/means_cam.npy  (N={len(xyz_cam)})  "
        f"[HD-EPIC RGB cam, frame v{SAM3D_FRAME_VERSION}]"
    )

    # ── masked pointmap in HD-EPIC RGB camera frame ──────────────────────────
    if "pointmap" in output:
        pm = output["pointmap"]
        pm = pm.cpu().numpy() if hasattr(pm, "cpu") else np.asarray(pm)  # (H, W, 3)

        h_pm, w_pm = pm.shape[:2]
        if mask.shape != (h_pm, w_pm):
            mask_rs = np.array(
                PILImage.fromarray(mask.astype(np.uint8) * 255).resize(
                    (w_pm, h_pm), PILImage.NEAREST
                )
            ) > 0
        else:
            mask_rs = mask

        pm_t = torch.from_numpy(pm.astype(np.float32)).to(device=dev, dtype=dtype)
        pm_flat = pm_t.reshape(-1, 3)
        pm_r3 = _p3d_to_r3_points(pm_flat, dev, dtype).reshape(h_pm, w_pm, 3)
        pts_obj = pm_r3[mask_rs].detach().cpu().numpy()
        np.save(out_dir / "pointmap_obj.npy", pts_obj)
        print(
            f"[sam3d] Saved ptmap    → {out_dir}/pointmap_obj.npy  (M={len(pts_obj)})  "
            f"[HD-EPIC RGB cam, frame v{SAM3D_FRAME_VERSION}]"
        )
    else:
        print("[sam3d] No pointmap in output; skipping pointmap_obj.npy")
        pts_obj = None

    # ── metadata ─────────────────────────────────────────────────────────────
    rot = output["rotation"].detach().cpu().numpy()
    trans = output["translation"].detach().cpu().numpy()
    scale = output["scale"].detach().cpu().numpy()
    meta = {
        "ply_path":           str(ply_path),
        "num_gaussians":      int(len(xyz_cam)),
        "sam3d_frame_version": SAM3D_FRAME_VERSION,
        "rotation":           rot,
        "translation":        trans,
        "scale":              scale,
    }
    if pts_obj is not None:
        meta["num_pointmap_points"] = int(len(pts_obj))

    with open(out_dir / "twin.pkl", "wb") as fh:
        pickle.dump(meta, fh)
    print(f"[sam3d] Saved meta     → {out_dir}/twin.pkl")
    print("[sam3d] Done.")


if __name__ == "__main__":
    main()
