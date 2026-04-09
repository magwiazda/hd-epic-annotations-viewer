import gzip
import io
import json
import pickle
import re
import subprocess
import zipfile
from pathlib import Path

import matplotlib as _mpl
import matplotlib.cm as mplcm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.ndimage
import trimesh
from IPython.display import display
from PIL import Image as PILImage

# ── Paths ──────────────────────────────────────────────────────────────────────

BLENDERS_ZIP   = Path("/usr/prakt/s0021/EH/HD-EPIC/Digital-Twin/blenders.zip")
FRAMEWISE_BASE = Path("/usr/prakt/s0021/EH/HD-EPIC/Intermediate-Data")
GAZE_HAND_BASE = Path("/usr/prakt/s0021/EH/HD-EPIC/SLAM-and-Gaze")
SLAM_BASE      = GAZE_HAND_BASE

_VIEWER_ROOT = Path("/usr/prakt/s0021/EH/hd-epic-annotations-viewer")
_VIDEOS_ROOT = Path("/usr/prakt/s0021/EH/HD-EPIC/Videos")
_SLURM_LOGS  = Path("/usr/prakt/s0021/slurm/logs")
_VIDEO_FPS   = 30

# ── Depth alignment options ────────────────────────────────────────────────────
# When True, the object mask is dilated before collecting SLAM correspondences
# so that nearby surface points are included in the linear fit.
# Set to False to use only the exact mask pixels (may give fewer correspondences).
ALIGN_USE_MASK_DILATION: bool = True

# ── Color palette ──────────────────────────────────────────────────────────────

_TYPE_COLORS = {
    "floor":          "#C8B8A2",
    "counter":        "#8C7B6E",
    "cupboard":       "#A89880",
    "top_cupboard":   "#7A6A5C",
    "shelf":          "#B0A090",
    "drawer":         "#9A8A7A",
    "sink":           "#9CB4BE",
    "fridge":         "#C8D8DC",
    "fridgefreezer":  "#C0D4D8",
    "freezer":        "#B8CCD0",
    "oven":           "#3C3C3C",
    "hob":            "#2C2C2C",
    "microwave":      "#505050",
    "top_microwave":  "#4A4A4A",
    "top_fridge":     "#B0C8CC",
    "dishwasher":     "#7890A0",
    "hook":           "#D4C4B0",
    "storage":        "#B4A494",
    "bin":            "#6A7060",
    "basket":         "#C0A878",
    "table":          "#A09080",
}
_DEFAULT_COLOR = "#B0B0B0"


# ── Helpers ────────────────────────────────────────────────────────────────────

def participant_from_video_id(video_id: str) -> str:
    """Extract participant ID (e.g. 'P01') from a video ID like 'P01-20240202-110250'."""
    match = re.match(r"(P\d+)", video_id)
    if not match:
        raise ValueError(
            f"Cannot parse participant ID from video_id='{video_id}'. "
            f"Expected format: P<nn>-YYYYMMDD-HHMMSS"
        )
    return match.group(1)


def _mesh_color(name: str) -> str:
    name_lower = name.lower()
    stripped = re.sub(r"^p\d+_", "", name_lower)
    stripped = re.sub(r"\.\d+$", "", stripped)
    for key, color in _TYPE_COLORS.items():
        if stripped.startswith(key):
            return color
    return _DEFAULT_COLOR


def _to_4x4(T3x4: np.ndarray) -> np.ndarray:
    M = np.eye(4)
    M[:3, :] = T3x4
    return M


# ── Scene meshes ───────────────────────────────────────────────────────────────

def load_scene_meshes(zip_path: Path, participant: str) -> list[dict]:
    """Load all OBJ meshes for `participant` from the blenders zip.

    Returns a list of dicts with keys: name, vertices, faces.
    """
    prefix = f"meshes/{participant}/"
    meshes = []

    with zipfile.ZipFile(zip_path) as zf:
        obj_names = [
            n for n in zf.namelist()
            if n.startswith(prefix) and n.endswith(".obj") and "__MACOSX" not in n
        ]

        if not obj_names:
            raise FileNotFoundError(
                f"No OBJ files found for participant '{participant}' in {zip_path}. "
                f"Available participants: " +
                str(sorted({Path(n).parts[1] for n in zf.namelist()
                             if n.startswith("meshes/") and len(Path(n).parts) > 2}))
            )

        for name in sorted(obj_names):
            raw = zf.read(name).decode("utf-8")
            mesh = trimesh.load(io.StringIO(raw), file_type="obj", force="mesh")
            if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
                continue
            meshes.append({
                "name":     Path(name).stem,
                "vertices": mesh.vertices,
                "faces":    mesh.faces,
            })

    print(f"Loaded {len(meshes)} meshes for {participant}.")
    return meshes


def build_figure(meshes: list[dict], participant: str, video_id: str) -> go.Figure:
    """Build a Plotly 3D figure from the loaded meshes."""
    traces = []

    for m in meshes:
        v = m["vertices"]
        f = m["faces"]
        color = _mesh_color(m["name"])

        traces.append(go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2],
            i=f[:, 0], j=f[:, 1], k=f[:, 2],
            color=color,
            opacity=0.85,
            name=m["name"],
            showlegend=True,
            hovertemplate=(
                f"<b>{m['name']}</b><br>"
                "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>"
            ),
            flatshading=True,
            lighting=dict(ambient=0.5, diffuse=0.8, specular=0.3, roughness=0.5),
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text=f"Kitchen 3D Scene — {participant}  |  video: {video_id}",
            font=dict(size=16),
        ),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
            bgcolor="#1a1a2e",
            xaxis=dict(showbackground=True, backgroundcolor="#0f0f1e",
                       gridcolor="#333355", color="#aaaacc"),
            yaxis=dict(showbackground=True, backgroundcolor="#0f0f1e",
                       gridcolor="#333355", color="#aaaacc"),
            zaxis=dict(showbackground=True, backgroundcolor="#0f0f1e",
                       gridcolor="#333355", color="#aaaacc"),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        ),
        paper_bgcolor="#12121f",
        font=dict(color="#ccccdd"),
        legend=dict(
            bgcolor="#1e1e30",
            bordercolor="#444466",
            borderwidth=1,
            font=dict(size=10),
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        height=700,
    )
    return fig


# ── Frame data ─────────────────────────────────────────────────────────────────

def load_frame_data(video_id: str, frame_index: int) -> dict | None:
    """Return the framewise_info.jsonl entry for *frame_index*, or None if not found."""
    participant = participant_from_video_id(video_id)
    jsonl_path  = FRAMEWISE_BASE / participant / video_id / "framewise_info.jsonl"
    with open(jsonl_path) as fh:
        for line in fh:
            entry = json.loads(line)
            if entry["frame_index"] == frame_index:
                return entry
    return None


def add_camera_pose_to_figure(
    fig: go.Figure,
    frame: dict,
    axis_len: float = 0.15,
) -> go.Figure:
    """Overlay the device camera pose (and optional gaze ray) on an existing 3D figure.

    Draws:
      • gold diamond        – camera centre in world coordinates
      • red / green / blue  – X / Y / Z axes of the device frame (with arrowheads)
      • pink dashed line    – gaze direction ray (2× axis_len long)
    """
    T   = np.array(frame["T_world_device"])
    pos = T[:, 3]
    R   = T[:, :3]

    fig.add_trace(go.Scatter3d(
        x=[pos[0]], y=[pos[1]], z=[pos[2]],
        mode="markers",
        marker=dict(size=8, color="#FFD700", symbol="diamond"),
        name=f"Camera – frame {frame['frame_index']}",
    ))

    for label, colour, direction in [
        ("X (right)",   "#FF4444", R[:, 0]),
        ("Y (down)",    "#44DD44", R[:, 1]),
        ("Z (forward)", "#4499FF", R[:, 2]),
    ]:
        tip = pos + axis_len * direction
        fig.add_trace(go.Scatter3d(
            x=[pos[0], tip[0]], y=[pos[1], tip[1]], z=[pos[2], tip[2]],
            mode="lines",
            line=dict(color=colour, width=5),
            name=label,
        ))
        fig.add_trace(go.Cone(
            x=[tip[0]], y=[tip[1]], z=[tip[2]],
            u=[direction[0]], v=[direction[1]], w=[direction[2]],
            sizeref=axis_len * 0.25,
            colorscale=[[0, colour], [1, colour]],
            showscale=False,
            showlegend=False,
        ))

    gaze = frame.get("gaze_direction_in_world")
    if gaze is not None:
        gaze     = np.array(gaze)
        gaze_end = pos + axis_len * 2.0 * gaze
        fig.add_trace(go.Scatter3d(
            x=[pos[0], gaze_end[0]], y=[pos[1], gaze_end[1]], z=[pos[2], gaze_end[2]],
            mode="lines",
            line=dict(color="#FF88FF", width=3, dash="dash"),
            name="Gaze direction",
        ))

    return fig


# ── Hand tracking ──────────────────────────────────────────────────────────────

def load_hand_poses_for_frame(video_id: str, frame: dict) -> dict:
    """Return wrist and palm positions in world coordinates for the given frame."""
    participant = participant_from_video_id(video_id)
    slug        = f"mps_{video_id}_vrs"
    zip_path    = GAZE_HAND_BASE / participant / "GAZE_HAND" / f"{slug}.zip"

    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(f"{slug}/hand_tracking/wrist_and_palm_poses.csv") as fh:
            df = pd.read_csv(fh)

    target_us = frame["tracking_timestamp_ns"] / 1_000
    row = df.iloc[(df["tracking_timestamp_us"] - target_us).abs().argmin()]

    T = np.array(frame["T_world_device"])
    R, t = T[:, :3], T[:, 3]

    def to_world(tx, ty, tz, conf):
        if conf <= 0:
            return None
        return R @ np.array([tx, ty, tz]) + t

    return {
        "left_wrist":       to_world(row.tx_left_wrist_device,  row.ty_left_wrist_device,
                                     row.tz_left_wrist_device,  row.left_tracking_confidence),
        "left_palm":        to_world(row.tx_left_palm_device,   row.ty_left_palm_device,
                                     row.tz_left_palm_device,   row.left_tracking_confidence),
        "right_wrist":      to_world(row.tx_right_wrist_device, row.ty_right_wrist_device,
                                     row.tz_right_wrist_device, row.right_tracking_confidence),
        "right_palm":       to_world(row.tx_right_palm_device,  row.ty_right_palm_device,
                                     row.tz_right_palm_device,  row.right_tracking_confidence),
        "left_confidence":  float(row.left_tracking_confidence),
        "right_confidence": float(row.right_tracking_confidence),
    }


def add_hands_to_figure(fig: go.Figure, hand_poses: dict) -> go.Figure:
    """Overlay wrist and palm markers on an existing 3D figure."""
    specs = [
        ("left_wrist",  "left_palm",  "#5599FF", "#88CCFF", "Left hand"),
        ("right_wrist", "right_palm", "#FF8833", "#FFBB66", "Right hand"),
    ]
    for wrist_key, palm_key, wrist_col, palm_col, label in specs:
        w = hand_poses.get(wrist_key)
        p = hand_poses.get(palm_key)
        if w is None and p is None:
            continue

        pts   = [(pt, col, sz) for pt, col, sz in [(w, wrist_col, 10), (p, palm_col, 7)]
                 if pt is not None]
        xs    = [pt[0] for pt, _, _  in pts]
        ys    = [pt[1] for pt, _, _  in pts]
        zs    = [pt[2] for pt, _, _  in pts]
        sizes = [sz    for _,  _, sz in pts]
        cols  = [col   for _,  col, _ in pts]

        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers",
            marker=dict(size=sizes, color=cols),
            name=label,
        ))
        if w is not None and p is not None:
            fig.add_trace(go.Scatter3d(
                x=[w[0], p[0]], y=[w[1], p[1]], z=[w[2], p[2]],
                mode="lines",
                line=dict(color=wrist_col, width=4),
                showlegend=False,
            ))

    return fig


# ── Point cloud ────────────────────────────────────────────────────────────────

def slam_index_for_video(participant: str, video_id: str) -> str:
    """Return the multi-SLAM segment string index for a given video ID."""
    mapping_path = SLAM_BASE / participant / "SLAM" / "multi" / "vrs_to_multi_slam.json"
    with open(mapping_path) as fh:
        mapping = json.load(fh)
    key = f"{participant}/{video_id}.vrs"
    return mapping[key]


def load_pointcloud(
    participant: str,
    slam_index: str,
    max_points: int = 150_000,
    max_dist_std: float = 0.5,
) -> np.ndarray:
    """Load and downsample the semidense point cloud for a SLAM segment.

    Returns an (N, 3) float array of world-frame (x, y, z) coordinates.
    """
    zip_path = SLAM_BASE / participant / "SLAM" / "multi" / f"{slam_index}.zip"
    inner    = f"{slam_index}/slam/semidense_points.csv.gz"

    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(inner) as fh:
            with gzip.open(fh) as gz:
                df = pd.read_csv(gz)

    df = df[df["dist_std"] < max_dist_std]
    if len(df) > max_points:
        df = df.sample(max_points, random_state=42)
    return df[["px_world", "py_world", "pz_world"]].to_numpy()


def build_slam_vs_sam3d_figure(
    result: dict,
    frame: dict,
    video_id: str,
    pts_world: np.ndarray,
    meshes: list,
    participant: str,
    mask_dilation: int = 60,
    max_depth: float = 5.0,
) -> go.Figure:
    """3-D figure comparing SLAM points near the object vs SAM3D pointmap_obj.

    Red    – SLAM semidense points that project into the (dilated) object mask.
    Violet – pointmap_obj from SAM3D transformed to world frame.

    Kitchen meshes are shown as a semi-transparent background.
    """
    from scipy.ndimage import binary_dilation as _bin_dil

    cam  = load_rgb_calibration(video_id)
    W, H = cam["image_size"]
    T_wc = _to_4x4(np.array(frame["T_world_device"])) @ _to_4x4(cam["T_device_camera"])
    T_cw = np.linalg.inv(T_wc)

    # ── Filter SLAM points that fall inside the dilated mask ──────────────────
    mask = result.get("mask")
    if mask is None:
        raise ValueError("result must contain 'mask'.")

    # project SLAM world pts → camera frame → pixels
    N     = len(pts_world)
    pts_h = np.hstack([pts_world, np.ones((N, 1))])
    pts_c = (T_cw @ pts_h.T).T[:, :3]
    front = pts_c[:, 2] > 0.05
    u_s = np.full(N, np.nan)
    v_s = np.full(N, np.nan)
    u_s[front], v_s[front] = project_fisheye624(pts_c[front], cam["params"])

    vr    = cam["valid_radius"] or (min(W, H) / 2)
    cu_px, cv_px = cam["params"][1], cam["params"][2]
    ok = (
        np.isfinite(u_s) & np.isfinite(v_s) &
        (u_s >= 0) & (u_s < W) &
        (v_s >= 0) & (v_s < H) &
        (pts_c[:, 2] <= max_depth) &
        (np.hypot(u_s - cu_px, v_s - cv_px) < vr)
    )

    dilated = _bin_dil(mask, iterations=mask_dilation)
    ui = np.where(ok, np.clip(u_s.astype(int), 0, W - 1), 0)
    vi = np.where(ok, np.clip(v_s.astype(int), 0, H - 1), 0)
    in_mask = ok & dilated[vi, ui]

    slam_near = pts_world[in_mask]

    # ── Transform pointmap_obj to world frame ─────────────────────────────────
    po     = result["pointmap_obj"].astype(np.float64)
    po_h   = np.hstack([po, np.ones((len(po), 1))])
    po_w   = (T_wc @ po_h.T).T[:, :3]

    # ── Build figure ──────────────────────────────────────────────────────────
    fig = build_figure(meshes, participant, video_id)
    add_camera_pose_to_figure(fig, frame, axis_len=0.15)

    if len(slam_near):
        fig.add_trace(go.Scatter3d(
            x=slam_near[:, 0], y=slam_near[:, 1], z=slam_near[:, 2],
            mode="markers",
            marker=dict(size=3, color="#FF4444", opacity=0.8),
            name=f"SLAM near object ({len(slam_near):,} pts)",
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra>SLAM</extra>",
        ))

    fig.add_trace(go.Scatter3d(
        x=po_w[:, 0], y=po_w[:, 1], z=po_w[:, 2],
        mode="markers",
        marker=dict(size=5, color="#BB44FF", opacity=0.95),
        name=f"SAM3D pointmap_obj ({len(po_w):,} pts)",
        hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra>SAM3D</extra>",
    ))

    fig.update_layout(
        title=dict(
            text=(
                f"SLAM (red) vs SAM3D (violet) — {participant}  |  {video_id}"
                f"  |  frame {frame['frame_index']}"
            ),
        ),
        legend=dict(
            bgcolor="#1e1e30", bordercolor="#444466",
            borderwidth=1, font=dict(size=11),
        ),
    )
    return fig


def build_pointcloud_figure(
    pts: np.ndarray,
    participant: str,
    video_id: str,
    slam_index: str,
) -> go.Figure:
    """Build a standalone Plotly 3D scatter figure from world-frame point cloud data."""
    fig = go.Figure(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="markers",
        marker=dict(
            size=1.5,
            color=pts[:, 2],
            colorscale="Viridis",
            colorbar=dict(title="Z (m)", thickness=12),
            opacity=0.6,
        ),
        name="Semidense points",
        hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=(f"Point Cloud — {participant}  |  {video_id}"
                  f"  (SLAM seg {slam_index})"),
            font=dict(size=16),
        ),
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode="data",
            bgcolor="#0a0a15",
            xaxis=dict(showbackground=True, backgroundcolor="#050510",
                       gridcolor="#222244", color="#aaaacc"),
            yaxis=dict(showbackground=True, backgroundcolor="#050510",
                       gridcolor="#222244", color="#aaaacc"),
            zaxis=dict(showbackground=True, backgroundcolor="#050510",
                       gridcolor="#222244", color="#aaaacc"),
        ),
        paper_bgcolor="#08080f",
        font=dict(color="#ccccdd"),
        margin=dict(l=0, r=0, b=0, t=50),
        height=700,
    )
    return fig


# ── RGB calibration ────────────────────────────────────────────────────────────

def load_rgb_calibration(video_id: str) -> dict:
    """Load RGB camera calibration from device_calibration.json.

    Returns a dict with:
      params           – 15-element FISHEYE624 list [f, cu, cv, k0..k5, p0, p1, s0..s3]
      image_size       – [width, height]
      T_device_camera  – (3, 4) ndarray
      valid_radius     – circular validity mask radius in pixels (or None)
    """
    participant = participant_from_video_id(video_id)
    calib_path  = FRAMEWISE_BASE / participant / video_id / "device_calibration.json"
    with open(calib_path) as fh:
        calib = json.load(fh)
    cam = calib["cameras"]["camera-rgb"]
    return {
        "params":          cam["projection_params"],
        "image_size":      cam["image_size"],
        "T_device_camera": np.array(cam["T_device_camera"]),
        "valid_radius":    cam.get("valid_radius"),
    }


def project_fisheye624(
    pts_cam: np.ndarray,
    params: list,
) -> tuple[np.ndarray, np.ndarray]:
    """Project (N, 3) camera-frame points with the FISHEYE624 model.

    Model: [f, cu, cv, k0..k5, p0, p1, s0..s3]

    Radial polynomial:
        theta_d = theta + k0*theta³ + k1*theta⁵ + … + k5*theta¹³
    followed by tangential (p0, p1) and thin-prism (s0..s3) distortion.

    Returns u, v pixel arrays (NaN for degenerate cases).
    """
    f, cu, cv = params[0], params[1], params[2]
    k          = params[3:9]
    p0, p1     = params[9], params[10]
    s          = params[11:15]

    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    r     = np.sqrt(x**2 + y**2)
    theta = np.arctan2(r, z)

    theta_d = theta.copy()
    for i, ki in enumerate(k):
        theta_d += ki * theta ** (2 * i + 3)

    safe_r = np.where(r < 1e-9, 1.0, r)
    mx = np.where(r < 1e-9, 0.0, x / safe_r * theta_d)
    my = np.where(r < 1e-9, 0.0, y / safe_r * theta_d)

    r2 = mx**2 + my**2
    dx = 2 * p0 * mx * my + p1 * (r2 + 2 * mx**2)
    dy = p0 * (r2 + 2 * my**2) + 2 * p1 * mx * my

    xd = mx + dx + s[0] * r2 + s[1] * r2**2
    yd = my + dy + s[2] * r2 + s[3] * r2**2

    return f * xd + cu, f * yd + cv


def render_depth_map(
    video_id: str,
    frame: dict,
    pts_world: np.ndarray,
    max_depth: float = 5.0,
    plot: bool = True,
    return_maps: bool = False,
):
    """Render pts_world as a 2D depth map seen from the RGB camera at the given frame.

    Parameters
    ----------
    plot        : If True, display a matplotlib scatter plot (default behaviour).
    return_maps : If True, return (depth_img, valid_mask) arrays instead of None.
    """
    cam  = load_rgb_calibration(video_id)
    W, H = cam["image_size"]

    T_wd  = np.array(frame["T_world_device"])
    T_wc  = _to_4x4(T_wd) @ _to_4x4(cam["T_device_camera"])
    T_cw  = np.linalg.inv(T_wc)

    N     = len(pts_world)
    pts_h = np.hstack([pts_world, np.ones((N, 1))])
    pts_c = (T_cw @ pts_h.T).T[:, :3]

    mask  = pts_c[:, 2] > 0.1
    pts_c = pts_c[mask]

    u, v  = project_fisheye624(pts_c, cam["params"])
    depth = pts_c[:, 2]

    vr    = cam["valid_radius"] or (min(W, H) / 2)
    cu_px, cv_px = cam["params"][1], cam["params"][2]
    keep  = (
        (u >= 0) & (u < W) &
        (v >= 0) & (v < H) &
        (depth <= max_depth) &
        (np.hypot(u - cu_px, v - cv_px) < vr)
    )
    u, v, depth = u[keep], v[keep], depth[keep]
    print(f"Projecting {keep.sum():,} / {mask.sum():,} points into the image.")

    if return_maps:
        depth_img = np.full((H, W), np.nan, dtype=np.float32)
        valid_mask = np.zeros((H, W), dtype=bool)
        ui = np.clip(u.astype(int), 0, W - 1)
        vi = np.clip(v.astype(int), 0, H - 1)
        depth_img[vi, ui] = depth
        valid_mask[vi, ui] = True
        return depth_img, valid_mask

    if not plot:
        return None

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#0a0a15")
    ax.set_facecolor("black")

    sc = ax.scatter(u, v, c=depth, s=0.5, cmap="plasma",
                    vmin=0, vmax=max_depth, linewidths=0, rasterized=True)

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.set_title(
        f"Depth Map (RGB) — {video_id}  |  frame {frame['frame_index']}",
        color="#ccccdd", fontsize=11, pad=8,
    )
    ax.set_xlabel("u  (px)", color="#aaaacc", fontsize=9)
    ax.set_ylabel("v  (px)", color="#aaaacc", fontsize=9)
    ax.tick_params(colors="#888899")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333344")

    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("depth  (m)", color="#ccccdd", fontsize=9)
    cb.ax.yaxis.set_tick_params(color="#888899")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#888899")
    plt.tight_layout()
    plt.show()
    return None


# ── Digital twin pipeline ──────────────────────────────────────────────────────

def extract_frame(video_id: str, frame_index: int, out_path: Path) -> Path:
    """Extract a single JPEG frame from an HD-EPIC MP4 using ffmpeg."""
    participant = participant_from_video_id(video_id)
    video_path  = _VIDEOS_ROOT / participant / f"{video_id}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    ts  = frame_index / _VIDEO_FPS
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{ts:.6f}",
        "-i", str(video_path),
        "-vframes", "1", "-q:v", "2",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-600:]}")
    return out_path


def submit_twin_job(
    video_id: str,
    frame_index: int,
    text_prompt: str,
    out_base: Path,
) -> tuple[Path, str]:
    """Extract the frame, write a Slurm sbatch script, and submit it.

    Returns (work_dir, job_id).
    """
    slug     = text_prompt.lower().replace(" ", "_")
    work_dir = out_base / video_id / f"frame{frame_index}_{slug}"
    work_dir.mkdir(parents=True, exist_ok=True)
    _SLURM_LOGS.mkdir(parents=True, exist_ok=True)

    frame_path   = work_dir / "frame.jpg"
    extract_frame(video_id, frame_index, frame_path)
    print(f"Extracted frame  → {frame_path}")

    sam3_script  = _VIEWER_ROOT / "run_sam3.py"
    sam3d_script = _VIEWER_ROOT / "run_sam3d.py"
    depth_script = _VIEWER_ROOT / "run_depth.py"
    sbatch_path  = work_dir / "job.sbatch"

    _conda_base = "/usr/prakt/s0021/miniconda3"
    sbatch_path.write_text(
        "#!/bin/bash\n"
        f'#SBATCH --job-name="twin_{slug}"\n'
        "#SBATCH --nodes=1\n"
        "#SBATCH --cpus-per-task=4\n"
        "#SBATCH --gres=gpu:1,VRAM:32G\n"
        "#SBATCH --mem=48G\n"
        "#SBATCH --time=00:30:00\n"
        "#SBATCH --mail-type=None\n"
        f"#SBATCH --output={_SLURM_LOGS}/slurm-%j.out\n"
        "\n"
        "set -e\n"
        f"source {_conda_base}/etc/profile.d/conda.sh\n"
        "\n"
        "echo '=== Step 1: SAM3 segmentation ==='\n"
        f"conda run -n sam3 python {sam3_script} \\\n"
        f"    --image  {frame_path} \\\n"
        f'    --prompt "{text_prompt}" \\\n'
        f"    --out-dir {work_dir}\n"
        "\n"
        "conda activate sam3\n"
        "echo '=== Step 2: Monocular depth estimation ==='\n"
        f"python {depth_script} \\\n"
        f"    --image   {frame_path} \\\n"
        f"    --out-dir {work_dir}\n"
        "\n"
        "echo '=== Step 3: SAM-3D-Objects reconstruction ==='\n"
        f"conda run -n sam3d-objects python {sam3d_script} \\\n"
        f"    --image   {frame_path} \\\n"
        f"    --mask    {work_dir}/mask.png \\\n"
        f"    --out-dir {work_dir}\n"
        "\n"
        "echo '=== Done ==='\n"
    )
    print(f"Wrote sbatch     → {sbatch_path}")

    result = subprocess.run(
        ["sbatch", str(sbatch_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed:\n{result.stderr}")
    job_id = result.stdout.strip().split()[-1]
    print(f"Submitted job {job_id}")
    print(f"Monitor: tail -f {_SLURM_LOGS}/slurm-{job_id}.out")
    return work_dir, job_id


def load_twin_result(work_dir: Path) -> dict:
    """Load artefacts written by run_sam3.py and run_sam3d.py from work_dir."""
    result = {}
    spec = [
        ("frame.jpg",        "frame",        "image"),
        ("mask.png",         "mask",         "mask"),
        ("box.npy",          "box",          "npy"),
        ("scores.npy",       "scores",       "npy"),
        ("pointmap_obj.npy", "pointmap_obj", "npy"),
        ("means_cam.npy",    "means_cam",    "npy"),
        ("depth_raw.npy",    "depth_raw",    "npy"),
        ("depth_vis.png",    "depth_vis",    "image"),
    ]
    for fname, key, kind in spec:
        p = work_dir / fname
        if not p.exists():
            print(f"  [missing] {fname}")
            continue
        if kind == "image":
            result[key] = np.array(PILImage.open(p).convert("RGB"))
        elif kind == "mask":
            result[key] = np.array(PILImage.open(p)) > 0
        else:
            result[key] = np.load(p)
        print(f"  [loaded]  {fname}")

    twin_pkl = work_dir / "twin.pkl"
    if twin_pkl.exists():
        with open(twin_pkl, "rb") as fh:
            result["twin_meta"] = pickle.load(fh)
        print("  [loaded]  twin.pkl")
    else:
        print("  [missing] twin.pkl")

    return result


def align_depth_raw_to_slam(
    result: dict,
    frame: dict,
    video_id: str,
    pts_world: np.ndarray,
    mask_dilation: int = 40,
    ransac_n_iter: int = 1000,
    ransac_inlier_thresh: float = 0.3,
    max_depth: float = 5.0,
    rng_seed: int = 42,
) -> dict:
    """Align depth_raw (Depth Anything disparity) to metric SLAM depth, then
    correct pointmap_obj by scaling along camera rays.

    Pipeline
    --------
    1. Project SLAM world pts → sparse metric depth map D_slam[H×W].
    2. Collect pixel-level correspondences (D_raw[v,u], D_slam[v,u]) in the
       dilated object mask.  Falls back to global (full image) if < 20 local
       pairs are found.
    3. RANSAC linear fit:  D_metric ≈ α · D_raw + β
       (α < 0 because depth_raw is disparity: larger = closer)
    4. Apply to mask pixels → median corrected depth D_metric_median.
    5. Scale pointmap_obj uniformly along camera rays:
         ratio       = D_metric_median / mean(pointmap_obj[:, 2])
         pts_aligned = pointmap_obj * ratio
       This moves the whole object cloud to the metrically-corrected depth
       while preserving its shape and each point's image-plane direction.

    Returns
    -------
    dict with keys:
        pts_aligned    – (N, 3) corrected camera-frame points
        alpha, beta    – RANSAC linear coefficients
        D_metric_median– median metric depth over object mask
        Z_sam3d        – original mean Z of pointmap_obj
        ratio          – D_metric_median / Z_sam3d
        inlier_mask    – bool array over correspondences used for RANSAC
        n_corr         – total correspondences before RANSAC
        D_hat          – D_raw values at correspondence pixels
        D_slam_corr    – D_slam values at correspondence pixels
        slam_depth_img – (H, W) sparse SLAM depth (NaN where missing)
        slam_valid     – (H, W) bool mask of valid SLAM pixels
        obj_mask       – (H, W) object segmentation mask
    """
    for key in ("depth_raw", "mask", "pointmap_obj"):
        if key not in result:
            raise ValueError(f"result must contain '{key}' for depth alignment.")

    cam  = load_rgb_calibration(video_id)
    W, H = cam["image_size"]

    depth_raw = result["depth_raw"].astype(np.float64)
    mask      = result["mask"]
    pts_c     = result["pointmap_obj"].astype(np.float64)

    if mask.shape[:2] != (H, W):
        raise ValueError(f"Mask shape {mask.shape[:2]} != camera (H, W)={H, W}")
    if depth_raw.shape != (H, W):
        raise ValueError(f"depth_raw shape {depth_raw.shape} != camera (H, W)={H, W}")

    # ── 1. Render sparse SLAM depth map ──────────────────────────────────────
    slam_depth_img, slam_valid = render_depth_map(
        video_id, frame, pts_world,
        max_depth=max_depth, plot=False, return_maps=True,
    )

    # ── 2. Mask (optionally dilated) → pixel correspondences ─────────────────
    if ALIGN_USE_MASK_DILATION:
        struct  = scipy.ndimage.generate_binary_structure(2, 1)
        search_mask = scipy.ndimage.binary_dilation(
            mask, structure=struct, iterations=mask_dilation
        )
        region_label = f"dilated mask ({mask_dilation} px) ∩ SLAM"
    else:
        search_mask  = mask
        region_label = "exact mask ∩ SLAM"

    region = search_mask & slam_valid
    D_hat  = depth_raw[region]
    D_slam = slam_depth_img[region]

    print(f"[align] SLAM valid pixels        : {slam_valid.sum():,}")
    print(f"[align] Local correspondences    : {region.sum():,}  ({region_label})")

    # Fallback: use full image if local region is too sparse
    if region.sum() < 20:
        print("[align] < 20 local pairs → falling back to global correspondences.")
        D_hat  = depth_raw[slam_valid]
        D_slam = slam_depth_img[slam_valid]
        print(f"[align] Global correspondences   : {slam_valid.sum():,}")

    N = len(D_hat)

    # ── 3. RANSAC linear fit:  D_metric ≈ α·D_raw + β ────────────────────────
    rng          = np.random.default_rng(rng_seed)
    best_alpha   = -1.0
    best_beta    = float(np.median(D_slam))
    best_inliers = np.zeros(N, dtype=bool)

    for _ in range(ransac_n_iter):
        idx = rng.choice(N, size=2, replace=False)
        # guard against identical D_hat values (degenerate)
        if np.abs(D_hat[idx[0]] - D_hat[idx[1]]) < 1e-6:
            continue
        a, b      = np.polyfit(D_hat[idx], D_slam[idx], 1)
        residuals = np.abs(a * D_hat + b - D_slam)
        inliers   = residuals < ransac_inlier_thresh
        if inliers.sum() > best_inliers.sum():
            best_alpha, best_beta, best_inliers = a, b, inliers

    # Refit OLS on all inliers for maximum accuracy
    if best_inliers.sum() >= 2:
        A = np.column_stack([D_hat[best_inliers], np.ones(best_inliers.sum())])
        coeffs, *_ = np.linalg.lstsq(A, D_slam[best_inliers], rcond=None)
        best_alpha, best_beta = float(coeffs[0]), float(coeffs[1])

    print(
        f"[align] RANSAC  α={best_alpha:.5f}  β={best_beta:.4f}  "
        f"inliers={best_inliers.sum():,}/{N:,}"
    )

    # ── 4. Corrected metric depth over the object mask ────────────────────────
    D_corrected     = best_alpha * depth_raw[mask] + best_beta
    valid_corrected = D_corrected > 0.05
    if valid_corrected.sum() == 0:
        print("[align] All corrected depths are non-positive — check fit. "
              "Returning uncorrected points.")
        D_metric_median = float(np.mean(pts_c[:, 2]))
    else:
        D_metric_median = float(np.median(D_corrected[valid_corrected]))

    print(
        f"[align] Mask pixels used         : {valid_corrected.sum():,}/{mask.sum():,}"
    )
    print(f"[align] D_metric_median (mask)   : {D_metric_median:.4f} m")

    # ── 5. Scale pointmap_obj along camera rays ───────────────────────────────
    Z_sam3d = float(np.mean(pts_c[:, 2]))
    ratio   = D_metric_median / Z_sam3d if abs(Z_sam3d) > 1e-6 else 1.0
    pts_aligned = pts_c * ratio

    print(f"[align] Z_sam3d (mean)           : {Z_sam3d:.4f} m")
    print(f"[align] ratio                    : {ratio:.4f}")
    print(
        f"[align] pts_aligned Z range      : "
        f"{pts_aligned[:, 2].min():.4f} → {pts_aligned[:, 2].max():.4f} m"
    )

    return {
        "pts_aligned":     pts_aligned,
        "alpha":           best_alpha,
        "beta":            best_beta,
        "D_metric_median": D_metric_median,
        "Z_sam3d":         Z_sam3d,
        "ratio":           ratio,
        "inlier_mask":     best_inliers,
        "n_corr":          N,
        "D_hat":           D_hat,
        "D_slam_corr":     D_slam,
        "slam_depth_img":  slam_depth_img,
        "slam_valid":      slam_valid,
        "obj_mask":        mask,
    }


def plot_depth_alignment(
    align_info: dict,
    frame: dict,
    video_id: str,
    text_prompt: str,
) -> None:
    """Diagnostic figure for the depth_raw → metric SLAM alignment.

    Panel 1  –  scatter of depth_raw (disparity) vs D_slam (metres) for all
                correspondence pixels; inliers (green) / outliers (red) with
                the fitted line α·D_raw + β.
    Panel 2  –  SLAM sparse depth image with the object mask overlaid and the
                correspondence pixels marked (green = inlier, red = outlier).
    """
    inlier_mask = align_info["inlier_mask"]
    alpha       = align_info["alpha"]
    beta        = align_info["beta"]
    D_hat       = align_info["D_hat"]
    D_slam      = align_info["D_slam_corr"]
    slam_depth  = align_info["slam_depth_img"]
    slam_valid  = align_info["slam_valid"]
    obj_mask    = align_info["obj_mask"]

    N = len(D_hat)
    if N == 0:
        print("[plot_depth_alignment] No correspondences — nothing to plot.")
        return

    outliers = ~inlier_mask

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0a0a15")
    for ax in axes:
        ax.set_facecolor("#0a0a15")

    # ── Panel 1: depth_raw vs D_slam scatter + fit line ───────────────────────
    ax = axes[0]
    ax.scatter(
        D_hat[outliers], D_slam[outliers],
        c="#FF4444", s=8, alpha=0.4,
        label=f"Outliers ({outliers.sum():,})", zorder=2,
    )
    ax.scatter(
        D_hat[inlier_mask], D_slam[inlier_mask],
        c="#44FF88", s=8, alpha=0.6,
        label=f"Inliers ({inlier_mask.sum():,})", zorder=3,
    )
    d_range = np.array([D_hat.min(), D_hat.max()])
    ax.plot(
        d_range, alpha * d_range + beta,
        color="#FFD700", linewidth=2,
        label=f"fit: α={alpha:.5f},  β={beta:.3f} m",
        zorder=4,
    )
    ax.set_xlabel("depth_raw  (disparity, larger = closer)", color="#aaaacc", fontsize=10)
    ax.set_ylabel("D_slam  (metric, metres)", color="#aaaacc", fontsize=10)
    ax.set_title(
        f"depth_raw → SLAM alignment  —  '{text_prompt}'\n"
        f"frame {frame['frame_index']}  |  {video_id}",
        color="#ccccdd", fontsize=10, pad=6,
    )
    ax.tick_params(colors="#888899")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333344")
    ax.legend(fontsize=9, labelcolor="#ccccdd",
               facecolor="#1a1a2e", edgecolor="#444466")

    # ── Panel 2: SLAM depth image + mask + correspondence pixels ──────────────
    ax2 = axes[1]
    slam_vis = slam_depth.copy()
    slam_vis[~slam_valid] = np.nan
    vmax = float(np.nanpercentile(slam_depth[slam_valid], 95)) if slam_valid.any() else 5.0
    ax2.imshow(slam_vis, cmap="plasma", origin="upper", interpolation="none",
               vmin=0, vmax=vmax)

    # Object mask outline
    if obj_mask is not None:
        mask_rgba = np.zeros((*obj_mask.shape[:2], 4), dtype=np.float32)
        mask_rgba[obj_mask] = [0.0, 0.8, 1.0, 0.35]
        ax2.imshow(mask_rgba, origin="upper", interpolation="none")

    # Correspondence pixels — need their (u, v) positions.
    # Reconstruct from slam_valid: take all pixels in region that are valid.
    ys_all, xs_all = np.where(slam_valid)
    # We used at most N of them (the first N after masking); just show all slam_valid
    # and colour by inlier status using the same ordering.
    # Since D_hat = depth_raw[region] and region = dilated & slam_valid (or global),
    # we don't store per-pixel coords in the dict.  Reconstruct the region mask.
    # Show slam_valid pixels coloured by whether they were inliers or not.
    # For simplicity: plot all slam_valid as dots; highlight the inlier subset.
    ax2.scatter(xs_all, ys_all,
                c="#888888", s=1.5, alpha=0.3, linewidths=0,
                label=f"SLAM pixels ({len(xs_all):,})", zorder=2)

    ax2.set_title(
        f"SLAM sparse depth  +  object mask\n"
        f"α={alpha:.5f}  β={beta:.3f} m  |  "
        f"D_metric_median={align_info.get('D_metric_median', float('nan')):.3f} m",
        color="#ccccdd", fontsize=10, pad=6,
    )
    ax2.axis("off")
    ax2.legend(fontsize=8, labelcolor="#ccccdd",
                facecolor="#1a1a2e", edgecolor="#444466")

    fig.tight_layout()
    plt.show()


def visualize_twin(
    result: dict,
    frame: dict,
    video_id: str,
    text_prompt: str,
    meshes: list,
    participant: str,
    pts_world: np.ndarray | None = None,
) -> None:
    """Render figures for the digital-twin result.

    Figure 1 (matplotlib)  – extracted frame with SAM3 mask overlay and 2-D bbox.
    Figure 1b (matplotlib) – monocular depth map with mask overlay (if depth_raw present).
    Figure 1c (matplotlib) – depth-alignment diagnostic (if pts_world provided).
    Figure 2 (Plotly 3D)   – kitchen scene with camera pose and the object's 3-D bbox.

    Parameters
    ----------
    pts_world : (N, 3) SLAM semidense point cloud in world frame.  When provided,
                the object's depth is aligned to metric SLAM scale via RANSAC
                linear regression before the 3-D bbox is constructed.
    """
    # ── Figure 1: frame + mask + 2-D bbox ────────────────────────────────────
    fig1, ax = plt.subplots(figsize=(9, 9), facecolor="#0a0a15")
    ax.set_facecolor("black")

    if "frame" in result:
        ax.imshow(result["frame"])

    if "mask" in result:
        h, w = result["mask"].shape[:2]
        ov   = np.zeros((h, w, 4), dtype=np.float32)
        ov[result["mask"]] = [0.0, 0.8, 1.0, 0.45]
        ax.imshow(ov)

    if "box" in result:
        x1, y1, x2, y2 = result["box"]
        ax.add_patch(plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2.5, edgecolor="#FFD700", facecolor="none",
        ))
        _depth_label = ""
        if "depth_raw" in result and "mask" in result:
            _d     = result["depth_raw"]
            _d_rng = _d.max() - _d.min() + 1e-8
            _med   = float(np.median(_d[result["mask"]]))
            _rel   = (_med - _d.min()) / _d_rng
            _depth_label = f"\ndist rank: {1.0 - _rel:.1%} of scene"
        ax.text(
            x1, y1 - 7, f'"{text_prompt}"{_depth_label}',
            color="#FFD700", fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5),
        )

    ax.axis("off")
    ax.set_title(
        f"SAM3  |  {video_id}  |  frame {frame['frame_index']}  |  '{text_prompt}'",
        color="#ccccdd", fontsize=11, pad=8,
    )
    fig1.tight_layout()
    plt.show()

    # ── Figure 1b: depth map + segmentation mask overlay ─────────────────────
    if "depth_raw" in result:
        _d = result["depth_raw"]
        _d_min, _d_max = _d.min(), _d.max()
        _d_norm = ((_d - _d_min) / (_d_max - _d_min + 1e-8) * 255).astype(np.uint8)
        _cmap_s = _mpl.colormaps.get_cmap("Spectral_r")
        _d_rgb  = (_cmap_s(_d_norm)[:, :, :3] * 255).astype(np.uint8)
        fig_d, ax_d = plt.subplots(figsize=(9, 5), facecolor="#0a0a15")
        ax_d.set_facecolor("black")
        ax_d.imshow(_d_rgb)
        _dist_info = ""
        if "mask" in result:
            _ov_d = np.zeros((*result["mask"].shape[:2], 4), dtype=np.float32)
            _ov_d[result["mask"]] = [0.0, 0.8, 1.0, 0.45]
            ax_d.imshow(_ov_d)
            _med  = float(np.median(_d[result["mask"]]))
            _rel  = (_med - _d_min) / (_d_max - _d_min + 1e-8)
            _dist_info = (
                f"monocular disparity w masce: {_rel:.1%}  "
                f"→ odleglosc wzgledna: {1.0 - _rel:.1%} zakresu sceny"
            )
        ax_d.axis("off")
        ax_d.set_title(
            f"Depth + Maska  |  {video_id}  |  klatka {frame['frame_index']}\n{_dist_info}",
            color="#ccccdd", fontsize=11, pad=8,
        )
        fig_d.tight_layout()
        plt.show()

    # ── Figure 2: 3-D scene + object bounding box ─────────────────────────────
    # Prefer pointmap_obj (dense pixel-wise depth) over means_cam (layout-posed
    # Gaussian means whose z-values can be un-physical after layout scaling).
    pts_key = "pointmap_obj" if "pointmap_obj" in result else (
              "means_cam"    if "means_cam"    in result else None)
    if pts_key is None:
        print("No 3-D data available — skipping 3-D figure (job may not be finished).")
        return

    meta = result.get("twin_meta") or {}
    ver  = meta.get("sam3d_frame_version")
    if ver is None or ver < 2:
        print(
            "Warning: Digital-twin point cloud may be from an old run (missing "
            "sam3d_frame_version >= 2). Re-run run_sam3d.py so means_cam / "
            "pointmap_obj are in HD-EPIC RGB camera coordinates."
        )

    cam  = load_rgb_calibration(video_id)
    T_wc = _to_4x4(np.array(frame["T_world_device"])) @ _to_4x4(cam["T_device_camera"])

    # ── depth alignment ───────────────────────────────────────────────────────
    pts_c = result[pts_key]
    can_align = (
        pts_world is not None
        and pts_key == "pointmap_obj"
        and "mask"      in result
        and "depth_raw" in result
    )
    if can_align:
        print("[visualize_twin] Running depth_raw → SLAM alignment …")
        align_info = align_depth_raw_to_slam(result, frame, video_id, pts_world)
        plot_depth_alignment(align_info, frame, video_id, text_prompt)
        pts_c = align_info["pts_aligned"]
    elif pts_world is None:
        print("[visualize_twin] No SLAM cloud provided — using raw SAM3D depth "
              "(pass pts_world= for metric alignment).")
    else:
        missing = [k for k in ("mask", "depth_raw") if k not in result]
        print(f"[visualize_twin] Missing artefacts for alignment: {missing} — "
              "using raw SAM3D depth.")

    pts_h = np.hstack([pts_c, np.ones((len(pts_c), 1))])
    pts_w = (T_wc @ pts_h.T).T[:, :3]

    centre = pts_w.mean(0)
    dists  = np.linalg.norm(pts_w - centre, axis=1)
    pts_w  = pts_w[dists < dists.mean() + 3 * dists.std()]
    if len(pts_w) == 0:
        print("Object point cloud is empty after outlier removal.")
        return

    lo, hi  = pts_w.min(0), pts_w.max(0)
    corners = np.array([
        [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
        [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
        [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]],
    ])
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]

    fig2 = build_figure(meshes, participant, video_id)
    add_camera_pose_to_figure(fig2, frame, axis_len=0.15)

    for i, (a, b) in enumerate(edges):
        fig2.add_trace(go.Scatter3d(
            x=[corners[a, 0], corners[b, 0]],
            y=[corners[a, 1], corners[b, 1]],
            z=[corners[a, 2], corners[b, 2]],
            mode="lines",
            line=dict(color="#FFD700", width=3),
            name="Object bbox" if i == 0 else None,
            showlegend=(i == 0),
        ))

    ctr = corners.mean(0)
    fig2.add_trace(go.Scatter3d(
        x=[ctr[0]], y=[ctr[1]], z=[ctr[2]],
        mode="markers+text",
        marker=dict(size=5, color="#FFD700"),
        text=[f'"{text_prompt}"'],
        textposition="top center",
        textfont=dict(color="#FFD700", size=12),
        name=f"'{text_prompt}'",
    ))

    fig2.update_layout(title=dict(
        text=(
            f"Digital Twin  —  {participant}  |  {video_id}"
            f"  |  frame {frame['frame_index']}  |  '{text_prompt}'"
        ),
    ))
    fig2.show()
