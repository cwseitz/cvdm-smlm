import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import yaml
from skimage.filters import median
from skimage.io import imread
from skimage.morphology import disk
from skimage.restoration import rolling_ball
import tifffile

from cvdm.make.kde import BasicKDE
from cvdm.psf.mle2d import PipelineMLE2D


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _read_frame(path: str, idx: int) -> np.ndarray:
    arr = imread(path)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    return arr[idx].astype(np.float32)


def _slice_2d(img: np.ndarray, coord: List[int], size: int) -> np.ndarray:
    x0, y0 = int(coord[0]), int(coord[1])
    return img[x0 : x0 + size, y0 : y0 + size]


def _find_existing(candidates: List[str]) -> str:
    for path in candidates:
        if os.path.exists(path):
            return path
    joined = "\n  - ".join(candidates)
    raise FileNotFoundError(f"None of these paths exist:\n  - {joined}")


def _profile_norm(row: np.ndarray, start: int, stop: int) -> np.ndarray:
    vals = row[start:stop].astype(np.float32)
    denom = float(np.max(vals)) if vals.size else 1.0
    if denom <= 0:
        denom = 1.0
    return vals / denom


def _find_z_shards(results_dir: str) -> List[str]:
    files = []
    for name in os.listdir(results_dir):
        if name.startswith("z-") and name.endswith("-0.tif"):
            files.append(name)

    def _sort_key(name: str) -> int:
        try:
            return int(name.split("-")[1])
        except Exception:
            return 10**9

    files.sort(key=_sort_key)
    return [os.path.join(results_dir, name) for name in files]


def _normalize_frame_stack(arr: np.ndarray, source: str) -> np.ndarray:
    """Normalize image-like arrays to (n_frames, height, width)."""
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr[None, ...].astype(np.float32)
    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            return arr[..., 0][None, ...].astype(np.float32)
        if arr.shape[0] == 1:
            return arr.astype(np.float32)
        return arr.astype(np.float32)
    if arr.ndim == 4:
        if arr.shape[0] == 1:
            return arr[0].astype(np.float32)
        if arr.shape[1] == 1:
            return arr[:, 0, :, :].astype(np.float32)
        if arr.shape[-1] == 1:
            return arr[:, :, :, 0].astype(np.float32)
    raise ValueError(
        f"Unsupported z-stack shape from {source}: {arr.shape}. Expected 2D/3D/4D with singleton channel."
    )


def _normalize_single_frame(arr: np.ndarray, source: str) -> np.ndarray:
    """Normalize a single shard image to (height, width)."""
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            return arr[0].astype(np.float32)
        if arr.shape[-1] == 1:
            return arr[..., 0].astype(np.float32)
    raise ValueError(f"Unsupported z-shard shape from {source}: {arr.shape}. Expected 2D or singleton-channel 3D.")


def _load_z_frames(results_dir: str) -> np.ndarray:
    stack_path = os.path.join(results_dir, "z_stack.tif")
    if os.path.exists(stack_path):
        arr = imread(stack_path)
        return _normalize_frame_stack(arr, stack_path)

    z_paths = _find_z_shards(results_dir)
    if not z_paths:
        raise FileNotFoundError(
            "No z outputs found. Expected one of:\n"
            f"  - {stack_path}\n"
            f"  - {results_dir}/z-*-0.tif"
        )
    frames = [_normalize_single_frame(imread(path), path) for path in z_paths]
    return np.stack(frames, axis=0)


def _load_x_stack_image(results_dir: str) -> np.ndarray:
    x_path = os.path.join(results_dir, "x_stack.tif")
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Missing x_stack.tif: {x_path}")
    arr = np.asarray(imread(x_path))
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3:
        return arr[0].astype(np.float32)
    raise ValueError(f"Unsupported x_stack shape from {x_path}: {arr.shape}")


def _rescale_coords(coords: np.ndarray, src_shape: Tuple[int, int], dst_shape: Tuple[int, int]) -> np.ndarray:
    if coords.size == 0:
        return coords
    src_h, src_w = float(src_shape[0]), float(src_shape[1])
    dst_h, dst_w = float(dst_shape[0]), float(dst_shape[1])
    scale_x = dst_h / src_h
    scale_y = dst_w / src_w
    out = coords.astype(np.float32).copy()
    out[:, 0] *= scale_x
    out[:, 1] *= scale_y
    return out


def _frame_to_rgb(frame: np.ndarray) -> np.ndarray:
    f = frame.astype(np.float32)
    lo = float(np.percentile(f, 1.0))
    hi = float(np.percentile(f, 99.0))
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((f - lo) / (hi - lo), 0.0, 1.0)
    gray = (255.0 * norm).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def _draw_points_overlay(rgb: np.ndarray, coords: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    h, w, _ = out.shape
    for x_val, y_val in coords:
        x = int(round(float(x_val)))
        y = int(round(float(y_val)))
        for dx in (-1, 0, 1):
            xi = x + dx
            if 0 <= xi < h and 0 <= y < w:
                out[xi, y] = np.array([255, 0, 0], dtype=np.uint8)
        for dy in (-1, 0, 1):
            yi = y + dy
            if 0 <= x < h and 0 <= yi < w:
                out[x, yi] = np.array([255, 0, 0], dtype=np.uint8)
    return out


def _imshow_percentile_gray(ax: plt.Axes, image: np.ndarray, low: float = 1.0, high: float = 99.5) -> None:
    img = np.asarray(image, dtype=np.float32)
    vmin, vmax = np.percentile(img, [low, high])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.min(img))
        vmax = float(np.max(img))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)


def _save_detection_gif(
    gif_path: str,
    frames: np.ndarray,
    detections: List[np.ndarray],
    threshold: float,
    min_sigma: float,
    max_sigma: float,
    stride: int,
) -> None:
    pil_frames: List[Image.Image] = []
    for idx in range(0, len(frames), max(1, stride)):
        rgb = _frame_to_rgb(frames[idx])
        overlay = _draw_points_overlay(rgb, detections[idx])
        img = Image.fromarray(overlay, mode="RGB")
        draw = ImageDraw.Draw(img)
        draw.text(
            (5, 5),
            f"f={idx} n={len(detections[idx])} thr={threshold:.3f} sig=({min_sigma:.2f},{max_sigma:.2f})",
            fill=(255, 255, 0),
        )
        pil_frames.append(img)

    if not pil_frames:
        return
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=180,
        loop=0,
    )


def _build_cvdm_render_from_detections(
    frames: np.ndarray,
    threshold: float,
    min_sigma: float,
    max_sigma: float,
    kde_sigma: float,
    max_spots_per_frame: int,
    median_filter_radius_px: int,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    h, w = frames.shape[1], frames.shape[2]
    accum = np.zeros((h, w), dtype=np.float32)
    all_detections: List[np.ndarray] = []

    for frame in frames:
        frame_for_detection = frame
        if median_filter_radius_px > 0:
            frame_for_detection = median(frame_for_detection, footprint=disk(median_filter_radius_px))

        det = PipelineMLE2D(frame_for_detection[None, ...]).localize(
            threshold=threshold,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            fit_enabled=False,
            show_tqdm=False,
        )
        if det.empty:
            coords = np.empty((0, 2), dtype=np.float32)
        else:
            if max_spots_per_frame > 0 and len(det) > max_spots_per_frame and "peak" in det.columns:
                det = det.sort_values("peak", ascending=False).head(max_spots_per_frame)
            coords = det[["x", "y"]].to_numpy(dtype=np.float32)

        all_detections.append(coords)
        if coords.shape[0] == 0:
            continue

        kde = BasicKDE(coords).forward(h, upsample=1, sigma=kde_sigma)
        accum += kde.astype(np.float32)

    if len(frames) > 0:
        accum /= float(len(frames))
    return accum, all_detections


def _aggregate_detection_coords(
    frames: np.ndarray,
    threshold: float,
    min_sigma: float,
    max_sigma: float,
    max_spots_per_frame: int,
    median_filter_radius_px: int,
    fit_enabled: bool,
) -> np.ndarray:
    all_coords: List[np.ndarray] = []
    for frame in frames:
        frame_for_detection = frame
        if median_filter_radius_px > 0:
            frame_for_detection = median(frame_for_detection, footprint=disk(median_filter_radius_px))

        det = PipelineMLE2D(frame_for_detection[None, ...]).localize(
            threshold=threshold,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            fit_enabled=fit_enabled,
            show_tqdm=False,
        )
        if det.empty:
            continue
        if max_spots_per_frame > 0 and len(det) > max_spots_per_frame and "peak" in det.columns:
            det = det.sort_values("peak", ascending=False).head(max_spots_per_frame)

        if fit_enabled and "x_mle" in det.columns and "y_mle" in det.columns:
            coords = det[["x_mle", "y_mle"]].to_numpy(dtype=np.float32)
        else:
            coords = det[["x", "y"]].to_numpy(dtype=np.float32)
        if coords.size:
            all_coords.append(coords)

    if not all_coords:
        return np.empty((0, 2), dtype=np.float32)
    return np.concatenate(all_coords, axis=0).astype(np.float32)


def _roll_coords(coords: np.ndarray, shape: Tuple[int, int], axis0_shift: int, axis1_shift: int) -> np.ndarray:
    if coords.size == 0:
        return coords
    h, w = int(shape[0]), int(shape[1])
    out = coords.copy()
    out[:, 0] = np.mod(out[:, 0] + axis0_shift, h)
    out[:, 1] = np.mod(out[:, 1] + axis1_shift, w)
    return out


def _detections_to_napari_points(detections: List[np.ndarray]) -> np.ndarray:
    points: List[List[float]] = []
    for t_idx, coords in enumerate(detections):
        for x_val, y_val in coords:
            points.append([float(t_idx), float(x_val), float(y_val)])
    if not points:
        return np.empty((0, 3), dtype=np.float32)
    return np.array(points, dtype=np.float32)


def _run_napari_detection_preview(config: Dict[str, Any]) -> bool:
    paths_cfg = config["paths"]
    fig_cfg = config.get("figure_4cd", {})
    detect_cfg = fig_cfg.get("detection", {})

    preview_enabled = bool(detect_cfg.get("napari_preview", False))
    if not preview_enabled:
        return False

    try:
        import napari
    except Exception as exc:
        raise RuntimeError(
            "Napari preview requested, but napari is not available. Install with: pip install napari[all]"
        ) from exc

    path_hd = paths_cfg.get("hd_dir", paths_cfg.get("high_density_dir"))
    path_ls = paths_cfg.get("ls_dir", paths_cfg.get("long_sequence_dir"))
    if path_hd is None or path_ls is None:
        raise KeyError("paths.hd_dir and paths.ls_dir are required")
    path_hd_results = paths_cfg.get("hd_results_dir", paths_cfg.get("high_density_results_dir", path_hd))
    path_ls_results = paths_cfg.get("ls_results_dir", paths_cfg.get("long_sequence_results_dir", path_ls))

    threshold = float(detect_cfg.get("log_threshold", 0.1))
    min_sigma = float(detect_cfg.get("min_sigma", 0.75))
    max_sigma = float(detect_cfg.get("max_sigma", 1.5))
    max_spots_per_frame = int(detect_cfg.get("max_spots_per_frame", 0))
    median_filter_radius_px = int(detect_cfg.get("median_filter_radius_px", 0))
    preview_dataset = str(detect_cfg.get("napari_preview_dataset", "both")).lower()
    preview_only = bool(detect_cfg.get("napari_preview_only", False))

    datasets: List[Tuple[str, str]] = []
    if preview_dataset in ("hd", "both"):
        datasets.append(("hd", path_hd_results))
    if preview_dataset in ("ls", "both"):
        datasets.append(("ls", path_ls_results))
    if not datasets:
        raise ValueError("detection.napari_preview_dataset must be one of: hd, ls, both")

    for tag, results_dir in datasets:
        viewer = napari.Viewer(title=f"CVDM LoG Detections Preview ({tag})")
        try:
            viewer.dims.ndisplay = 2
        except Exception:
            pass

        frames = _load_z_frames(results_dir)
        _, detections = _build_cvdm_render_from_detections(
            frames=frames,
            threshold=threshold,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            kde_sigma=float(detect_cfg.get("kde_sigma", 1.0)),
            max_spots_per_frame=max_spots_per_frame,
            median_filter_radius_px=median_filter_radius_px,
        )
        points = _detections_to_napari_points(detections)
        viewer.add_image(frames, name=f"z_{tag}", colormap="gray")
        try:
            layer = viewer.add_points(points, name=f"detections_{tag}", size=1.2, symbol="disc")
        except TypeError:
            layer = viewer.add_points(points, name=f"detections_{tag}", size=1.2)

        # Napari style args vary by version; set whichever attributes exist.
        if hasattr(layer, "symbol"):
            try:
                layer.symbol = "disc"
            except Exception:
                pass
        if hasattr(layer, "opacity"):
            layer.opacity = 0.9
        if hasattr(layer, "edge_color"):
            layer.edge_color = "red"
        if hasattr(layer, "border_color"):
            layer.border_color = "red"
        if hasattr(layer, "face_color"):
            layer.face_color = "red"
        if hasattr(layer, "edge_width"):
            layer.edge_width = 0.3
        if hasattr(layer, "border_width"):
            layer.border_width = 0.3

        napari.run()
        try:
            viewer.close()
        except Exception:
            pass

    return preview_only


def _ensure_single_cvdm_render(
    tag: str,
    results_dir: str,
    output_dir: str,
    detect_cfg: Dict[str, Any],
) -> None:
    render_out = os.path.join(results_dir, "render-cvdm.tif")
    if os.path.exists(render_out) and not bool(detect_cfg.get("regenerate_cvdm_render", False)):
        return

    frames = _load_z_frames(results_dir)
    threshold = float(detect_cfg.get("log_threshold", 0.1))
    min_sigma = float(detect_cfg.get("min_sigma", 0.75))
    max_sigma = float(detect_cfg.get("max_sigma", 1.5))
    kde_sigma = float(detect_cfg.get("kde_sigma", 1.0))
    max_spots_per_frame = int(detect_cfg.get("max_spots_per_frame", 0))
    movie_stride = int(detect_cfg.get("movie_stride", 1))
    median_filter_radius_px = int(detect_cfg.get("median_filter_radius_px", 0))

    render, detections = _build_cvdm_render_from_detections(
        frames=frames,
        threshold=threshold,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        kde_sigma=kde_sigma,
        max_spots_per_frame=max_spots_per_frame,
        median_filter_radius_px=median_filter_radius_px,
    )

    os.makedirs(results_dir, exist_ok=True)
    tifffile.imwrite(render_out, render.astype(np.float32))

    movie_out = os.path.join(output_dir, f"log-detections-{tag}.gif")
    _save_detection_gif(
        gif_path=movie_out,
        frames=frames,
        detections=detections,
        threshold=threshold,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        stride=movie_stride,
    )


def _ensure_cvdm_renders(config: Dict[str, Any]) -> None:
    paths_cfg = config["paths"]
    fig_cfg = config.get("figure_4cd", {})
    detect_cfg = fig_cfg.get("detection", {})

    path_hd = paths_cfg.get("hd_dir", paths_cfg.get("high_density_dir"))
    path_ls = paths_cfg.get("ls_dir", paths_cfg.get("long_sequence_dir"))
    if path_hd is None or path_ls is None:
        raise KeyError("paths.hd_dir and paths.ls_dir are required")

    path_hd_results = paths_cfg.get("hd_results_dir", paths_cfg.get("high_density_results_dir", path_hd))
    path_ls_results = paths_cfg.get("ls_results_dir", paths_cfg.get("long_sequence_results_dir", path_ls))
    output_dir = paths_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    _ensure_single_cvdm_render("hd", path_hd_results, output_dir, detect_cfg)
    _ensure_single_cvdm_render("ls", path_ls_results, output_dir, detect_cfg)


def _validate_required_renders(config: Dict[str, Any]) -> None:
    """Fail before plotting if any required Figure 4cd render input is missing."""
    paths_cfg = config["paths"]

    path_hd = paths_cfg.get("hd_dir", paths_cfg.get("high_density_dir"))
    path_ls = paths_cfg.get("ls_dir", paths_cfg.get("long_sequence_dir"))
    if path_hd is None or path_ls is None:
        raise KeyError("paths.hd_dir and paths.ls_dir are required")

    path_hd_results = paths_cfg.get("hd_results_dir", paths_cfg.get("high_density_results_dir", path_hd))
    path_ls_results = paths_cfg.get("ls_results_dir", paths_cfg.get("long_sequence_results_dir", path_ls))

    _find_existing([
        os.path.join(path_ls_results, "eval", "render-cvdm.tif"),
        os.path.join(path_ls_results, "render-cvdm.tif"),
    ])
    _find_existing([
        os.path.join(path_ls, "thunderstorm", "render.tif"),
        os.path.join(path_ls, "render.tif"),
        os.path.join(path_ls_results, "thunderstorm", "render.tif"),
        os.path.join(path_ls_results, "render.tif"),
    ])
    _find_existing([
        os.path.join(path_ls, "thunderstorm-multi", "render.tif"),
        os.path.join(path_ls, "thunderstorm", "render.tif"),
        os.path.join(path_ls, "render.tif"),
        os.path.join(path_ls_results, "thunderstorm-multi", "render.tif"),
        os.path.join(path_ls_results, "thunderstorm", "render.tif"),
        os.path.join(path_ls_results, "render.tif"),
    ])
    _find_existing([
        os.path.join(path_hd_results, "eval", "render-cvdm.tif"),
        os.path.join(path_hd_results, "render-cvdm.tif"),
    ])
    _find_existing([
        os.path.join(path_hd, "thunderstorm", "render-crop.tif"),
        os.path.join(path_hd, "thunderstorm", "render.tif"),
        os.path.join(path_hd_results, "thunderstorm", "render-crop.tif"),
        os.path.join(path_hd_results, "thunderstorm", "render.tif"),
    ])


def run_figure_4b(config: Dict[str, Any]) -> None:
    paths_cfg = config["paths"]
    fig_cfg = config["figure_4b"]

    path_hd = paths_cfg.get("hd_dir", paths_cfg.get("high_density_dir"))
    path_ls = paths_cfg.get("ls_dir", paths_cfg.get("long_sequence_dir"))
    if path_hd is None or path_ls is None:
        raise KeyError("paths.hd_dir and paths.ls_dir are required")
    path_hd_results = paths_cfg.get("hd_results_dir", paths_cfg.get("high_density_results_dir", path_hd))
    path_ls_results = paths_cfg.get("ls_results_dir", paths_cfg.get("long_sequence_results_dir", path_ls))
    output_dir = paths_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    hd_idx = int(fig_cfg.get("hd_idx", 0))
    ls_idx = int(fig_cfg.get("ls_idx", 0))
    ls_sum_idx = int(fig_cfg.get("ls_sum_idx", 0))

    hd_1x = _read_frame(os.path.join(path_hd, "lr-1x-crop.tif"), hd_idx)
    ls_1x = _read_frame(os.path.join(path_ls, "lr-1x.tif"), ls_idx)
    ls_sum_1x = _read_frame(os.path.join(path_ls, "lr-1x-sum.tif"), ls_sum_idx)

    hd_4x_path = _find_existing([
        os.path.join(path_hd_results, "eval", f"z-{hd_idx}-0.tif"),
        os.path.join(path_hd_results, f"z-{hd_idx}-0.tif"),
    ])
    ls_4x_path = _find_existing([
        os.path.join(path_ls_results, "eval", f"z-{ls_idx}-0.tif"),
        os.path.join(path_ls_results, f"z-{ls_idx}-0.tif"),
    ])
    ls_sum_4x_path = _find_existing([
        os.path.join(path_ls_results, "eval", f"z-{ls_sum_idx}-0.tif"),
        os.path.join(path_ls_results, f"z-{ls_sum_idx}-0.tif"),
    ])
    hd_4x = imread(hd_4x_path).astype(np.float32)
    ls_4x = imread(ls_4x_path).astype(np.float32)
    ls_sum_4x = imread(ls_sum_4x_path).astype(np.float32)

    hd_4x[hd_4x < 0.0] = 0
    ls_sum_4x[ls_sum_4x < 0.0] = 0

    radius = float(fig_cfg.get("rolling_ball_radius", 5.0))
    hd_4x -= rolling_ball(hd_4x, radius=radius)
    ls_sum_4x -= rolling_ball(ls_sum_4x, radius=radius)

    trim_border_px = int(fig_cfg.get("trim_border_px", 5))
    if trim_border_px > 0:
        hd_4x[:trim_border_px, :] = 0
        ls_sum_4x[:trim_border_px, :] = 0
        hd_4x[:, :trim_border_px] = 0
        ls_sum_4x[:, :trim_border_px] = 0

    fig, ax = plt.subplots(2, 2, figsize=(5, 5))

    ax[0, 0].imshow(ls_sum_1x, cmap="gray", vmin=0.0)
    ax[0, 1].imshow(hd_1x, cmap="gray", vmin=0.0)
    ax[1, 0].imshow(ls_sum_4x, cmap="gray")
    ax[1, 1].imshow(hd_4x, cmap="gray")

    for axi in ax.ravel():
        axi.set_aspect(1.0)
        axi.set_xticks([])
        axi.set_yticks([])

    ax[0, 0].set_ylabel("$x$", fontsize=14, labelpad=10)
    ax[1, 0].set_ylabel("$\\hat{y}_{0}$", fontsize=14, labelpad=10)

    hr_inset_coords = fig_cfg.get("hr_inset_coords", [[40, 12], [8, 16]])
    lr_inset_coords = fig_cfg.get("lr_inset_coords", [[10, 3], [2, 4]])
    lr_inset_size = int(fig_cfg.get("lr_inset_size", 15))
    hr_inset_size = int(fig_cfg.get("hr_inset_size", 60))

    inset = ax[0, 0].inset_axes([0.65, 0.65, 0.4, 0.4])
    inset.imshow(_slice_2d(ls_sum_1x, lr_inset_coords[0], lr_inset_size), cmap="gray", interpolation="nearest")
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color("red")
        spine.set_linewidth(1)

    inset = ax[0, 1].inset_axes([0.65, 0.65, 0.4, 0.4])
    inset.imshow(_slice_2d(hd_1x, lr_inset_coords[1], lr_inset_size), cmap="gray", interpolation="nearest")
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color("red")
        spine.set_linewidth(1)

    inset = ax[1, 0].inset_axes([0.65, 0.65, 0.4, 0.4])
    inset.imshow(_slice_2d(ls_sum_4x, hr_inset_coords[0], hr_inset_size), cmap="gray", interpolation="nearest")
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color("red")
        spine.set_linewidth(1)

    inset = ax[1, 1].inset_axes([0.65, 0.65, 0.4, 0.4])
    inset.imshow(_slice_2d(hd_4x, hr_inset_coords[1], hr_inset_size), cmap="gray", interpolation="nearest")
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color("red")
        spine.set_linewidth(1)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.1, hspace=0.1)
    out_name = fig_cfg.get("output_name", "figure-10.png")
    out_path = os.path.join(output_dir, out_name)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def run_figure_4cd(config: Dict[str, Any]) -> None:
    paths_cfg = config["paths"]
    fig_cfg = config["figure_4cd"]

    path_hd = paths_cfg.get("hd_dir", paths_cfg.get("high_density_dir"))
    path_ls = paths_cfg.get("ls_dir", paths_cfg.get("long_sequence_dir"))
    if path_hd is None or path_ls is None:
        raise KeyError("paths.hd_dir and paths.ls_dir are required")
    path_hd_results = paths_cfg.get("hd_results_dir", paths_cfg.get("high_density_results_dir", path_hd))
    path_ls_results = paths_cfg.get("ls_results_dir", paths_cfg.get("long_sequence_results_dir", path_ls))
    path_hd_100it = paths_cfg.get("hd_100it_dir", None)
    path_ls_100it = paths_cfg.get("ls_100it_dir", None)
    output_dir = paths_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    summed_hd = imread(_find_existing([
        os.path.join(path_hd_results, "SUM_lr-1x.tif"),
        os.path.join(path_hd, "SUM_lr-1x.tif"),
    ]))
    summed_ls = imread(_find_existing([
        os.path.join(path_ls_results, "SUM_lr-1x.tif"),
        os.path.join(path_ls, "SUM_lr-1x.tif"),
    ]))

    ls_cvdm = imread(_find_existing([
        os.path.join(path_ls_results, "eval", "render-cvdm.tif"),
        os.path.join(path_ls_results, "render-cvdm.tif"),
    ])).astype(np.float32)
    ls_thunder = imread(_find_existing([
        os.path.join(path_ls, "thunderstorm", "render.tif"),
        os.path.join(path_ls, "render.tif"),
        os.path.join(path_ls_results, "thunderstorm", "render.tif"),
        os.path.join(path_ls_results, "render.tif"),
    ])).astype(np.float32)
    ls_thunder_multi = imread(_find_existing([
        os.path.join(path_ls, "thunderstorm-multi", "render.tif"),
        os.path.join(path_ls, "thunderstorm", "render.tif"),
        os.path.join(path_ls, "render.tif"),
        os.path.join(path_ls_results, "thunderstorm-multi", "render.tif"),
        os.path.join(path_ls_results, "thunderstorm", "render.tif"),
        os.path.join(path_ls_results, "render.tif"),
    ])).astype(np.float32)

    hd_cvdm = imread(_find_existing([
        os.path.join(path_hd_results, "eval", "render-cvdm.tif"),
        os.path.join(path_hd_results, "render-cvdm.tif"),
    ])).astype(np.float32)
    hd_thunder = imread(_find_existing([
        os.path.join(path_hd, "thunderstorm", "render-crop.tif"),
        os.path.join(path_hd, "thunderstorm", "render.tif"),
        os.path.join(path_hd_results, "thunderstorm", "render-crop.tif"),
        os.path.join(path_hd_results, "thunderstorm", "render.tif"),
    ])).astype(np.float32)

    ls_roll = fig_cfg.get("ls_cvdm_roll", [0, 1])
    detect_cfg = fig_cfg.get("detection", {})
    threshold = float(detect_cfg.get("log_threshold", 0.1))
    min_sigma = float(detect_cfg.get("min_sigma", 0.75))
    max_sigma = float(detect_cfg.get("max_sigma", 1.5))
    max_spots_per_frame = int(detect_cfg.get("max_spots_per_frame", 0))
    median_filter_radius_px = int(detect_cfg.get("median_filter_radius_px", 0))
    fit_enabled = bool(detect_cfg.get("fit_enabled", False))
    overlay_dot_size = float(fig_cfg.get("overlay_dot_size", 6.0))
    overlay_alpha = float(fig_cfg.get("overlay_alpha", 0.7))
    sum_100it_display_threshold = fig_cfg.get("sum_100it_display_threshold", None)
    if sum_100it_display_threshold is not None:
        sum_100it_display_threshold = float(sum_100it_display_threshold)

    ls_cvdm_100it: Optional[np.ndarray] = None
    hd_cvdm_100it: Optional[np.ndarray] = None
    ls_x_100it: Optional[np.ndarray] = None
    hd_x_100it: Optional[np.ndarray] = None
    ls_100it_src_shape: Optional[Tuple[int, int]] = None
    hd_100it_src_shape: Optional[Tuple[int, int]] = None
    ls_100it_coords = np.empty((0, 2), dtype=np.float32)
    hd_100it_coords = np.empty((0, 2), dtype=np.float32)
    ls_100it_coords_raw = np.empty((0, 2), dtype=np.float32)
    hd_100it_coords_raw = np.empty((0, 2), dtype=np.float32)

    if path_ls_100it:
        ls_100it_frames = _load_z_frames(path_ls_100it)
        ls_100it_src_shape = (int(ls_100it_frames.shape[1]), int(ls_100it_frames.shape[2]))
        ls_x_100it = _load_x_stack_image(path_ls_100it)
        ls_cvdm_100it = np.sum(ls_100it_frames, axis=0).astype(np.float32)
        ls_100it_coords = _aggregate_detection_coords(
            frames=ls_100it_frames,
            threshold=threshold,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            max_spots_per_frame=max_spots_per_frame,
            median_filter_radius_px=median_filter_radius_px,
            fit_enabled=fit_enabled,
        )
        ls_100it_coords_raw = ls_100it_coords.copy()

    if path_hd_100it:
        hd_100it_frames = _load_z_frames(path_hd_100it)
        hd_100it_src_shape = (int(hd_100it_frames.shape[1]), int(hd_100it_frames.shape[2]))
        hd_x_100it = _load_x_stack_image(path_hd_100it)
        hd_cvdm_100it = np.sum(hd_100it_frames, axis=0).astype(np.float32)
        hd_100it_coords = _aggregate_detection_coords(
            frames=hd_100it_frames,
            threshold=threshold,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            max_spots_per_frame=max_spots_per_frame,
            median_filter_radius_px=median_filter_radius_px,
            fit_enabled=fit_enabled,
        )
        hd_100it_coords_raw = hd_100it_coords.copy()

    ls_cvdm = np.roll(ls_cvdm, int(ls_roll[1]), axis=int(ls_roll[0]))
    if ls_cvdm_100it is not None:
        ls_cvdm_100it = np.roll(ls_cvdm_100it, int(ls_roll[1]), axis=int(ls_roll[0]))
        if ls_100it_coords.size:
            if int(ls_roll[0]) == 0:
                ls_100it_coords = _roll_coords(ls_100it_coords, ls_cvdm_100it.shape, int(ls_roll[1]), 0)
            else:
                ls_100it_coords = _roll_coords(ls_100it_coords, ls_cvdm_100it.shape, 0, int(ls_roll[1]))

    hd_roll_axis0 = int(fig_cfg.get("hd_cvdm_roll_axis0", 5))
    hd_roll_axis1 = int(fig_cfg.get("hd_cvdm_roll_axis1", 4))
    hd_cvdm = np.roll(hd_cvdm, hd_roll_axis0, axis=0)
    hd_cvdm = np.roll(hd_cvdm, hd_roll_axis1, axis=1)
    if hd_cvdm_100it is not None:
        hd_cvdm_100it = np.roll(hd_cvdm_100it, hd_roll_axis0, axis=0)
        hd_cvdm_100it = np.roll(hd_cvdm_100it, hd_roll_axis1, axis=1)
        if hd_100it_coords.size:
            hd_100it_coords = _roll_coords(hd_100it_coords, hd_cvdm_100it.shape, hd_roll_axis0, hd_roll_axis1)

    ls_thunder_vmax = float(fig_cfg.get("ls_thunder_vmax", 40.0))
    ls_thunder_multi_vmax = float(fig_cfg.get("ls_thunder_multi_vmax", 30.0))

    fig, ax = plt.subplots(1, 4, figsize=(10, 4))
    ax[0].imshow(summed_ls, cmap="gray", vmin=0.0)
    ax[1].imshow(ls_thunder, cmap="gray", vmin=0.0, vmax=ls_thunder_vmax)
    ax[2].imshow(ls_thunder_multi, cmap="gray", vmin=0.0, vmax=ls_thunder_multi_vmax)
    ax[3].imshow(ls_cvdm, cmap="gray", vmin=0.0)
    for axi in ax.ravel():
        axi.set_xticks([])
        axi.set_yticks([])

    panel_ls_name = fig_cfg.get("output_panel_ls", "figure-11-1-1.png")
    plt.savefig(os.path.join(output_dir, panel_ls_name), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    ax[0].imshow(summed_hd, cmap="gray", vmin=0.0)
    ax[1].imshow(hd_thunder, cmap="gray", vmin=0.0)
    ax[2].imshow(hd_cvdm, cmap="gray")
    for axi in ax.ravel():
        axi.set_xticks([])
        axi.set_yticks([])

    panel_hd_name = fig_cfg.get("output_panel_hd", "figure-11-1-2.png")
    plt.savefig(os.path.join(output_dir, panel_hd_name), dpi=200)
    plt.close(fig)

    fig_line, ax_line = plt.subplots(2, 1, figsize=(10, 5))

    pixel_size = float(fig_cfg.get("pixel_size_nm", 25.0))
    profile_offset = int(fig_cfg.get("profile_offset", 40))

    ls_row = int(fig_cfg.get("ls_profile_row", 120))
    ls_start = int(fig_cfg.get("ls_profile_start", 50))
    ls_stop = int(fig_cfg.get("ls_profile_stop", 80))
    x_ls = np.arange(ls_start, ls_stop)

    y_ls_cvdm = _profile_norm(ls_cvdm[ls_row], ls_start, ls_stop)
    y_ls_thunder = _profile_norm(ls_thunder[ls_row], ls_start, ls_stop)
    y_ls_thunder_multi = _profile_norm(ls_thunder_multi[ls_row], ls_start, ls_stop)

    x_ls = x_ls - profile_offset
    x_ls_nm = (x_ls - x_ls[0]) * pixel_size

    ax_line[0].plot(x_ls_nm, y_ls_cvdm, "r-", marker="o", label="CVDM (LS-SUM)")
    ax_line[0].plot(x_ls_nm, y_ls_thunder, "b-", marker="o", label="ThunderSTORM (LS)")
    ax_line[0].plot(x_ls_nm, y_ls_thunder_multi, color="cyan", marker="o", label="ThunderSTORM (LS-SUM)")
    ax_line[0].legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.6, 1.5), ncol=3, frameon=False)

    ax_line[0].set_xlabel("Distance (nm)", fontsize=12)
    ax_line[0].set_ylabel("Intensity (a.u.)", fontsize=12)
    ax_line[0].spines["top"].set_visible(False)
    ax_line[0].spines["right"].set_visible(False)

    hd_row = int(fig_cfg.get("hd_profile_row", 182))
    hd_start = int(fig_cfg.get("hd_profile_start", 83))
    hd_stop = int(fig_cfg.get("hd_profile_stop", 113))
    x_hd = np.arange(hd_start, hd_stop)

    y_hd_cvdm = _profile_norm(hd_cvdm[hd_row], hd_start, hd_stop)
    y_hd_thunder = _profile_norm(hd_thunder[hd_row], hd_start, hd_stop)

    x_hd = x_hd - profile_offset
    x_hd_nm = (x_hd - x_hd[0]) * pixel_size

    ax_line[1].plot(x_hd_nm, y_hd_cvdm, "r-", marker="o", label="CVDM (HD)")
    ax_line[1].plot(x_hd_nm, y_hd_thunder, "b-", marker="o", label="ThunderSTORM (HD)")
    ax_line[1].legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, 1.5), ncol=3, frameon=False)

    ax_line[1].set_xlabel("Distance (nm)", fontsize=12)
    ax_line[1].set_ylabel("Intensity (a.u.)", fontsize=12)
    ax_line[1].spines["top"].set_visible(False)
    ax_line[1].spines["right"].set_visible(False)

    plt.tight_layout()
    out_line = fig_cfg.get("output_line_name", "figure-4c.png")
    plt.savefig(os.path.join(output_dir, out_line), dpi=300)
    plt.close(fig_line)

    row_start = int(fig_cfg.get("hd_crop_row_start", 162))
    row_stop = int(fig_cfg.get("hd_crop_row_stop", 212))
    col_start = int(fig_cfg.get("hd_crop_col_start", 73))
    col_stop = int(fig_cfg.get("hd_crop_col_stop", 123))

    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].imshow(hd_thunder[row_start:row_stop, col_start:col_stop], cmap="gray")
    ax[1].imshow(hd_cvdm[row_start:row_stop, col_start:col_stop], cmap="gray")
    for axi in ax.ravel():
        axi.set_xticks([])
        axi.set_yticks([])

    out_crop = fig_cfg.get("output_crop_name", "figure-4d.png")
    plt.savefig(os.path.join(output_dir, out_crop), dpi=300)
    plt.close(fig)

    # Additional standalone outputs from 100-iteration stacks (if provided)
    if ls_x_100it is not None:
        ls_display = ls_x_100it.copy()
        if sum_100it_display_threshold is not None:
            ls_display[ls_display < sum_100it_display_threshold] = 0.0
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        _imshow_percentile_gray(ax, ls_display)
        if ls_100it_coords_raw.size and ls_100it_src_shape is not None:
            ls_plot_coords = _rescale_coords(ls_100it_coords_raw, ls_100it_src_shape, ls_display.shape)
            ax.scatter(ls_plot_coords[:, 1], ls_plot_coords[:, 0], c="red", s=overlay_dot_size, alpha=overlay_alpha)
        ax.set_xticks([])
        ax.set_yticks([])
        out_ls_100it = fig_cfg.get("output_ls_100it_scatter", "figure-100it-ls-scatter.png")
        plt.savefig(os.path.join(output_dir, out_ls_100it), dpi=300)
        plt.close(fig)

    if hd_x_100it is not None:
        hd_display = hd_x_100it.copy()
        if sum_100it_display_threshold is not None:
            hd_display[hd_display < sum_100it_display_threshold] = 0.0
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        _imshow_percentile_gray(ax, hd_display)
        if hd_100it_coords_raw.size and hd_100it_src_shape is not None:
            hd_plot_coords = _rescale_coords(hd_100it_coords_raw, hd_100it_src_shape, hd_display.shape)
            ax.scatter(hd_plot_coords[:, 1], hd_plot_coords[:, 0], c="red", s=overlay_dot_size, alpha=overlay_alpha)
        ax.set_xticks([])
        ax.set_yticks([])
        out_hd_100it = fig_cfg.get("output_hd_100it_scatter", "figure-100it-hd-scatter.png")
        plt.savefig(os.path.join(output_dir, out_hd_100it), dpi=300)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tube summary plots (Figure 4b, 4c, 4d).")
    parser.add_argument("--config", required=True, type=str, help="Path to plot_tubes_summary YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    if _run_napari_detection_preview(config):
        return
    _ensure_cvdm_renders(config)
    _validate_required_renders(config)
    run_figure_4b(config)
    run_figure_4cd(config)


if __name__ == "__main__":
    main()
