import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def _load_cvdm_localizations_nm(
    csv_path: str,
    lr_pixel_size_nm: float,
    upsample_factor: int,
    frame_column: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    df = pd.read_csv(csv_path)
    if "x" not in df.columns or "y" not in df.columns:
        raise ValueError(f"Expected columns 'x' and 'y' in CVDM csv: {csv_path}")
    xy = df[["x", "y"]].to_numpy(dtype=np.float32)
    nm_per_px = float(lr_pixel_size_nm) / float(max(1, upsample_factor))
    points_nm = xy * nm_per_px

    frames: Optional[np.ndarray] = None
    if frame_column and frame_column in df.columns:
        frames = df[frame_column].to_numpy(dtype=np.int64)
    return points_nm, frames


def _load_thunderstorm_localizations_nm(
    csv_path: str,
    frame_column: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    df = pd.read_csv(csv_path)
    x_col = "x [nm]" if "x [nm]" in df.columns else "x"
    y_col = "y [nm]" if "y [nm]" in df.columns else "y"
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Expected Thunderstorm columns 'x [nm]'/'y [nm]' or 'x'/'y' in: {csv_path}")
    points_nm = df[[x_col, y_col]].to_numpy(dtype=np.float32)

    frames: Optional[np.ndarray] = None
    if frame_column and frame_column in df.columns:
        frames = df[frame_column].to_numpy(dtype=np.int64)
    return points_nm, frames


def _split_localizations(
    points_nm: np.ndarray,
    split_mode: str = "random",
    seed: int = 0,
    frames: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if points_nm.size == 0:
        return points_nm.copy(), points_nm.copy()

    split_mode = str(split_mode).lower()
    n_points = int(points_nm.shape[0])

    if split_mode == "odd_even" and frames is not None and len(frames) == n_points:
        mask_even = (frames % 2) == 0
        a = points_nm[mask_even]
        b = points_nm[~mask_even]
        if a.shape[0] > 0 and b.shape[0] > 0:
            return a, b

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_points)
    mid = n_points // 2
    a_idx = perm[:mid]
    b_idx = perm[mid:]
    if a_idx.size == 0 or b_idx.size == 0:
        return points_nm.copy(), points_nm.copy()
    return points_nm[a_idx], points_nm[b_idx]


def _subsample_localizations(
    points_nm: np.ndarray,
    frames: Optional[np.ndarray],
    n_target: int,
    seed: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    n_total = int(points_nm.shape[0])
    if n_target <= 0 or n_total <= n_target:
        return points_nm, frames
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=n_target, replace=False)
    sampled_points = points_nm[idx]
    sampled_frames = frames[idx] if frames is not None and len(frames) == n_total else frames
    return sampled_points, sampled_frames


def _frc_crossing(freqs: np.ndarray, frc_vals: np.ndarray, threshold: float) -> Tuple[float, float]:
    crossing_freq = np.nan
    for idx in range(1, len(freqs)):
        y0 = float(frc_vals[idx - 1])
        y1 = float(frc_vals[idx])
        if (y0 >= threshold and y1 < threshold) or (y0 <= threshold and y1 > threshold):
            x0 = float(freqs[idx - 1])
            x1 = float(freqs[idx])
            if x1 != x0:
                t = (threshold - y0) / (y1 - y0)
                crossing_freq = x0 + t * (x1 - x0)
            else:
                crossing_freq = x0
            break
    resolution_nm = float(1.0 / crossing_freq) if np.isfinite(crossing_freq) and crossing_freq > 0 else float("nan")
    return crossing_freq, resolution_nm


def _rasterize_points_nm(points_nm: np.ndarray, image_size_nm: float, bin_size_nm: float) -> np.ndarray:
    n_bins = int(np.ceil(float(image_size_nm) / float(bin_size_nm)))
    n_bins = max(n_bins, 8)
    edges = np.linspace(0.0, image_size_nm, n_bins + 1, dtype=np.float32)
    if points_nm.size == 0:
        return np.zeros((n_bins, n_bins), dtype=np.float32)
    x = points_nm[:, 0]
    y = points_nm[:, 1]
    valid = (x >= 0.0) & (x <= image_size_nm) & (y >= 0.0) & (y <= image_size_nm)
    x = x[valid]
    y = y[valid]
    hist, _, _ = np.histogram2d(x, y, bins=[edges, edges])
    return hist.astype(np.float32)


def _fourier_ring_correlation(image_a: np.ndarray, image_b: np.ndarray, bin_size_nm: float) -> Tuple[np.ndarray, np.ndarray]:
    if image_a.shape != image_b.shape:
        raise ValueError(f"FRC images must have identical shape, got {image_a.shape} vs {image_b.shape}")

    fa = np.fft.fftshift(np.fft.fft2(image_a))
    fb = np.fft.fftshift(np.fft.fft2(image_b))

    h, w = image_a.shape
    yy, xx = np.indices((h, w))
    cy, cx = h // 2, w // 2
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rr_int = rr.astype(np.int32)

    max_r = int(min(h, w) // 2)
    freqs = []
    frc_vals = []

    for radius in range(1, max_r):
        mask = rr_int == radius
        if not np.any(mask):
            continue
        a_ring = fa[mask]
        b_ring = fb[mask]
        num = np.sum(a_ring * np.conj(b_ring))
        den = np.sqrt(np.sum(np.abs(a_ring) ** 2) * np.sum(np.abs(b_ring) ** 2))
        if den == 0:
            continue
        frc = float(np.real(num / den))
        freq = float(radius) / (float(h) * float(bin_size_nm))
        freqs.append(freq)
        frc_vals.append(frc)

    return np.array(freqs, dtype=np.float32), np.array(frc_vals, dtype=np.float32)


def run_figure_4frc(config: Dict[str, Any]) -> None:
    fig_cfg = config.get("figure_4frc", None)
    if not fig_cfg:
        return

    paths_cfg = config["paths"]
    output_dir = paths_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    thunder_spots_csv = fig_cfg.get("thunder_spots_csv", None)
    if thunder_spots_csv is None:
        raise KeyError("figure_4frc requires 'thunder_spots_csv'")

    cvdm_results_dir = fig_cfg.get(
        "cvdm_results_dir",
        paths_cfg.get("hd_results_dir", paths_cfg.get("high_density_results_dir", None)),
    )
    if cvdm_results_dir is None:
        raise KeyError("figure_4frc requires 'cvdm_results_dir' or paths.hd_results_dir")

    lr_pixel_size_nm = float(fig_cfg.get("pixel_size_nm", 80.0))
    upsample_factor = int(fig_cfg.get("upsample_factor", 4))
    image_size_lr_px = int(fig_cfg.get("image_size_lr_px", 64))
    image_size_nm = float(image_size_lr_px) * lr_pixel_size_nm
    bin_size_nm = float(fig_cfg.get("frc_bin_size_nm", lr_pixel_size_nm / max(1, upsample_factor)))
    split_mode = str(fig_cfg.get("split_mode", "random")).lower()
    split_seed = int(fig_cfg.get("split_seed", 0))
    thunder_frame_column = fig_cfg.get("thunder_frame_column", None)

    detect_cfg = fig_cfg.get("detection", config.get("figure_4cd", {}).get("detection", {}))
    det_threshold = float(detect_cfg.get("log_threshold", 0.1))
    det_min_sigma = float(detect_cfg.get("min_sigma", 0.75))
    det_max_sigma = float(detect_cfg.get("max_sigma", 1.5))
    det_max_spots_per_frame = int(detect_cfg.get("max_spots_per_frame", 0))
    det_median_filter_radius_px = int(detect_cfg.get("median_filter_radius_px", 0))
    det_fit_enabled = bool(detect_cfg.get("fit_enabled", True))

    cvdm_frames_stack = _load_z_frames(cvdm_results_dir)
    cvdm_px, cvdm_frames = _aggregate_detection_coords_with_frames(
        frames=cvdm_frames_stack,
        threshold=det_threshold,
        min_sigma=det_min_sigma,
        max_sigma=det_max_sigma,
        max_spots_per_frame=det_max_spots_per_frame,
        median_filter_radius_px=det_median_filter_radius_px,
        fit_enabled=det_fit_enabled,
    )
    cvdm_nm_per_px = float(lr_pixel_size_nm) / float(max(1, upsample_factor))
    cvdm_nm = cvdm_px * cvdm_nm_per_px

    thunder_nm, thunder_frames = _load_thunderstorm_localizations_nm(
        thunder_spots_csv,
        frame_column=thunder_frame_column,
    )

    print(f"[figure_4frc] CVDM localizations total: {int(cvdm_nm.shape[0])}")
    print(f"[figure_4frc] ThunderSTORM localizations total: {int(thunder_nm.shape[0])}")

    count_match_total = fig_cfg.get("count_match_total_localizations", None)
    if count_match_total is not None:
        count_match_total = int(count_match_total)
        if count_match_total > 0:
            n_match = min(count_match_total, int(cvdm_nm.shape[0]), int(thunder_nm.shape[0]))
            cvdm_nm, cvdm_frames = _subsample_localizations(cvdm_nm, cvdm_frames, n_match, split_seed + 101)
            thunder_nm, thunder_frames = _subsample_localizations(thunder_nm, thunder_frames, n_match, split_seed + 202)
            print(
                "[figure_4frc] Count-matched localizations: "
                f"target={count_match_total}, used_per_method={n_match}"
            )
            print(
                "[figure_4frc] After count-match totals: "
                f"CVDM={int(cvdm_nm.shape[0])}, ThunderSTORM={int(thunder_nm.shape[0])}"
            )

    cvdm_a, cvdm_b = _split_localizations(
        cvdm_nm,
        split_mode=split_mode,
        seed=split_seed,
        frames=cvdm_frames,
    )
    thunder_a, thunder_b = _split_localizations(
        thunder_nm,
        split_mode=split_mode,
        seed=split_seed + 1,
        frames=thunder_frames,
    )

    img_cvdm_a = _rasterize_points_nm(cvdm_a, image_size_nm=image_size_nm, bin_size_nm=bin_size_nm)
    img_cvdm_b = _rasterize_points_nm(cvdm_b, image_size_nm=image_size_nm, bin_size_nm=bin_size_nm)
    img_thunder_a = _rasterize_points_nm(thunder_a, image_size_nm=image_size_nm, bin_size_nm=bin_size_nm)
    img_thunder_b = _rasterize_points_nm(thunder_b, image_size_nm=image_size_nm, bin_size_nm=bin_size_nm)

    out_cvdm_a_tif = fig_cfg.get("output_frc_cvdm_split_a_tif", "figure-4e-frc-cvdm-split-a.tif")
    out_cvdm_b_tif = fig_cfg.get("output_frc_cvdm_split_b_tif", "figure-4e-frc-cvdm-split-b.tif")
    out_thunder_a_tif = fig_cfg.get("output_frc_thunder_split_a_tif", "figure-4e-frc-thunderstorm-split-a.tif")
    out_thunder_b_tif = fig_cfg.get("output_frc_thunder_split_b_tif", "figure-4e-frc-thunderstorm-split-b.tif")
    tifffile.imwrite(os.path.join(output_dir, out_cvdm_a_tif), img_cvdm_a.astype(np.float32))
    tifffile.imwrite(os.path.join(output_dir, out_cvdm_b_tif), img_cvdm_b.astype(np.float32))
    tifffile.imwrite(os.path.join(output_dir, out_thunder_a_tif), img_thunder_a.astype(np.float32))
    tifffile.imwrite(os.path.join(output_dir, out_thunder_b_tif), img_thunder_b.astype(np.float32))

    vmax_all = float(
        max(
            np.max(img_cvdm_a) if img_cvdm_a.size else 0.0,
            np.max(img_cvdm_b) if img_cvdm_b.size else 0.0,
            np.max(img_thunder_a) if img_thunder_a.size else 0.0,
            np.max(img_thunder_b) if img_thunder_b.size else 0.0,
        )
    )
    if vmax_all <= 0.0:
        vmax_all = 1.0

    fig_inputs, ax_inputs = plt.subplots(2, 2, figsize=(6, 6))
    ax_inputs[0, 0].imshow(img_cvdm_a, cmap="gray", vmin=0.0, vmax=vmax_all)
    ax_inputs[0, 0].set_title("CVDM split A", fontsize=9)
    ax_inputs[0, 1].imshow(img_cvdm_b, cmap="gray", vmin=0.0, vmax=vmax_all)
    ax_inputs[0, 1].set_title("CVDM split B", fontsize=9)
    ax_inputs[1, 0].imshow(img_thunder_a, cmap="gray", vmin=0.0, vmax=vmax_all)
    ax_inputs[1, 0].set_title("ThunderSTORM split A", fontsize=9)
    ax_inputs[1, 1].imshow(img_thunder_b, cmap="gray", vmin=0.0, vmax=vmax_all)
    ax_inputs[1, 1].set_title("ThunderSTORM split B", fontsize=9)
    for axi in ax_inputs.ravel():
        axi.set_xticks([])
        axi.set_yticks([])
    fig_inputs.tight_layout()
    out_inputs_plot = fig_cfg.get("output_frc_inputs_plot", "figure-4e-frc-inputs.png")
    fig_inputs.savefig(os.path.join(output_dir, out_inputs_plot), dpi=300)
    plt.close(fig_inputs)

    freqs_cvdm, frc_cvdm = _fourier_ring_correlation(img_cvdm_a, img_cvdm_b, bin_size_nm=bin_size_nm)
    freqs_thunder, frc_thunder = _fourier_ring_correlation(img_thunder_a, img_thunder_b, bin_size_nm=bin_size_nm)

    threshold = float(fig_cfg.get("frc_threshold", 1.0 / 7.0))
    crossing_freq_cvdm, resolution_nm_cvdm = _frc_crossing(freqs_cvdm, frc_cvdm, threshold)
    crossing_freq_thunder, resolution_nm_thunder = _frc_crossing(freqs_thunder, frc_thunder, threshold)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.plot(freqs_cvdm, frc_cvdm, color="red", linewidth=1.5, label="CVDM")
    ax.plot(freqs_thunder, frc_thunder, color="blue", linewidth=1.5, label="ThunderSTORM")
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1.0, label=f"threshold={threshold:.3f}")
    ax.set_xlabel(r"Spatial frequency (nm$^{-1}$)")
    ax.set_ylabel("FRC")
    ax.set_title("FRC")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()

    out_plot = fig_cfg.get("output_frc_plot", "figure-4e-frc.png")
    fig.savefig(os.path.join(output_dir, out_plot), dpi=300)
    plt.close(fig)

    freq_union = np.unique(np.concatenate([freqs_cvdm, freqs_thunder])).astype(np.float32)
    frc_cvdm_interp = np.interp(freq_union, freqs_cvdm, frc_cvdm, left=np.nan, right=np.nan)
    frc_thunder_interp = np.interp(freq_union, freqs_thunder, frc_thunder, left=np.nan, right=np.nan)
    df_out = pd.DataFrame(
        {
            "spatial_frequency_nm_inv": freq_union,
            "frc_cvdm": frc_cvdm_interp,
            "frc_thunderstorm": frc_thunder_interp,
        }
    )
    out_csv = fig_cfg.get("output_frc_csv", "figure-4e-frc.csv")
    df_out.to_csv(os.path.join(output_dir, out_csv), index=False)

    summary = {
        "threshold": threshold,
        "split_mode": split_mode,
        "split_seed": split_seed,
        "count_match_total_localizations": count_match_total,
        "cvdm_results_dir": cvdm_results_dir,
        "frc_input_images": {
            "cvdm_split_a_tif": out_cvdm_a_tif,
            "cvdm_split_b_tif": out_cvdm_b_tif,
            "thunderstorm_split_a_tif": out_thunder_a_tif,
            "thunderstorm_split_b_tif": out_thunder_b_tif,
            "panel_png": out_inputs_plot,
        },
        "cvdm": {
            "n_localizations": int(cvdm_nm.shape[0]),
            "n_split_a": int(cvdm_a.shape[0]),
            "n_split_b": int(cvdm_b.shape[0]),
            "crossing_frequency_nm_inv": None if not np.isfinite(crossing_freq_cvdm) else float(crossing_freq_cvdm),
            "resolution_nm": None if not np.isfinite(resolution_nm_cvdm) else float(resolution_nm_cvdm),
        },
        "thunderstorm": {
            "n_localizations": int(thunder_nm.shape[0]),
            "n_split_a": int(thunder_a.shape[0]),
            "n_split_b": int(thunder_b.shape[0]),
            "crossing_frequency_nm_inv": None if not np.isfinite(crossing_freq_thunder) else float(crossing_freq_thunder),
            "resolution_nm": None if not np.isfinite(resolution_nm_thunder) else float(resolution_nm_thunder),
        },
        "pixel_size_nm": lr_pixel_size_nm,
        "upsample_factor": upsample_factor,
        "cvdm_nm_per_px": cvdm_nm_per_px,
        "frc_bin_size_nm": bin_size_nm,
    }
    out_summary = fig_cfg.get("output_frc_summary", "figure-4e-frc-summary.yaml")
    with open(os.path.join(output_dir, out_summary), "w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)


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


def _aggregate_detection_coords_with_frames(
    frames: np.ndarray,
    threshold: float,
    min_sigma: float,
    max_sigma: float,
    max_spots_per_frame: int,
    median_filter_radius_px: int,
    fit_enabled: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    all_coords: List[np.ndarray] = []
    all_frame_ids: List[np.ndarray] = []
    for frame_idx, frame in enumerate(frames):
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
        if coords.size == 0:
            continue

        all_coords.append(coords)
        all_frame_ids.append(np.full((coords.shape[0],), frame_idx, dtype=np.int64))

    if not all_coords:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.concatenate(all_coords, axis=0).astype(np.float32), np.concatenate(all_frame_ids, axis=0)


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
    run_figure_4frc(config)


if __name__ == "__main__":
    main()
