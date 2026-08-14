import argparse
import math
import os
import re
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import yaml
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.ticker import MaxNLocator
from skimage.measure import profile_line
from tifffile import imread as tiff_read
from tifffile import imwrite

from cvdm.psf.mle2d import PipelineMLE2D


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _imshow_scaled(ax, img: np.ndarray, low: float, high: float, **kwargs) -> None:
    vmin = np.percentile(img, low)
    vmax = np.percentile(img, high)
    ax.imshow(img, vmin=vmin, vmax=vmax, cmap="gray", **kwargs)


def _draw_inset_box(ax, x0: int, y0: int, size: int) -> None:
    ax.add_patch(
        plt.Rectangle((y0, x0), size, size, linewidth=1, edgecolor="red", facecolor="none")
    )


def _detect_nanorulers(
    image: np.ndarray,
    expected_distance_px: float,
    distance_tol_px: float,
    log_threshold: float,
    min_sigma: float,
    max_sigma: float,
    min_distance: int = 3,
    require_exact_two: bool = True,
) -> tuple[list[tuple[tuple[float, float], tuple[float, float]]], list[tuple[float, float]]]:
    _ = min_distance
    detector = PipelineMLE2D(image[None, ...])
    spots = detector.localize(
        threshold=log_threshold,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        fit_enabled=False,
        show_tqdm=False,
    )
    if spots.empty:
        return [], []
    peaks = spots[["x", "y"]].to_numpy(dtype=float)
    pairs: list[tuple[tuple[float, float], tuple[float, float]]] = []
    used = np.zeros(len(peaks), dtype=bool)
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            p1 = peaks[i]
            p2 = peaks[j]
            dist = float(np.linalg.norm(p1 - p2))
            if abs(dist - expected_distance_px) > distance_tol_px:
                continue
            if require_exact_two:
                mid = (p1 + p2) / 2.0
                radius = expected_distance_px / 2.0 + distance_tol_px
                in_window = np.linalg.norm(peaks - mid, axis=1) <= radius
                if int(np.sum(in_window)) != 2:
                    continue
            pairs.append(((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))))
            used[i] = True
            used[j] = True
    rejected = [(float(p[0]), float(p[1])) for idx, p in enumerate(peaks) if not used[idx]]
    return pairs, rejected


def _cluster_points(points: np.ndarray, radius: float) -> list[list[int]]:
    if points.size == 0:
        return []
    n = len(points)
    visited = np.zeros(n, dtype=bool)
    clusters: list[list[int]] = []
    for idx in range(n):
        if visited[idx]:
            continue
        queue = [idx]
        visited[idx] = True
        cluster = [idx]
        while queue:
            current = queue.pop()
            dists = np.linalg.norm(points - points[current], axis=1)
            neighbors = np.where(dists <= radius)[0]
            for nb in neighbors:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(int(nb))
                    cluster.append(int(nb))
        clusters.append(cluster)
    return clusters


def _kmeans_points(points: np.ndarray, k: int, n_iter: int = 20) -> np.ndarray:
    if points.size == 0 or k <= 1:
        return np.zeros(len(points), dtype=int)
    n = len(points)
    k = min(k, n)
    rng = np.random.default_rng(0)
    centroids = points[rng.choice(n, size=k, replace=False)]
    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        dists = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for idx in range(k):
            mask = labels == idx
            if np.any(mask):
                centroids[idx] = points[mask].mean(axis=0)
    return labels


def _gmm_fit(points: np.ndarray, k: int, n_iter: int = 30) -> tuple[np.ndarray, np.ndarray, float]:
    if points.size == 0:
        return np.empty((0, 2)), np.empty((0, 2)), float("-inf")
    n = len(points)
    k = max(1, min(k, n))
    labels = _kmeans_points(points, k)
    means = np.zeros((k, 2), dtype=float)
    covs = np.zeros((k, 2), dtype=float)
    weights = np.ones(k, dtype=float) / k
    for idx in range(k):
        mask = labels == idx
        if np.any(mask):
            means[idx] = points[mask].mean(axis=0)
            covs[idx] = points[mask].var(axis=0) + 1e-3
        else:
            means[idx] = points[np.random.randint(0, n)]
            covs[idx] = np.array([1.0, 1.0])

    for _ in range(n_iter):
        resp = np.zeros((n, k), dtype=float)
        for idx in range(k):
            diff = points - means[idx]
            var = covs[idx]
            log_det = np.log(var[0]) + np.log(var[1])
            log_prob = -0.5 * (
                (diff[:, 0] ** 2) / var[0] + (diff[:, 1] ** 2) / var[1] + log_det
            )
            resp[:, idx] = np.log(weights[idx] + 1e-8) + log_prob
        resp = resp - resp.max(axis=1, keepdims=True)
        resp = np.exp(resp)
        resp_sum = resp.sum(axis=1, keepdims=True) + 1e-8
        resp = resp / resp_sum

        nk = resp.sum(axis=0) + 1e-8
        weights = nk / float(n)
        means = (resp.T @ points) / nk[:, None]
        for idx in range(k):
            diff = points - means[idx]
            covs[idx] = (resp[:, idx][:, None] * diff ** 2).sum(axis=0) / nk[idx] + 1e-3

    log_likelihood = float(np.sum(np.log(resp_sum)))
    return means, covs, log_likelihood


def _draw_nanoruler_marks(
    ax,
    pairs: list[tuple[tuple[float, float], tuple[float, float]]],
    rejected: list[tuple[float, float]],
    dot_diameter_px: float,
    line_width_px: float,
    dot_alpha: float,
    line_alpha: float,
) -> None:
    radius = max(dot_diameter_px / 2.0, 0.1)
    for (x1, y1), (x2, y2) in pairs:
        ax.plot([y1, y2], [x1, x2], color="blue", linewidth=line_width_px, alpha=line_alpha)
        ax.add_patch(patches.Circle((y1, x1), radius=radius, color="red", alpha=dot_alpha))
        ax.add_patch(patches.Circle((y2, x2), radius=radius, color="red", alpha=dot_alpha))
    if rejected:
        ys = [p[1] for p in rejected]
        xs = [p[0] for p in rejected]
        ax.scatter(ys, xs, marker="x", color="red", s=max(6.0, dot_diameter_px * 10.0), alpha=dot_alpha)


def _extract_patch(image: np.ndarray, x_center: float, y_center: float, size: int) -> np.ndarray:
    half = size // 2
    x0 = int(round(x_center)) - half
    y0 = int(round(y_center)) - half
    x1 = x0 + size
    y1 = y0 + size
    patch = np.zeros((size, size), dtype=image.dtype)
    src_x0 = max(x0, 0)
    src_y0 = max(y0, 0)
    src_x1 = min(x1, image.shape[0])
    src_y1 = min(y1, image.shape[1])
    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    if src_x1 > src_x0 and src_y1 > src_y0:
        patch[dst_x0:dst_x1, dst_y0:dst_y1] = image[src_x0:src_x1, src_y0:src_y1]
    return patch


def _pick_center_inset_coords(
    image: np.ndarray,
    inset_size: int,
    threshold_percentile: float,
) -> Tuple[int, int]:
    h, w = image.shape[:2]
    center = np.array([h / 2.0, w / 2.0])
    thresh = np.percentile(image, threshold_percentile)
    candidates = np.argwhere(image >= thresh)
    if candidates.size == 0:
        x0 = max(int(round(center[0] - inset_size / 2)), 0)
        y0 = max(int(round(center[1] - inset_size / 2)), 0)
        return x0, y0
    d2 = np.sum((candidates - center) ** 2, axis=1)
    idx = int(np.argmin(d2))
    x_val, y_val = candidates[idx]
    x0 = int(round(x_val - inset_size / 2))
    y0 = int(round(y_val - inset_size / 2))
    x0 = max(min(x0, h - inset_size), 0)
    y0 = max(min(y0, w - inset_size), 0)
    return x0, y0


def _ensure_probe_dirs(fig_root: str, probes: List[str]) -> Dict[str, str]:
    probe_dirs = {
        "2a": os.path.join(fig_root, "probe_2a"),
        "3a": os.path.join(fig_root, "probe_3a"),
        "3b": os.path.join(fig_root, "probe_3b"),
        "3b_mark": os.path.join(fig_root, "probe_3b_mark"),
        "3b_rand": os.path.join(fig_root, "probe_3b_rand"),
        "3b_spots": os.path.join(fig_root, "probe_3b_spots"),
        "3b_map": os.path.join(fig_root, "probe_3b_map"),
    }
    for key in probes:
        if key in probe_dirs:
            os.makedirs(probe_dirs[key], exist_ok=True)
    return probe_dirs


def _read_2d_image(path: str) -> np.ndarray:
    arr = np.asarray(tiff_read(path)).astype(np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image at {path}, got shape {arr.shape}")
    return arr


def _load_or_assemble_test_stacks(output_path: str, test_cfg: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_stack_path = os.path.join(output_path, "x_stack.tif")
    y_stack_path = os.path.join(output_path, "y_stack.tif")
    z_stack_path = os.path.join(output_path, "z_stack.tif")

    if os.path.exists(x_stack_path) and os.path.exists(y_stack_path) and os.path.exists(z_stack_path):
        x_stack = tiff_read(x_stack_path)
        y_stack = tiff_read(y_stack_path)
        z_stack = tiff_read(z_stack_path)
        if x_stack.ndim == 2:
            x_stack = x_stack[None, ...]
        if y_stack.ndim == 2:
            y_stack = y_stack[None, ...]
        if z_stack.ndim == 3:
            z_stack = z_stack[None, ...]
        return x_stack, y_stack, z_stack

    shard_pattern = re.compile(r"^([xyz])-(\d+)-(\d+)\.tif$")
    shard_map: Dict[str, Dict[int, Dict[int, str]]] = {"x": {}, "y": {}, "z": {}}
    for name in os.listdir(output_path):
        match = shard_pattern.fullmatch(name)
        if not match:
            continue
        key = match.group(1)
        step = int(match.group(2))
        sample = int(match.group(3))
        shard_map[key].setdefault(step, {})[sample] = os.path.join(output_path, name)

    common_steps = sorted(set(shard_map["x"]).intersection(shard_map["y"]).intersection(shard_map["z"]))
    if not common_steps:
        raise FileNotFoundError(
            f"Missing stack artifacts under {output_path}. Also found no compatible shard files "
            "matching x-<step>-<iter>.tif, y-<step>-<iter>.tif, z-<step>-<iter>.tif."
        )

    step_infos = []
    preferred_sample_idx = int(test_cfg.get("shard_sample_idx", 0))
    for step in common_steps:
        z_samples = sorted(shard_map["z"][step].keys())
        if not z_samples:
            continue
        x_samples = sorted(shard_map["x"][step].keys())
        y_samples = sorted(shard_map["y"][step].keys())
        if not x_samples or not y_samples:
            continue
        x_sample = preferred_sample_idx if preferred_sample_idx in shard_map["x"][step] else x_samples[0]
        y_sample = preferred_sample_idx if preferred_sample_idx in shard_map["y"][step] else y_samples[0]
        step_infos.append((step, x_sample, y_sample, z_samples))

    if not step_infos:
        raise FileNotFoundError(f"Found shard files in {output_path}, but none form complete x/y/z sets per step")

    inferred_iters = min(len(info[3]) for info in step_infos)
    target_iters = int(test_cfg.get("shard_n_iters", inferred_iters))
    target_iters = max(1, min(target_iters, inferred_iters))

    x_frames = []
    y_frames = []
    z_frames = []
    for step, x_sample, y_sample, z_samples in step_infos:
        if len(z_samples) < target_iters:
            continue
        x_img = _read_2d_image(shard_map["x"][step][x_sample])
        y_img = _read_2d_image(shard_map["y"][step][y_sample])
        pred_stack = []
        for z_sample in z_samples[:target_iters]:
            pred_stack.append(_read_2d_image(shard_map["z"][step][z_sample]))
        x_frames.append(x_img)
        y_frames.append(y_img)
        z_frames.append(np.stack(pred_stack, axis=0))

    if not x_frames or not y_frames or not z_frames:
        raise FileNotFoundError(f"Could not assemble any frame stacks from shards in {output_path}")

    x_stack = np.stack(x_frames, axis=0)
    y_stack = np.stack(y_frames, axis=0)
    z_stack = np.stack(z_frames, axis=0)

    if bool(test_cfg.get("write_stacks_from_shards", True)):
        imwrite(x_stack_path, x_stack.astype(np.float32))
        imwrite(y_stack_path, y_stack.astype(np.float32))
        imwrite(z_stack_path, z_stack.astype(np.float32))

    return x_stack, y_stack, z_stack


def render_test_plots(output_path: str, test_cfg: Dict, probes: Optional[List[str]] = None) -> None:
    x_stack, y_stack, z_stack = _load_or_assemble_test_stacks(output_path, test_cfg)

    probe_list = [p.lower() for p in (probes or test_cfg.get("probes", ["2a", "3a"]))]
    fig_root = os.path.join(output_path, "figures")
    probe_dirs = _ensure_probe_dirs(fig_root, probe_list)

    input_upsample = int(test_cfg.get("input_upsample", 4))
    contrast_low = float(test_cfg.get("contrast_low", 1.0))
    contrast_high = float(test_cfg.get("contrast_high", 99.0))
    inset_contrast_low = float(test_cfg.get("inset_contrast_low", contrast_low))
    inset_contrast_high = float(test_cfg.get("inset_contrast_high", contrast_high))
    inset_lr_size = int(test_cfg.get("inset_lr_size", 15))
    inset_hr_size = int(test_cfg.get("inset_hr_size", 60))
    inset_threshold = float(test_cfg.get("inset_threshold_percentile", 99.5))
    pixel_size_nm = float(test_cfg.get("pixel_size_nm", 44.0))
    pixel_size_hr_nm = pixel_size_nm / float(input_upsample)
    nanoruler_spacing_nm = float(test_cfg.get("nanoruler_spacing_nm", 94.0))
    nanoruler_tol_nm = float(test_cfg.get("nanoruler_tol_nm", 10.0))
    nanoruler_log_threshold = float(test_cfg.get("nanoruler_log_threshold", 0.1))
    nanoruler_min_sigma = float(test_cfg.get("nanoruler_min_sigma", 0.75))
    nanoruler_max_sigma = float(test_cfg.get("nanoruler_max_sigma", 1.5))
    nanoruler_dot_diameter_px = float(test_cfg.get("nanoruler_dot_diameter_px", 1.0))
    nanoruler_line_width_px = float(test_cfg.get("nanoruler_line_width_px", 0.7))
    nanoruler_dot_alpha = float(test_cfg.get("nanoruler_dot_alpha", 0.6))
    nanoruler_line_alpha = float(test_cfg.get("nanoruler_line_alpha", 0.6))
    nanoruler_rand_seed = int(test_cfg.get("nanoruler_rand_seed", 0))
    spot_max_spots_per_iter = test_cfg.get("spot_max_spots_per_iter")
    if spot_max_spots_per_iter is not None:
        spot_max_spots_per_iter = int(spot_max_spots_per_iter)
    spot_patch_size = int(test_cfg.get("spot_patch_size", 16))
    spot_seed = int(test_cfg.get("spot_seed", 0))

    rng = np.random.default_rng(nanoruler_rand_seed)
    spot_rng = np.random.default_rng(spot_seed)

    for frame_idx in range(x_stack.shape[0]):
        lr_raw = x_stack[frame_idx].astype(np.float32)
        x_up = y_stack[frame_idx].astype(np.float32)
        preds = z_stack[frame_idx].astype(np.float32)
        pred_mean = np.mean(preds, axis=0)
        pred_std = np.std(preds, axis=0)

        hr_x0, hr_y0 = _pick_center_inset_coords(pred_mean, inset_hr_size, inset_threshold)
        lr_x0 = max(min(hr_x0 // input_upsample, lr_raw.shape[0] - inset_lr_size), 0)
        lr_y0 = max(min(hr_y0 // input_upsample, lr_raw.shape[1] - inset_lr_size), 0)
        hr_inset = (slice(hr_x0, hr_x0 + inset_hr_size), slice(hr_y0, hr_y0 + inset_hr_size))
        lr_inset = (slice(lr_x0, lr_x0 + inset_lr_size), slice(lr_y0, lr_y0 + inset_lr_size))

        if "2a" in probe_list:
            fig2a, ax2a = plt.subplots(1, 3, figsize=(9, 3))
            _imshow_scaled(ax2a[0], lr_raw, contrast_low, contrast_high)
            ax2a[0].set_title(r"$x_{\mathrm{LR}}$")
            ax2a[0].set_xticks([])
            ax2a[0].set_yticks([])
            inset = ax2a[0].inset_axes([0.65, 0.65, 0.4, 0.4])
            _imshow_scaled(inset, lr_raw[lr_inset], inset_contrast_low, inset_contrast_high)
            inset.set_xticks([])
            inset.set_yticks([])
            for spine in inset.spines.values():
                spine.set_color("red")
                spine.set_linewidth(1)
            _draw_inset_box(ax2a[0], lr_x0, lr_y0, inset_lr_size)

            _imshow_scaled(ax2a[1], x_up, contrast_low, contrast_high)
            ax2a[1].set_title(r"$y_0$")
            ax2a[1].set_xticks([])
            ax2a[1].set_yticks([])
            inset = ax2a[1].inset_axes([0.65, 0.65, 0.4, 0.4])
            _imshow_scaled(inset, x_up[hr_inset], inset_contrast_low, inset_contrast_high)
            inset.set_xticks([])
            inset.set_yticks([])
            for spine in inset.spines.values():
                spine.set_color("red")
                spine.set_linewidth(1)
            _draw_inset_box(ax2a[1], hr_x0, hr_y0, inset_hr_size)

            _imshow_scaled(ax2a[2], pred_mean, contrast_low, contrast_high)
            ax2a[2].set_title(r"$\hat{y}$")
            ax2a[2].set_xticks([])
            ax2a[2].set_yticks([])
            inset = ax2a[2].inset_axes([0.65, 0.65, 0.4, 0.4])
            _imshow_scaled(inset, pred_mean[hr_inset], inset_contrast_low, inset_contrast_high)
            inset.set_xticks([])
            inset.set_yticks([])
            for spine in inset.spines.values():
                spine.set_color("red")
                spine.set_linewidth(1)
            _draw_inset_box(ax2a[2], hr_x0, hr_y0, inset_hr_size)
            plt.tight_layout()
            fig2a.savefig(os.path.join(probe_dirs["2a"], f"probe_2a_frame-{frame_idx:04d}.png"), dpi=200)
            plt.close(fig2a)

        if "3a" in probe_list:
            fig3a, ax3a = plt.subplots(2, 2, figsize=(6, 5))
            _imshow_scaled(ax3a[0, 0], lr_raw, contrast_low, contrast_high)
            ax3a[0, 0].set_title(r"$x_{\mathrm{LR}}$")
            ax3a[0, 0].set_xticks([])
            ax3a[0, 0].set_yticks([])
            inset = ax3a[0, 0].inset_axes([0.65, 0.65, 0.4, 0.4])
            _imshow_scaled(inset, lr_raw[lr_inset], inset_contrast_low, inset_contrast_high)
            inset.set_xticks([])
            inset.set_yticks([])
            for spine in inset.spines.values():
                spine.set_color("red")
                spine.set_linewidth(1)
            _draw_inset_box(ax3a[0, 0], lr_x0, lr_y0, inset_lr_size)

            _imshow_scaled(ax3a[0, 1], x_up, contrast_low, contrast_high)
            ax3a[0, 1].set_title(r"$y_0$")
            ax3a[0, 1].set_xticks([])
            ax3a[0, 1].set_yticks([])
            inset = ax3a[0, 1].inset_axes([0.65, 0.65, 0.4, 0.4])
            _imshow_scaled(inset, x_up[hr_inset], inset_contrast_low, inset_contrast_high)
            inset.set_xticks([])
            inset.set_yticks([])
            for spine in inset.spines.values():
                spine.set_color("red")
                spine.set_linewidth(1)
            _draw_inset_box(ax3a[0, 1], hr_x0, hr_y0, inset_hr_size)

            _imshow_scaled(ax3a[1, 0], pred_mean, contrast_low, contrast_high)
            ax3a[1, 0].set_title(r"$\langle \hat{y} \rangle$")
            ax3a[1, 0].set_xticks([])
            ax3a[1, 0].set_yticks([])
            inset = ax3a[1, 0].inset_axes([0.65, 0.65, 0.4, 0.4])
            _imshow_scaled(inset, pred_mean[hr_inset], inset_contrast_low, inset_contrast_high)
            inset.set_xticks([])
            inset.set_yticks([])
            for spine in inset.spines.values():
                spine.set_color("red")
                spine.set_linewidth(1)
            _draw_inset_box(ax3a[1, 0], hr_x0, hr_y0, inset_hr_size)

            _imshow_scaled(ax3a[1, 1], pred_std, contrast_low, contrast_high)
            ax3a[1, 1].set_title(r"$\sigma$")
            ax3a[1, 1].set_xticks([])
            ax3a[1, 1].set_yticks([])
            inset = ax3a[1, 1].inset_axes([0.65, 0.65, 0.4, 0.4])
            _imshow_scaled(inset, pred_std[hr_inset], inset_contrast_low, inset_contrast_high)
            inset.set_xticks([])
            inset.set_yticks([])
            for spine in inset.spines.values():
                spine.set_color("red")
                spine.set_linewidth(1)
            _draw_inset_box(ax3a[1, 1], hr_x0, hr_y0, inset_hr_size)
            plt.tight_layout()
            fig3a.savefig(os.path.join(probe_dirs["3a"], f"probe_3a_frame-{frame_idx:04d}.png"), dpi=200)
            plt.close(fig3a)

        if "3b" in probe_list:
            fig3b_samples = [int(i) for i in test_cfg.get("fig3b_samples", list(range(preds.shape[0])))]
            available = [i for i in fig3b_samples if 0 <= i < preds.shape[0]]
            if not available:
                available = list(range(preds.shape[0]))
            fig3b, ax3b = plt.subplots(1, len(available), figsize=(2 * len(available), 2))
            if len(available) == 1:
                ax3b = [ax3b]
            for idx, iter_idx in enumerate(available):
                _imshow_scaled(ax3b[idx], preds[iter_idx], contrast_low, contrast_high)
                ax3b[idx].set_title(rf"$\hat{{y}}_{{0,{iter_idx}}}$", fontsize=12)
                ax3b[idx].set_xticks([])
                ax3b[idx].set_yticks([])
            plt.tight_layout()
            fig3b.savefig(os.path.join(probe_dirs["3b"], f"probe_3b_frame-{frame_idx:04d}.png"), dpi=200)
            plt.close(fig3b)

        if "3b_mark" in probe_list:
            fig3b_samples = [int(i) for i in test_cfg.get("fig3b_samples", list(range(preds.shape[0])))]
            available = [i for i in fig3b_samples if 0 <= i < preds.shape[0]]
            if not available:
                available = list(range(preds.shape[0]))
            expected_px = nanoruler_spacing_nm / pixel_size_hr_nm
            tol_px = nanoruler_tol_nm / pixel_size_hr_nm
            fig3m, ax3m = plt.subplots(2, len(available), figsize=(2 * len(available), 4))
            if len(available) == 1:
                ax3m = np.array([[ax3m[0]], [ax3m[1]]])
            for idx, iter_idx in enumerate(available):
                _imshow_scaled(ax3m[0, idx], preds[iter_idx], contrast_low, contrast_high)
                pairs, rejected = _detect_nanorulers(
                    preds[iter_idx],
                    expected_distance_px=expected_px,
                    distance_tol_px=tol_px,
                    log_threshold=nanoruler_log_threshold,
                    min_sigma=nanoruler_min_sigma,
                    max_sigma=nanoruler_max_sigma,
                    require_exact_two=True,
                )
                _draw_nanoruler_marks(
                    ax3m[0, idx],
                    pairs,
                    rejected,
                    dot_diameter_px=nanoruler_dot_diameter_px,
                    line_width_px=nanoruler_line_width_px,
                    dot_alpha=nanoruler_dot_alpha,
                    line_alpha=nanoruler_line_alpha,
                )
                ax3m[0, idx].set_title(rf"$\hat{{y}}_{{0,{iter_idx}}}$", fontsize=12)
                ax3m[0, idx].set_xticks([])
                ax3m[0, idx].set_yticks([])
                distances_nm = [
                    float(np.linalg.norm(np.array(p1) - np.array(p2)) * pixel_size_hr_nm)
                    for p1, p2 in pairs
                ]
                ax_hist = ax3m[1, idx]
                if distances_nm:
                    hist_min = nanoruler_spacing_nm - nanoruler_tol_nm
                    hist_max = nanoruler_spacing_nm + nanoruler_tol_nm
                    bins = np.linspace(hist_min, hist_max, 6)
                    ax_hist.hist(distances_nm, bins=bins, color="gray", edgecolor="black")
                    ax_hist.set_xlabel(r"nm", fontsize=9)
                else:
                    ax_hist.axis("off")
            plt.tight_layout()
            fig3m.savefig(os.path.join(probe_dirs["3b_mark"], f"probe_3b_mark_frame-{frame_idx:04d}.png"), dpi=200)
            plt.close(fig3m)

        if "3b_rand" in probe_list:
            expected_px = nanoruler_spacing_nm / pixel_size_hr_nm
            tol_px = nanoruler_tol_nm / pixel_size_hr_nm
            pairs_first, _ = _detect_nanorulers(
                preds[0],
                expected_distance_px=expected_px,
                distance_tol_px=tol_px,
                log_threshold=nanoruler_log_threshold,
                min_sigma=nanoruler_min_sigma,
                max_sigma=nanoruler_max_sigma,
                require_exact_two=True,
            )
            fig_rand, ax_rand = plt.subplots(1, 3, figsize=(10, 3))
            hr_img = preds[0]
            if pairs_first:
                pair = pairs_first[int(rng.integers(0, len(pairs_first)))]
                (x1, y1), (x2, y2) = pair
                dist_nm = float(np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2])) * pixel_size_hr_nm)
                hr_x0 = int(round((x1 + x2) / 2.0 - inset_hr_size / 2))
                hr_y0 = int(round((y1 + y2) / 2.0 - inset_hr_size / 2))
                hr_x0 = max(min(hr_x0, hr_img.shape[0] - inset_hr_size), 0)
                hr_y0 = max(min(hr_y0, hr_img.shape[1] - inset_hr_size), 0)
                lr_x0 = max(min(hr_x0 // input_upsample, lr_raw.shape[0] - inset_lr_size), 0)
                lr_y0 = max(min(hr_y0 // input_upsample, lr_raw.shape[1] - inset_lr_size), 0)
                hr_inset = (slice(hr_x0, hr_x0 + inset_hr_size), slice(hr_y0, hr_y0 + inset_hr_size))
                lr_inset = (slice(lr_x0, lr_x0 + inset_lr_size), slice(lr_y0, lr_y0 + inset_lr_size))
            else:
                pair = None

            _imshow_scaled(ax_rand[0], lr_raw, contrast_low, contrast_high)
            ax_rand[0].set_title(r"$x_{\mathrm{LR}}$")
            ax_rand[0].set_xticks([])
            ax_rand[0].set_yticks([])
            ax_rand[0].set_aspect("equal")
            ax_rand[0].set_anchor("C")
            _imshow_scaled(ax_rand[1], hr_img, contrast_low, contrast_high)
            ax_rand[1].set_title(r"$\hat{y}$")
            ax_rand[1].set_xticks([])
            ax_rand[1].set_yticks([])
            ax_rand[1].set_aspect("equal")
            ax_rand[1].set_anchor("C")
            ax_rand[2].set_xlabel("distance (nm)", fontsize=9)
            ax_rand[2].set_ylabel("ADU", fontsize=9)
            ax_rand[2].spines["top"].set_visible(False)
            ax_rand[2].spines["right"].set_visible(False)
            ax_rand[2].set_box_aspect(1)

            if pair is not None:
                (x1, y1), (x2, y2) = pair
                _draw_nanoruler_marks(
                    ax_rand[1],
                    [pair],
                    [],
                    dot_diameter_px=nanoruler_dot_diameter_px,
                    line_width_px=nanoruler_line_width_px,
                    dot_alpha=nanoruler_dot_alpha,
                    line_alpha=nanoruler_line_alpha,
                )

                inset = ax_rand[0].inset_axes([0.65, 0.65, 0.4, 0.4])
                _imshow_scaled(inset, lr_raw[lr_inset], inset_contrast_low, inset_contrast_high)
                inset.set_xticks([])
                inset.set_yticks([])
                for spine in inset.spines.values():
                    spine.set_color("red")
                    spine.set_linewidth(1)
                _draw_inset_box(ax_rand[0], lr_x0, lr_y0, inset_lr_size)

                inset = ax_rand[1].inset_axes([0.65, 0.65, 0.4, 0.4])
                _imshow_scaled(inset, hr_img[hr_inset], inset_contrast_low, inset_contrast_high)
                inset.set_xticks([])
                inset.set_yticks([])
                for spine in inset.spines.values():
                    spine.set_color("red")
                    spine.set_linewidth(1)
                inset_pair = ((x1 - hr_x0, y1 - hr_y0), (x2 - hr_x0, y2 - hr_y0))
                _draw_nanoruler_marks(
                    inset,
                    [inset_pair],
                    [],
                    dot_diameter_px=nanoruler_dot_diameter_px,
                    line_width_px=nanoruler_line_width_px,
                    dot_alpha=nanoruler_dot_alpha,
                    line_alpha=nanoruler_line_alpha,
                )
                _draw_inset_box(ax_rand[1], hr_x0, hr_y0, inset_hr_size)

                midpoint = (np.array([x1, y1]) + np.array([x2, y2])) / 2.0
                direction = np.array([x2 - x1, y2 - y1])
                norm = float(np.linalg.norm(direction))
                if norm > 0:
                    direction = direction / norm
                half_len_px = (130.0 / 2.0) / pixel_size_hr_nm
                start = midpoint - direction * half_len_px
                end = midpoint + direction * half_len_px
                hr_profile = profile_line(hr_img, start, end, mode="reflect")
                x_nm = np.linspace(-65.0, 65.0, len(hr_profile))
                ax_rand[2].plot(x_nm, hr_profile, color="blue", alpha=0.3)
                peak_x = np.array([-dist_nm / 2.0, dist_nm / 2.0])
                inset_prof = ax_rand[1].inset_axes([0.02, 0.02, 0.4, 0.35])
                inset_prof.plot(x_nm, hr_profile, color="blue", alpha=0.3)
                inset_prof.set_xlabel("distance (nm)", fontsize=8)
                inset_prof.set_ylabel("ADU", fontsize=8)
                inset_prof.spines["top"].set_visible(False)
                inset_prof.spines["right"].set_visible(False)
                for px in peak_x:
                    inset_prof.axvline(px, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
                iy_min, iy_max = inset_prof.get_ylim()
                iy_arrow = iy_min + 0.75 * (iy_max - iy_min)
                inset_prof.annotate(
                    "",
                    xy=(peak_x[1], iy_arrow),
                    xytext=(peak_x[0], iy_arrow),
                    arrowprops=dict(arrowstyle="<->", color="black", lw=1.0),
                )
                inset_prof.text(
                    0.0,
                    iy_arrow + 0.02 * (iy_max - iy_min),
                    rf"${dist_nm:.1f}\,\mathrm{{nm}}$",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    bbox=dict(facecolor="none", edgecolor="none", alpha=0.0),
                )
                for px in peak_x:
                    ax_rand[2].axvline(px, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
                y_min, y_max = ax_rand[2].get_ylim()
                y_arrow = y_min + 0.75 * (y_max - y_min)
                ax_rand[2].annotate(
                    "",
                    xy=(peak_x[1], y_arrow),
                    xytext=(peak_x[0], y_arrow),
                    arrowprops=dict(arrowstyle="<->", color="black", lw=1.0),
                )
                ax_rand[2].text(
                    0.0,
                    y_arrow + 0.02 * (y_max - y_min),
                    rf"${dist_nm:.1f}\,\mathrm{{nm}}$",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    bbox=dict(facecolor="none", edgecolor="none", alpha=0.0),
                )
            else:
                ax_rand[0].text(0.05, 0.05, "no pair", color="red", transform=ax_rand[0].transAxes)
                ax_rand[1].text(0.05, 0.05, "no pair", color="red", transform=ax_rand[1].transAxes)

            plt.tight_layout()
            fig_rand.savefig(os.path.join(probe_dirs["3b_rand"], f"probe_3b_rand_frame-{frame_idx:04d}.png"), dpi=200)
            plt.close(fig_rand)

        if "3b_spots" in probe_list:
            frame_hr = preds
            spots = PipelineMLE2D(frame_hr[0][None, ...]).localize(
                threshold=nanoruler_log_threshold,
                min_sigma=nanoruler_min_sigma,
                max_sigma=nanoruler_max_sigma,
                fit_enabled=False,
                show_tqdm=False,
            )
            if not spots.empty:
                peak_coords = spots[["x", "y"]].to_numpy(dtype=float)
                half_patch = spot_patch_size / 2.0
                height, width = frame_hr[0].shape
                valid_mask = (
                    (peak_coords[:, 0] >= half_patch)
                    & (peak_coords[:, 0] <= height - half_patch)
                    & (peak_coords[:, 1] >= half_patch)
                    & (peak_coords[:, 1] <= width - half_patch)
                )
                peak_coords = peak_coords[valid_mask]
                if peak_coords.size == 0:
                    continue
                count = min(3, len(peak_coords))
                n_iters_local = frame_hr.shape[0]
                detections_per_iter = []
                for c_idx in range(n_iters_local):
                    det = PipelineMLE2D(frame_hr[c_idx][None, ...]).localize(
                        threshold=nanoruler_log_threshold,
                        min_sigma=nanoruler_min_sigma,
                        max_sigma=nanoruler_max_sigma,
                        fit_enabled=False,
                        show_tqdm=False,
                    )
                    if det.empty:
                        detections_per_iter.append(np.empty((0, 2), dtype=float))
                    else:
                        detections_per_iter.append(det[["x", "y"]].to_numpy(dtype=float))

                if spot_max_spots_per_iter is not None:
                    filtered_coords = []
                    for sx, sy in peak_coords:
                        x0 = sx - half_patch
                        x1 = sx + half_patch
                        y0 = sy - half_patch
                        y1 = sy + half_patch
                        exceeds = False
                        for peaks_iter in detections_per_iter:
                            if peaks_iter.size == 0:
                                continue
                            in_region = (
                                (peaks_iter[:, 0] >= x0)
                                & (peaks_iter[:, 0] <= x1)
                                & (peaks_iter[:, 1] >= y0)
                                & (peaks_iter[:, 1] <= y1)
                            )
                            if int(np.sum(in_region)) > spot_max_spots_per_iter:
                                exceeds = True
                                break
                        if not exceeds:
                            filtered_coords.append((sx, sy))
                    peak_coords = np.array(filtered_coords, dtype=float)
                    if peak_coords.size == 0:
                        continue
                    count = min(3, len(peak_coords))

                sel_idx = spot_rng.choice(len(peak_coords), size=count, replace=False)
                sel_coords = peak_coords[sel_idx]
                colors = ["red", "blue", "cyan"]
                fig_lr, ax_lr = plt.subplots(1, 1, figsize=(4, 4))
                _imshow_scaled(ax_lr, lr_raw, contrast_low, contrast_high)
                ax_lr.set_title(r"$x$", fontsize=10, pad=2)
                ax_lr.set_xticks([])
                ax_lr.set_yticks([])
                for r_idx, (sx, sy) in enumerate(sel_coords):
                    color = colors[r_idx % len(colors)]
                    lr_cx = sx / float(input_upsample)
                    lr_cy = sy / float(input_upsample)
                    lr_size = spot_patch_size / float(input_upsample)
                    lr_box_x0 = lr_cx - lr_size / 2.0
                    lr_box_y0 = lr_cy - lr_size / 2.0
                    ax_lr.add_patch(
                        patches.Rectangle(
                            (lr_box_y0, lr_box_x0),
                            lr_size,
                            lr_size,
                            linewidth=1.2,
                            edgecolor=color,
                            facecolor="none",
                        )
                    )
                fig_lr.tight_layout(pad=0.1)
                fig_lr.savefig(
                    os.path.join(probe_dirs["3b_spots"], f"probe_3b_spots_lr_frame-{frame_idx:04d}.png"),
                    dpi=200,
                )
                plt.close(fig_lr)

                fig_montage, axes_montage = plt.subplots(
                    count,
                    n_iters_local,
                    figsize=(1.5 * n_iters_local, 1.5 * count),
                    squeeze=False,
                )
                for r_idx, (sx, sy) in enumerate(sel_coords):
                    color = colors[r_idx % len(colors)]
                    for c_idx in range(n_iters_local):
                        patch = _extract_patch(frame_hr[c_idx], sx, sy, spot_patch_size)
                        _imshow_scaled(axes_montage[r_idx, c_idx], patch, contrast_low, contrast_high)
                        axes_montage[r_idx, c_idx].set_xticks([])
                        axes_montage[r_idx, c_idx].set_yticks([])
                        for spine in axes_montage[r_idx, c_idx].spines.values():
                            spine.set_edgecolor(color)
                            spine.set_linewidth(1.2)
                fig_montage.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.02, hspace=0.02)
                fig_montage.savefig(
                    os.path.join(probe_dirs["3b_spots"], f"probe_3b_spots_montage_frame-{frame_idx:04d}.png"),
                    dpi=200,
                )
                plt.close(fig_montage)

                fig_hist, axes_hist = plt.subplots(1, count, figsize=(3.0 * count, 2.5), squeeze=False)
                for r_idx, (sx, sy) in enumerate(sel_coords):
                    color = colors[r_idx % len(colors)]
                    region_counts = []
                    x0 = sx - half_patch
                    x1 = sx + half_patch
                    y0 = sy - half_patch
                    y1 = sy + half_patch
                    for c_idx in range(n_iters_local):
                        peaks_iter = detections_per_iter[c_idx]
                        if peaks_iter.size == 0:
                            region_counts.append(0)
                        else:
                            in_region = (
                                (peaks_iter[:, 0] >= x0)
                                & (peaks_iter[:, 0] <= x1)
                                & (peaks_iter[:, 1] >= y0)
                                & (peaks_iter[:, 1] <= y1)
                            )
                            region_counts.append(int(np.sum(in_region)))
                    ax_hist = axes_hist[0, r_idx]
                    max_count = max(region_counts) if region_counts else 0
                    bins = np.arange(max_count + 2) - 0.5
                    ax_hist.hist(region_counts, bins=bins, color=color, edgecolor="black")
                    ax_hist.set_xlim(-0.5, max_count + 0.5)
                    ax_hist.set_xticks(range(max_count + 1))
                    ax_hist.set_xlabel("spots", fontsize=9)
                    ax_hist.set_ylabel("count", fontsize=9)
                    ax_hist.yaxis.set_major_locator(MaxNLocator(integer=True))
                    ax_hist.spines["top"].set_visible(False)
                    ax_hist.spines["right"].set_visible(False)
                fig_hist.tight_layout(pad=0.4)
                fig_hist.savefig(
                    os.path.join(probe_dirs["3b_spots"], f"probe_3b_spots_hist_frame-{frame_idx:04d}.png"),
                    dpi=200,
                )
                plt.close(fig_hist)

        if "3b_map" in probe_list:
            frame_hr = preds
            n_iters_local = frame_hr.shape[0]
            lr_scale = float(frame_hr.shape[1]) / float(lr_raw.shape[0]) if lr_raw.shape[0] else 1.0
            if lr_scale < 1.5:
                lr_scale = 1.0
            detections_per_iter = []
            for c_idx in range(n_iters_local):
                det = PipelineMLE2D(frame_hr[c_idx][None, ...]).localize(
                    threshold=nanoruler_log_threshold,
                    min_sigma=nanoruler_min_sigma,
                    max_sigma=nanoruler_max_sigma,
                    fit_enabled=False,
                    show_tqdm=False,
                )
                if det.empty:
                    detections_per_iter.append(np.empty((0, 2), dtype=float))
                else:
                    detections_per_iter.append(det[["x", "y"]].to_numpy(dtype=float))

            fig_map = plt.figure(figsize=(18, 12))
            grid = fig_map.add_gridspec(3, 2, height_ratios=[0.9, 1, 1.1], width_ratios=[1, 1.2])
            ax_raw = fig_map.add_subplot(grid[0, 0])
            _imshow_scaled(ax_raw, lr_raw, contrast_low, contrast_high)
            ax_raw.set_title("LR", fontsize=10)
            ax_raw.set_xticks([])
            ax_raw.set_yticks([])
            ax_raw.set_aspect("equal", adjustable="box")

            ax_cluster = fig_map.add_subplot(grid[1, 0])
            _imshow_scaled(ax_cluster, lr_raw, contrast_low, contrast_high)
            ax_cluster.set_title("3b_map: by cluster", fontsize=10)
            ax_cluster.set_xticks([])
            ax_cluster.set_yticks([])
            ax_cluster.set_aspect("equal", adjustable="box")

            lr_points = []
            lr_point_iters = []
            for t_idx, peaks in enumerate(detections_per_iter):
                if peaks.size == 0:
                    continue
                lr_x = peaks[:, 0] / lr_scale
                lr_y = peaks[:, 1] / lr_scale
                for x_val, y_val in zip(lr_x, lr_y):
                    lr_points.append((float(x_val), float(y_val)))
                    lr_point_iters.append(int(t_idx))

            if lr_points:
                lr_points_arr = np.array(lr_points, dtype=float)
                cluster_radius_lr = 2.0
                clusters = _cluster_points(lr_points_arr, cluster_radius_lr)
                cluster_colors = plt.cm.tab20(np.linspace(0, 1, max(len(clusters), 1)))
                cluster_centers = []
                for c_idx, cluster in enumerate(clusters):
                    pts = lr_points_arr[cluster]
                    center = np.array([float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))])
                    cluster_centers.append(center)
                    ax_cluster.scatter(pts[:, 1], pts[:, 0], s=8, color=cluster_colors[c_idx], alpha=0.5, zorder=5)
                    counts = np.zeros(n_iters_local, dtype=int)
                    for pt_idx in cluster:
                        counts[lr_point_iters[pt_idx]] += 1
                    n_mode = int(np.bincount(counts).argmax())
                    n_mode = max(1, min(n_mode, len(pts)))
                    means, _, _ = _gmm_fit(pts, n_mode)
                    if len(means):
                        ax_cluster.scatter(
                            means[:, 1],
                            means[:, 0],
                            s=40,
                            color="red",
                            marker="x",
                            linewidths=1.4,
                            zorder=6,
                        )

                if cluster_centers:
                    n_clusters = len(cluster_centers)
                    n_cols = int(math.ceil(math.sqrt(n_clusters)))
                    n_rows = int(math.ceil(n_clusters / n_cols))
                    hist_grid = GridSpecFromSubplotSpec(
                        n_rows,
                        n_cols,
                        subplot_spec=grid[1, 1],
                        wspace=0.5,
                        hspace=0.6,
                    )
                    gmm_grid = GridSpecFromSubplotSpec(
                        n_rows,
                        n_cols,
                        subplot_spec=grid[2, :],
                        wspace=0.3,
                        hspace=0.4,
                    )
                    for idx in range(n_rows * n_cols):
                        ax_hist = fig_map.add_subplot(hist_grid[idx // n_cols, idx % n_cols])
                        ax_gmm = fig_map.add_subplot(gmm_grid[idx // n_cols, idx % n_cols])
                        ax_hist.set_xticks([])
                        ax_hist.set_yticks([])
                        ax_gmm.set_xticks([])
                        ax_gmm.set_yticks([])
                        if idx >= n_clusters:
                            ax_hist.axis("off")
                            ax_gmm.axis("off")
                            continue
                        cluster = clusters[idx]
                        counts = np.zeros(n_iters_local, dtype=int)
                        for pt_idx in cluster:
                            counts[lr_point_iters[pt_idx]] += 1
                        ax_hist.hist(
                            counts,
                            bins=np.arange(np.max(counts) + 2) - 0.5,
                            density=True,
                            color=cluster_colors[idx],
                            edgecolor="black",
                        )
                        ax_hist.set_xlabel("Spot Count", fontsize=8)
                        ax_hist.set_ylabel("Density", fontsize=8)
                        ax_hist.set_box_aspect(1)
                        ax_hist.set_xticks(np.arange(np.max(counts) + 1))
                        ax_hist.xaxis.set_major_locator(MaxNLocator(integer=True))
                        ax_hist.set_yticks([])
                        ax_hist.tick_params(axis="x", labelsize=7)
                        pts = lr_points_arr[cluster]
                        if pts.size == 0:
                            ax_gmm.axis("off")
                            continue
                        n_mode = int(np.bincount(counts).argmax())
                        n_mode = max(1, min(n_mode, len(pts)))
                        means, covs, ll = _gmm_fit(pts, n_mode)
                        patch_size_lr = max(3, int(round(spot_patch_size / lr_scale)))
                        center = cluster_centers[idx]
                        patch = _extract_patch(lr_raw, center[0], center[1], patch_size_lr)
                        _imshow_scaled(ax_gmm, patch, contrast_low, contrast_high)
                        ax_gmm.set_xlim(0, patch_size_lr)
                        ax_gmm.set_ylim(patch_size_lr, 0)
                        ax_gmm.set_aspect("equal", adjustable="box")

                        x0 = center[0] - patch_size_lr / 2.0
                        y0 = center[1] - patch_size_lr / 2.0
                        pts_local = pts - np.array([x0, y0])
                        ax_gmm.scatter(pts_local[:, 1], pts_local[:, 0], s=8, color=cluster_colors[idx], alpha=0.5)
                        if len(means):
                            grid_n = 60
                            xs = np.linspace(0, patch_size_lr, grid_n)
                            ys = np.linspace(0, patch_size_lr, grid_n)
                            xx, yy = np.meshgrid(xs, ys)
                            if len(means) > 1:
                                mean_dists = []
                                for i in range(len(means)):
                                    for j in range(i + 1, len(means)):
                                        mean_dists.append(float(np.linalg.norm(means[i] - means[j])))
                                avg_mean_dist = float(np.mean(mean_dists)) if mean_dists else float("nan")
                            else:
                                avg_mean_dist = float("nan")
                            for k_idx in range(len(means)):
                                var = covs[k_idx]
                                mean_local = means[k_idx] - np.array([x0, y0])
                                diff_x = xx - mean_local[0]
                                diff_y = yy - mean_local[1]
                                exponent = -0.5 * (diff_x ** 2 / var[0] + diff_y ** 2 / var[1])
                                norm = 1.0 / (2.0 * np.pi * np.sqrt(var[0] * var[1]))
                                zz = norm * np.exp(exponent)
                                ax_gmm.contour(yy, xx, zz, colors="black", linewidths=0.8, alpha=0.8)
                                ax_gmm.scatter(
                                    mean_local[1],
                                    mean_local[0],
                                    s=50,
                                    color="red",
                                    marker="x",
                                    linewidths=1.5,
                                )
                        if np.isfinite(avg_mean_dist):
                            ax_gmm.set_title(f"N={n_mode}, LL={ll:.1f}, d={avg_mean_dist:.2f}", fontsize=8)
                        else:
                            ax_gmm.set_title(f"N={n_mode}, LL={ll:.1f}", fontsize=8)

                    rng_local = np.random.default_rng(spot_seed + frame_idx)
                    n_cols = 5
                    if n_iters_local >= n_cols:
                        iter_samples = rng_local.choice(n_iters_local, size=n_cols, replace=False)
                    else:
                        iter_samples = rng_local.choice(n_iters_local, size=n_cols, replace=True)
                    montage_fig = plt.figure(figsize=(2 * n_cols, 2 * len(cluster_centers)))
                    montage_grid = montage_fig.add_gridspec(len(cluster_centers), n_cols, wspace=0.0, hspace=0.0)
                    for r_idx, center in enumerate(cluster_centers):
                        hr_center = center * lr_scale
                        for c_idx, iter_idx in enumerate(iter_samples):
                            ax = montage_fig.add_subplot(montage_grid[r_idx, c_idx])
                            patch_hr = _extract_patch(frame_hr[iter_idx], hr_center[0], hr_center[1], spot_patch_size)
                            _imshow_scaled(ax, patch_hr, contrast_low, contrast_high)
                            ax.set_xticks([])
                            ax.set_yticks([])
                            row_color = cluster_colors[r_idx]
                            for spine in ax.spines.values():
                                spine.set_edgecolor(row_color)
                                spine.set_linewidth(1.2)
                    montage_fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
                    montage_fig.savefig(
                        os.path.join(probe_dirs["3b_map"], f"probe_3b_map_montage_frame-{frame_idx:04d}.png"),
                        dpi=200,
                    )
                    plt.close(montage_fig)

            fig_map.tight_layout(pad=0.2)
            fig_map.savefig(os.path.join(probe_dirs["3b_map"], f"probe_3b_map_frame-{frame_idx:04d}.png"), dpi=200)
            plt.close(fig_map)


def _resolve_test_outputs_only(plot_cfg: Dict) -> List[Tuple[str, Dict]]:
    output_dirs = plot_cfg.get("output_dirs", [])
    if isinstance(output_dirs, str):
        output_dirs = [output_dirs]
    if not output_dirs:
        single = plot_cfg.get("output_dir")
        if single:
            output_dirs = [single]
    if not output_dirs:
        raise ValueError("Set plot.output_dir or plot.output_dirs for mode=test")
    return [(path, plot_cfg) for path in output_dirs]


def _infer_probe_cache_layout(output_dir: str, cache_dir: str) -> Tuple[List[int], int]:
    cache_root = os.path.join(output_dir, cache_dir)
    if not os.path.isdir(cache_root):
        raise FileNotFoundError(
            "Probe cache directory not found: "
            f"{cache_root}. "
            "Probe plotting reads existing cached files. "
            "First run probe generation with cache enabled, or point plot.probe_cache_dir "
            "(or plot.cache_dir) to an existing cache location."
        )

    density_dirs = []
    for name in os.listdir(cache_root):
        match = re.fullmatch(r"density_(.+)", name)
        if not match:
            continue
        try:
            density_val = int(match.group(1))
        except ValueError:
            continue
        full_path = os.path.join(cache_root, name)
        if os.path.isdir(full_path):
            density_dirs.append((density_val, full_path))

    if not density_dirs:
        raise FileNotFoundError(f"No density_* folders found in: {cache_root}")

    density_dirs.sort(key=lambda item: item[0])
    densities = [item[0] for item in density_dirs]

    per_density_counts = []
    for _, folder in density_dirs:
        sample_indices = []
        for fname in os.listdir(folder):
            match = re.fullmatch(r"sample_(\d+)\.npz", fname)
            if match:
                sample_indices.append(int(match.group(1)))
        if sample_indices:
            per_density_counts.append(max(sample_indices) + 1)

    if not per_density_counts:
        raise FileNotFoundError(f"No sample_*.npz cache files found under: {cache_root}")

    # Use the minimum complete count across densities so plotting never asks for missing samples.
    n_images = int(min(per_density_counts))
    return densities, n_images


def _build_probe_runtime_config(plot_cfg: Dict) -> Dict:
    output_dir = plot_cfg.get("probe_output_dir", plot_cfg.get("output_dir", None))
    if not output_dir:
        raise ValueError("mode=probe requires plot.probe_output_dir (or plot.output_dir)")
    os.makedirs(output_dir, exist_ok=True)

    cache_dir = plot_cfg.get("probe_cache_dir", plot_cfg.get("cache_dir", "probe_cache"))
    inferred_densities, inferred_n_images = _infer_probe_cache_layout(output_dir, cache_dir)

    densities = plot_cfg.get("densities", inferred_densities)
    n_images = int(plot_cfg.get("n_images", inferred_n_images))

    return {
        "output_dir": output_dir,
        "model": {
            "noise_model_type": "unet",
            "alpha": float(plot_cfg.get("alpha", 0.001)),
            "load_weights": None,
            "load_mu_weights": None,
            "snr_expansion_n": int(plot_cfg.get("snr_expansion_n", 1)),
            "zmd": bool(plot_cfg.get("zmd", False)),
            "diff_inp": bool(plot_cfg.get("diff_inp", False)),
        },
        "eval": {
            "generation_timesteps": int(plot_cfg.get("generation_timesteps", 200)),
            "image_freq": int(plot_cfg.get("image_freq", 100)),
            "checkpoint_freq": int(plot_cfg.get("checkpoint_freq", 1000)),
            "log_freq": int(plot_cfg.get("log_freq", 100)),
            "val_freq": int(plot_cfg.get("val_freq", 200)),
            "val_len": int(plot_cfg.get("val_len", 10)),
        },
        "sim": {
            "generator": plot_cfg.get("generator", "Nanoruler2D"),
            "size": int(plot_cfg.get("size", 64)),
            "sigma": float(plot_cfg.get("sigma", 1.0)),
            "input_upsample": int(plot_cfg.get("input_upsample", 4)),
            "label_upsample": int(plot_cfg.get("label_upsample", 4)),
            "label_sigma": float(plot_cfg.get("label_sigma", 2.0)),
            "label_scale": float(plot_cfg.get("label_scale", 100.0)),
            "label_centering": bool(plot_cfg.get("label_centering", True)),
        },
        "probe": {
            "probes": plot_cfg.get("probes", ["3b_map"]),
            "densities": densities,
            "n_images": n_images,
            "n_iters": int(plot_cfg.get("n_iters", 100)),
            "show_tqdm": bool(plot_cfg.get("show_tqdm", True)),
            "save_cache": False,
            "use_probe_cache": True,
            "subtract_offset": bool(plot_cfg.get("subtract_offset", False)),
            "input_centering": plot_cfg.get("input_centering", "zscore"),
            "cache_dir": cache_dir,
            "detect_on": plot_cfg.get("detect_on", "mean"),
            "inset_lr_size": int(plot_cfg.get("inset_lr_size", 15)),
            "inset_hr_size": int(plot_cfg.get("inset_hr_size", 60)),
            "fig3_density": int(plot_cfg.get("fig3_density", densities[0])),
            "fig3_inset": plot_cfg.get("fig3_inset", [50, 50]),
            "fig3_inset_in_lr": bool(plot_cfg.get("fig3_inset_in_lr", True)),
            "fig3_samples": plot_cfg.get("fig3_samples", [0, 1, 2, 3]),
            "std_overlay_percentile": float(plot_cfg.get("std_overlay_percentile", 99.5)),
            "contrast_low": float(plot_cfg.get("contrast_low", 1.0)),
            "contrast_high": float(plot_cfg.get("contrast_high", 99.0)),
            "inset_contrast_low": float(plot_cfg.get("inset_contrast_low", 1.0)),
            "inset_contrast_high": float(plot_cfg.get("inset_contrast_high", 99.0)),
            "corr_window": int(plot_cfg.get("corr_window", 3)),
            "map_cluster_radius_lr": float(plot_cfg.get("map_cluster_radius_lr", 2.0)),
            "map_patch_size_lr": int(plot_cfg.get("map_patch_size_lr", 9)),
            "fig3c_bins": int(plot_cfg.get("fig3c_bins", 20)),
        },
        "metrics": {
            "tol": float(plot_cfg.get("tol", 5.0)),
            "pixel_size_nm": float(plot_cfg.get("pixel_size_nm", 44.0)),
            "error_bins": plot_cfg.get("error_bins", [500, 600, 700, 800, 900]),
        },
        "detection": {
            "log_threshold": float(plot_cfg.get("log_threshold", 2.0)),
            "min_sigma": float(plot_cfg.get("min_sigma", 0.75)),
            "max_sigma": float(plot_cfg.get("max_sigma", 1.5)),
            "fit_enabled": bool(plot_cfg.get("fit_enabled", False)),
        },
    }


def _run_probe_plot_wrapper(runtime_cfg: Dict) -> None:
    from cvdm.mains import probe as probe_main

    fd, temp_path = tempfile.mkstemp(prefix="cvdm_plot_probe_", suffix=".yaml")
    os.close(fd)
    try:
        with open(temp_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(runtime_cfg, handle, sort_keys=False)
        argv_prev = sys.argv
        try:
            sys.argv = ["probe.py", "--config", temp_path]
            probe_main.main()
        finally:
            sys.argv = argv_prev
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render CVDM probe plots from saved outputs.")
    parser.add_argument("--config", required=True, help="Path to plotting config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mode = cfg.get("mode", "test")

    if mode == "test":
        plot_cfg = cfg.get("plot", {})
        if "test_config" in cfg:
            raise ValueError("mode=test does not allow test_config; use plot.output_dir or plot.output_dirs only")
        datasets = _resolve_test_outputs_only(plot_cfg)
        if not datasets:
            raise ValueError("No test output paths resolved. Set plot.output_dir or plot.output_dirs")
        probes = plot_cfg.get("probes", None)
        for output_dir, test_cfg in datasets:
            render_test_plots(output_dir, test_cfg, probes=probes)
        return

    if mode == "probe":
        plot_cfg = cfg.get("plot", {})
        if "probe_config" in cfg or "probe_template_config" in cfg:
            raise ValueError("mode=probe does not allow probe_config/probe_template_config; use plot.probe_output_dir and inline plot settings")
        runtime_cfg = _build_probe_runtime_config(plot_cfg)
        _run_probe_plot_wrapper(runtime_cfg)
        return

    raise ValueError("mode must be one of: test, probe")


if __name__ == "__main__":
    main()
