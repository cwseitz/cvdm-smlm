import argparse
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpecFromSubplotSpec
import tensorflow as tf
from tqdm import tqdm
from skimage.transform import resize
from skimage.filters import gaussian

from cvdm.models.joint_model import instantiate_cvdm
from cvdm.configs_pkg.utils import create_model_config
from cvdm.make.kde import BasicKDE
from cvdm.generators.generators import Nanoruler2D, Uniform2D
from cvdm.psf.mle2d import PipelineMLE2D
from cvdm.utils.inference_utils import ddpm_obtain_sr_img
from cvdm.utils.zoom import custom_zoom
from cvdm.psf.psf2d.psf2d import lamx, lamy
from cvdm.utils.errors import errors2d


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_generator(name: str):
    if name == "Nanoruler2D":
        return Nanoruler2D
    if name == "Uniform2D":
        return Uniform2D
    raise ValueError("sim.generator must be 'Nanoruler2D' or 'Uniform2D'")


def _upsample_image(image: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return image
    steps = int(np.log2(factor))
    out = image
    for _ in range(steps):
        out = custom_zoom(out)
    return out


def _center_input_zscore(image: np.ndarray) -> np.ndarray:
    mean = float(np.mean(image))
    std = float(np.std(image))
    if std <= 0:
        return image - mean
    return (image - mean) / std


def _kde_half_max(sigma: float) -> float:
    coord = np.array([[0.0]], dtype=np.float64)
    peak = float(lamx(coord, 0.0, sigma) * lamy(coord, 0.0, sigma))
    return 0.5 * peak


def _make_kde_label(theta: np.ndarray, size: int, upsample: int, sigma: float, scale: float, center: bool) -> np.ndarray:
    theta_xy = theta[:2, :].T
    kde = BasicKDE(theta_xy).forward(size, upsample=upsample, sigma=sigma)
    if scale != 1.0:
        kde = kde * scale
    if center:
        kde = kde - _kde_half_max(sigma) * scale
    return kde.astype(np.float32)


def _match_spots(gt_xy: np.ndarray, pred_xy: np.ndarray, tol: float) -> Tuple[np.ndarray, np.ndarray]:
    if gt_xy.size == 0 or pred_xy.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    used_pred = set()
    gt_idx = []
    pred_idx = []
    for i, gt in enumerate(gt_xy):
        dists = np.linalg.norm(pred_xy - gt, axis=1)
        j = int(np.argmin(dists))
        if dists[j] <= tol and j not in used_pred:
            used_pred.add(j)
            gt_idx.append(i)
            pred_idx.append(j)
    return np.array(gt_idx, dtype=int), np.array(pred_idx, dtype=int)


def _compute_precision_recall(tp: int, fp: int, fn: int) -> Tuple[float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def _save_fig(fig: plt.Figure, output_dir: str, name: str) -> None:
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _scaled_limits(image: np.ndarray, low: float, high: float) -> Tuple[float, float]:
    vmin, vmax = np.nanpercentile(image, [low, high])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(image))
        vmax = float(np.nanmax(image))
    if vmin == vmax:
        vmax = vmin + 1e-6
    return float(vmin), float(vmax)


def _imshow_scaled(
    ax: plt.Axes,
    image: np.ndarray,
    low: float,
    high: float,
    cmap: str = "gray",
    **kwargs,
) -> None:
    vmin, vmax = _scaled_limits(image, low, high)
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)


def _normalize_weight_map(weight_map) -> Dict[str, str]:
    if not isinstance(weight_map, dict):
        return {}
    return {str(key): value for key, value in weight_map.items()}


def _detect_spots(
    frame: np.ndarray,
    detection_cfg: dict,
) -> pd.DataFrame:
    stack = frame[None, ...]
    detector = PipelineMLE2D(stack)
    cam_params = detection_cfg.get("cam_params", None)
    fit_enabled = bool(detection_cfg.get("fit_enabled", True))
    return detector.localize(
        plot_spots=False,
        plot_fit=False,
        tmax=None,
        threshold=float(detection_cfg.get("log_threshold", 0.1)),
        min_sigma=float(detection_cfg.get("min_sigma", 0.75)),
        max_sigma=float(detection_cfg.get("max_sigma", 1.5)),
        n_jobs=int(detection_cfg.get("n_jobs", 1)),
        max_fit_distance=detection_cfg.get("max_fit_distance", None) if fit_enabled else None,
        cam_params=cam_params,
        sigma_psf=float(detection_cfg.get("sigma_psf", 1.0)),
        fit_enabled=fit_enabled,
        max_iters=int(detection_cfg.get("max_iters", 100)),
        patchw=int(detection_cfg.get("patchw", 3)),
        fit_model=detection_cfg.get("fit_model", "aniso"),
        sigma_x_init=detection_cfg.get("sigma_x_init", None),
        sigma_y_init=detection_cfg.get("sigma_y_init", None),
        theta_init=detection_cfg.get("theta_init", None),
        show_tqdm=False,
    )


def _pick_inset_coords(
    spots,
    hr_size: int,
    inset_size: int,
) -> List[Tuple[int, int]]:
    center_val = hr_size / 2.0
    center = max(int(round(center_val - inset_size / 2)), 0)
    if spots is None or spots.empty:
        return [(center, center)]
    if "x_mle" in spots.columns and "y_mle" in spots.columns:
        xs = spots["x_mle"].to_numpy()
        ys = spots["y_mle"].to_numpy()
    else:
        xs = spots["x"].to_numpy()
        ys = spots["y"].to_numpy()
    if xs.size == 0:
        return [(center, center)]
    dist2 = (xs - center_val) ** 2 + (ys - center_val) ** 2
    idx = int(np.argmin(dist2))
    x_val = xs[idx]
    y_val = ys[idx]
    x0 = int(round(x_val - inset_size / 2))
    y0 = int(round(y_val - inset_size / 2))
    x0 = max(min(x0, hr_size - inset_size), 0)
    y0 = max(min(y0, hr_size - inset_size), 0)
    return [(x0, y0)]


def _draw_inset_box(ax, x0: int, y0: int, size: int) -> None:
    rect = patches.Rectangle(
        (y0, x0),
        size,
        size,
        linewidth=1,
        edgecolor="red",
        facecolor="none",
    )
    ax.add_patch(rect)


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


def _gmm_fit(points: np.ndarray, k: int, n_iter: int = 30) -> Tuple[np.ndarray, np.ndarray, float]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run probe figures on simulated data.")
    parser.add_argument("--config", required=True, type=str, help="Path to probe config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = config.get("output_dir", "./probe_outputs")
    os.makedirs(output_dir, exist_ok=True)

    model_cfg = config["model"]
    eval_cfg = config.get("eval", {})
    model_config = create_model_config(config)

    sim_cfg = config.get("sim", {})
    probe_cfg = config.get("probe", {})
    contrast_low = float(probe_cfg.get("contrast_low", 1.0))
    contrast_high = float(probe_cfg.get("contrast_high", 99.0))
    inset_contrast_low = float(probe_cfg.get("inset_contrast_low", contrast_low))
    inset_contrast_high = float(probe_cfg.get("inset_contrast_high", contrast_high))
    std_overlay_percentile = float(probe_cfg.get("std_overlay_percentile", 80.0))
    detection_cfg = config.get("detection", {})
    metrics_cfg = config.get("metrics", {})

    generator_cls = _resolve_generator(sim_cfg.get("generator", "Nanoruler2D"))
    size = int(sim_cfg.get("size", 64))
    generator = generator_cls(size)

    input_upsample = int(sim_cfg.get("input_upsample", 4))
    label_upsample = int(sim_cfg.get("label_upsample", 4))
    label_sigma = float(sim_cfg.get("label_sigma", 2.0))
    label_scale = float(sim_cfg.get("label_scale", 1.0))
    label_centering = bool(sim_cfg.get("label_centering", True))

    densities = probe_cfg.get("densities", [100])
    n_images = int(probe_cfg.get("n_images", 10))
    n_iters = int(probe_cfg.get("n_iters", 5))
    show_tqdm = bool(probe_cfg.get("show_tqdm", True))
    save_cache = bool(probe_cfg.get("save_cache", True))
    use_probe_cache = bool(probe_cfg.get("use_probe_cache", False))
    subtract_offset = bool(probe_cfg.get("subtract_offset", False))
    input_centering = probe_cfg.get("input_centering", sim_cfg.get("input_centering", "zscore"))
    cache_dir = probe_cfg.get("cache_dir", "probe_cache")
    cache_root = os.path.join(output_dir, cache_dir)
    os.makedirs(cache_root, exist_ok=True)
    detect_on = probe_cfg.get("detect_on", "mean")  # mean | single
    probes = probe_cfg.get("probes", []) or []
    map_cluster_radius_lr = float(probe_cfg.get("map_cluster_radius_lr", 2.0))
    map_patch_size_lr = int(probe_cfg.get("map_patch_size_lr", 9))

    load_weights_map = _normalize_weight_map(model_cfg.get("load_weights_map", {}))
    load_mu_weights_map = _normalize_weight_map(model_cfg.get("load_mu_weights_map", {}))

    if not use_probe_cache:
        # Build model
        cond_shape = (size * input_upsample, size * input_upsample, 1)
        generation_timesteps = int(eval_cfg.get("generation_timesteps", 200))
        training_cfg = config.get("training", {})
        probe_lr = float(training_cfg.get("lr", 1e-4))
        noise_model, joint_model, schedule_model, mu_model = instantiate_cvdm(
            lr=probe_lr,
            generation_timesteps=generation_timesteps,
            cond_shape=tf.TensorShape(cond_shape),
            out_shape=tf.TensorShape((size * label_upsample, size * label_upsample, 1)),
            model_config=model_config,
        )
        base_weights = model_cfg.get("load_weights")
        base_mu_weights = model_cfg.get("load_mu_weights")
        if base_weights:
            joint_model.load_weights(base_weights)
        if base_mu_weights and mu_model is not None:
            mu_model.load_weights(base_mu_weights)

    # Storage for metrics
    density_metrics = {}
    error_records = {str(d): [] for d in densities}
    sample_by_density: Dict[str, Dict[str, np.ndarray]] = {}
    std_records = {str(d): {"x": [], "y": []} for d in densities}

    density_iter = tqdm(densities, desc="Densities") if show_tqdm else densities
    for density in density_iter:
        if not use_probe_cache:
            density_key = str(density)
            density_weights = load_weights_map.get(density_key)
            density_mu_weights = load_mu_weights_map.get(density_key)
            if density_weights:
                joint_model.load_weights(density_weights)
            elif load_weights_map and base_weights is None:
                raise ValueError(f"Missing load_weights_map entry for density {density_key}")
            if density_mu_weights and mu_model is not None:
                mu_model.load_weights(density_mu_weights)
            elif load_mu_weights_map and base_mu_weights is None and mu_model is not None:
                raise ValueError(f"Missing load_mu_weights_map entry for density {density_key}")
        precision_list = []
        recall_list = []
        sample_lr = None
        sample_hr_true = None
        sample_hr_pred = None
        sample_preds = None
        sample_mean = None
        sample_std = None
        sample_detect_frame = None
        sample_spots = None
        sample_lr_raw = None
        sample_theta = None

        image_iter = tqdm(range(n_images), desc=f"Images@{density}", leave=False) if show_tqdm else range(n_images)
        for img_idx in image_iter:
            density_dir = os.path.join(cache_root, f"density_{density}")
            os.makedirs(density_dir, exist_ok=True)
            cache_path = os.path.join(density_dir, f"sample_{img_idx}.npz")
            if use_probe_cache:
                if not os.path.exists(cache_path):
                    raise FileNotFoundError(f"Missing cache file: {cache_path}")
                cached = np.load(cache_path)
                pred_mean = cached["pred_mean"]
                pred_std = cached["pred_std"]
                preds = cached["preds"]
                theta = cached["theta"]
                x = cached["x"]
                lr_raw = cached["lr_raw"] if "lr_raw" in cached else None
                hr_true = _make_kde_label(theta, size, label_upsample, label_sigma, label_scale, label_centering)
            else:
                nspots = int(density)
                spacing_px_val = sim_cfg.get("spacing_px", 4.0)
                if spacing_px_val is not None:
                    spacing_px_val = float(spacing_px_val)
                sigma_min = sim_cfg.get("sigma_min", None)
                sigma_max = sim_cfg.get("sigma_max", None)
                if sigma_min is not None and sigma_max is not None:
                    sigma_val = float(np.random.uniform(float(sigma_min), float(sigma_max)))
                else:
                    sigma_val = float(sim_cfg.get("sigma", 1.0))

                b0_min = sim_cfg.get("B0_min", None)
                b0_max = sim_cfg.get("B0_max", None)
                if b0_min is not None and b0_max is not None:
                    b0_val = float(np.random.uniform(float(b0_min), float(b0_max)))
                else:
                    b0_val = sim_cfg.get("B0", None)

                grf_sigma_min = sim_cfg.get("grf_sigma_min", None)
                grf_sigma_max = sim_cfg.get("grf_sigma_max", None)
                if grf_sigma_min is not None and grf_sigma_max is not None:
                    grf_sigma_val = float(np.random.uniform(float(grf_sigma_min), float(grf_sigma_max)))
                else:
                    grf_sigma_val = float(sim_cfg.get("grf_sigma", 0.0))

                grf_alpha_min = sim_cfg.get("grf_alpha_min", None)
                grf_alpha_max = sim_cfg.get("grf_alpha_max", None)
                if grf_alpha_min is not None and grf_alpha_max is not None:
                    grf_alpha_val = float(np.random.uniform(float(grf_alpha_min), float(grf_alpha_max)))
                else:
                    grf_alpha_val = float(sim_cfg.get("grf_alpha", 0.0))

                sim_kwargs = dict(
                    nspots=nspots,
                    sigma=sigma_val,
                    texp=float(sim_cfg.get("texp", 1.0)),
                    N0_min=float(sim_cfg.get("N0_min", 500.0)),
                    N0_max=float(sim_cfg.get("N0_max", 1000.0)),
                    eta=float(sim_cfg.get("eta", 1.0)),
                    gain=float(sim_cfg.get("gain", 1.0)),
                    B0=b0_val,
                    nframes=1,
                    offset=float(sim_cfg.get("offset", 100.0)),
                    var=float(sim_cfg.get("var", 5.0)),
                    halo_alpha=float(sim_cfg.get("halo_alpha", 0.0)),
                    halo_sigma=float(sim_cfg.get("halo_sigma", 0.0)),
                    grf_alpha=grf_alpha_val,
                    grf_sigma=grf_sigma_val,
                    grf_seed=sim_cfg.get("grf_seed", None),
                )
                if generator_cls is Nanoruler2D:
                    position_sigma_min = sim_cfg.get("position_sigma_min", None)
                    position_sigma_max = sim_cfg.get("position_sigma_max", None)
                    if position_sigma_min is not None and position_sigma_max is not None:
                        position_sigma_val = float(
                            np.random.uniform(float(position_sigma_min), float(position_sigma_max))
                        )
                    else:
                        position_sigma_val = float(sim_cfg.get("position_sigma", 0.0))
                    sim_kwargs.update(
                        spacing_px=spacing_px_val,
                        spacing_nm=sim_cfg.get("spacing_nm", None),
                        pixel_size_nm=sim_cfg.get("pixel_size_nm", None),
                        edgew=float(sim_cfg.get("edgew", 5.0)),
                        position_sigma=position_sigma_val,
                        pattern=sim_cfg.get("pattern", "uniform"),
                        parent_rate=sim_cfg.get("parent_rate", None),
                        parent_count=int(nspots),
                        children_sigma=float(sim_cfg.get("children_sigma", 1.0)),
                        children_min=int(sim_cfg.get("children_min", 0)),
                        children_pmf=sim_cfg.get("children_pmf", None),
                        burst_prob=sim_cfg.get("burst_prob", None),
                    )

                adu, _, theta = generator.forward(**sim_kwargs)
                if adu.ndim == 3:
                    adu = adu[0]
                lr_raw = adu.astype(np.float32)
                x = lr_raw
                if subtract_offset:
                    x = x - float(sim_cfg.get("offset", 0.0))
                if input_upsample > 1:
                    x = _upsample_image(x, input_upsample)
                if input_centering == "zscore":
                    x = _center_input_zscore(x)
                x_in = x[None, ..., None]

                hr_true = _make_kde_label(theta, size, label_upsample, label_sigma, label_scale, label_centering)

                preds = []
                out_shape = (1, size * label_upsample, size * label_upsample, 1)
                iter_loop = tqdm(range(n_iters), desc="DDPM", leave=False) if show_tqdm else range(n_iters)
                for _ in iter_loop:
                    pred, _, _ = ddpm_obtain_sr_img(
                        x_in,
                        generation_timesteps,
                        noise_model,
                        schedule_model,
                        mu_model,
                        out_shape,
                    )
                    pred = np.squeeze(pred)
                    preds.append(pred)
                preds = np.array(preds)
                pred_mean = np.mean(preds, axis=0)
                pred_std = np.std(preds, axis=0)

                if save_cache:
                    np.savez(
                        cache_path,
                        pred_mean=pred_mean,
                        pred_std=pred_std,
                        preds=preds,
                        theta=theta,
                        x=x,
                        lr_raw=lr_raw,
                    )

            # detection
            detect_frame = pred_mean if detect_on == "mean" else preds[0]
            fit_enabled = bool(detection_cfg.get("fit_enabled", True))
            spots = _detect_spots(detect_frame, detection_cfg)

            if spots.empty:
                pred_xy = np.zeros((0, 2))
            elif fit_enabled and "x_mle" in spots.columns and "y_mle" in spots.columns:
                pred_xy = np.vstack([spots["x_mle"].to_numpy(), spots["y_mle"].to_numpy()]).T
            else:
                pred_xy = np.vstack([spots["x"].to_numpy(), spots["y"].to_numpy()]).T
            coordsgt = theta.copy()
            coordsgt[0:2, :] *= label_upsample
            tol_val = float(metrics_cfg.get("tol", 5.0))
            all_x_err, all_y_err, all_label, all_n0, inter, union, fp, fn = errors2d(
                coordsgt,
                pred_xy,
                tol=tol_val,
            )
            precision, recall = _compute_precision_recall(int(inter), int(fp), int(fn))
            precision_list.append(precision)
            recall_list.append(recall)

            if img_idx == 0:
                sample_lr = x
                if lr_raw is None and input_upsample > 1:
                    target_shape = (x.shape[0] // input_upsample, x.shape[1] // input_upsample)
                    lr_raw = resize(x, target_shape, order=1, preserve_range=True, anti_aliasing=False).astype(np.float32)
                sample_lr_raw = lr_raw
                sample_hr_true = hr_true
                sample_hr_pred = pred_mean
                sample_mean = pred_mean
                sample_std = pred_std
                sample_preds = preds
                sample_detect_frame = detect_frame
                sample_spots = spots
                sample_theta = theta
                kde_spots = _detect_spots(hr_true, detection_cfg)
                sample_inset_coords = _pick_inset_coords(
                    kde_spots,
                    size * label_upsample,
                    int(probe_cfg.get("inset_hr_size", 60)),
                )

            # error records
            if all_x_err.size:
                for x_err, y_err, n0 in zip(all_x_err, all_y_err, all_n0):
                    error_records[str(density)].append(
                        {"x_err": float(x_err), "y_err": float(y_err), "N0": float(n0)}
                    )

            # Figure 3c stats (per-spot std over iterations)
            if preds is not None and preds.shape[0] > 1:
                coordsgt_iter = theta.copy()
                coordsgt_iter[0:2, :] *= label_upsample
                per_label_coords = {idx: [] for idx in range(coordsgt_iter.shape[1])}
                tol_val = float(metrics_cfg.get("tol", 5.0))
                for pred_frame in preds:
                    iter_spots = _detect_spots(pred_frame, detection_cfg)
                    if iter_spots.empty:
                        continue
                    if fit_enabled and "x_mle" in iter_spots.columns and "y_mle" in iter_spots.columns:
                        iter_xy = np.vstack([iter_spots["x_mle"].to_numpy(), iter_spots["y_mle"].to_numpy()]).T
                    else:
                        iter_xy = np.vstack([iter_spots["x"].to_numpy(), iter_spots["y"].to_numpy()]).T
                    if iter_xy.size == 0:
                        continue
                    x_err, y_err, labels, _, _, _, _, _ = errors2d(coordsgt_iter, iter_xy, tol=tol_val)
                    for dx, dy, label in zip(x_err, y_err, labels):
                        label_idx = int(label)
                        gt_x = coordsgt_iter[0, label_idx]
                        gt_y = coordsgt_iter[1, label_idx]
                        per_label_coords[label_idx].append([gt_x + dx, gt_y + dy])
                for coords in per_label_coords.values():
                    if len(coords) < 2:
                        continue
                    coords_arr = np.array(coords)
                    std_records[str(density)]["x"].append(float(np.std(coords_arr[:, 0])))
                    std_records[str(density)]["y"].append(float(np.std(coords_arr[:, 1])))

        density_metrics[str(density)] = (np.array(precision_list), np.array(recall_list))

        if sample_lr is not None:
            sample_by_density[str(density)] = {
                "lr": sample_lr,
                "lr_raw": sample_lr_raw,
                "hr_true": sample_hr_true,
                "hr_pred": sample_hr_pred,
                "mean": sample_mean,
                "std": sample_std,
                "preds": sample_preds,
                "detect_frame": sample_detect_frame,
                "spots": sample_spots,
                "theta": sample_theta,
                "inset_coords": sample_inset_coords if sample_inset_coords is not None else [(0, 0)],
            }

    # Figure 2a (LR/HR true/pred) with insets
    lr_inset_size = int(probe_cfg.get("inset_lr_size", 15))
    hr_inset_size = int(probe_cfg.get("inset_hr_size", 60))
    if "2a" in probes:
        fig, axes = plt.subplots(len(densities), 3, figsize=(9, 9))
        for row, density in enumerate(densities):
            sample = sample_by_density.get(str(density))
            if sample is None:
                continue
            lr = sample["lr"]
            lr_raw = sample.get("lr_raw")
            hr_true = sample["hr_true"]
            hr_pred = sample["hr_pred"]
            hr_inset_coords = sample.get("inset_coords", [(0, 0)])
            hr_inset_coords = hr_inset_coords[min(row, len(hr_inset_coords) - 1)]
            if lr_raw is None:
                lr_raw = lr
            lr_h, lr_w = lr_raw.shape[:2]
            lr_x0 = max(min(hr_inset_coords[0] // label_upsample, lr_h - lr_inset_size), 0)
            lr_y0 = max(min(hr_inset_coords[1] // label_upsample, lr_w - lr_inset_size), 0)
            lr_inset_coords = (lr_x0, lr_y0)
            _imshow_scaled(axes[row, 0], lr_raw, contrast_low, contrast_high)
            axes[row, 0].set_ylabel(rf"$\rho$={density}", fontsize=16)
            axes[row, 0].set_xticks([])
            axes[row, 0].set_yticks([])
            inset = axes[row, 0].inset_axes([0.65, 0.65, 0.4, 0.4])
            _imshow_scaled(
                inset,
                lr_raw[lr_inset_coords[0]:lr_inset_coords[0] + lr_inset_size,
                      lr_inset_coords[1]:lr_inset_coords[1] + lr_inset_size],
                inset_contrast_low,
                inset_contrast_high,
                interpolation="nearest",
            )
            inset.set_xticks([])
            inset.set_yticks([])
            for spine in inset.spines.values():
                spine.set_color("red")
                spine.set_linewidth(1)
            _draw_inset_box(axes[row, 0], lr_inset_coords[0], lr_inset_coords[1], lr_inset_size)

            _imshow_scaled(axes[row, 1], hr_true, contrast_low, contrast_high)
            axes[row, 1].set_xticks([])
            axes[row, 1].set_yticks([])
            inset = axes[row, 1].inset_axes([0.65, 0.65, 0.4, 0.4])
            _imshow_scaled(
                inset,
                hr_true[hr_inset_coords[0]:hr_inset_coords[0] + hr_inset_size,
                    hr_inset_coords[1]:hr_inset_coords[1] + hr_inset_size],
                inset_contrast_low,
                inset_contrast_high,
                interpolation="nearest",
            )
            inset.set_xticks([])
            inset.set_yticks([])
            for spine in inset.spines.values():
                spine.set_color("red")
                spine.set_linewidth(1)
            _draw_inset_box(axes[row, 1], hr_inset_coords[0], hr_inset_coords[1], hr_inset_size)

            _imshow_scaled(axes[row, 2], hr_pred, contrast_low, contrast_high)
            axes[row, 2].set_xticks([])
            axes[row, 2].set_yticks([])
            inset = axes[row, 2].inset_axes([0.65, 0.65, 0.4, 0.4])
            _imshow_scaled(
                inset,
                hr_pred[hr_inset_coords[0]:hr_inset_coords[0] + hr_inset_size,
                    hr_inset_coords[1]:hr_inset_coords[1] + hr_inset_size],
                inset_contrast_low,
                inset_contrast_high,
                interpolation="nearest",
            )
            inset.set_xticks([])
            inset.set_yticks([])
            for spine in inset.spines.values():
                spine.set_color("red")
                spine.set_linewidth(1)
            _draw_inset_box(axes[row, 2], hr_inset_coords[0], hr_inset_coords[1], hr_inset_size)

        axes[0, 0].set_title(r"$x$", fontsize=16)
        axes[0, 1].set_title(r"$y_0$", fontsize=16)
        axes[0, 2].set_title(r"$\hat{y}$", fontsize=16)
        plt.tight_layout()
        _save_fig(fig, output_dir, "probe_2a.png")

    # Figure 3a (mean/std)
    fig3_density = str(probe_cfg.get("fig3_density", densities[0]))
    sample = sample_by_density.get(fig3_density)
    if sample is not None and "3a" in probes:
        fig, ax = plt.subplots(2, 2, figsize=(6, 5))
        lr = sample["lr"]
        lr_raw = sample.get("lr_raw")
        hr_true = sample["hr_true"]
        mean = sample["mean"]
        std = sample["std"]
        inset_coords = sample.get("inset_coords") or [probe_cfg.get("fig3_inset", [50, 50])]
        inset_coords = inset_coords[0]
        hr_inset_coords = (int(inset_coords[0]), int(inset_coords[1]))
        lr_inset_coords = (hr_inset_coords[0] // label_upsample, hr_inset_coords[1] // label_upsample)
        lr_h, lr_w = lr.shape[:2]
        hr_h, hr_w = hr_true.shape[:2]
        lr_x0 = max(min(lr_inset_coords[0], lr_h - lr_inset_size), 0)
        lr_y0 = max(min(lr_inset_coords[1], lr_w - lr_inset_size), 0)
        hr_x0 = max(min(hr_inset_coords[0], hr_h - hr_inset_size), 0)
        hr_y0 = max(min(hr_inset_coords[1], hr_w - hr_inset_size), 0)
        lr_inset = (slice(lr_x0, lr_x0 + lr_inset_size), slice(lr_y0, lr_y0 + lr_inset_size))
        hr_inset = (slice(hr_x0, hr_x0 + hr_inset_size), slice(hr_y0, hr_y0 + hr_inset_size))

        if lr_raw is None:
            lr_raw = lr
        _imshow_scaled(ax[0, 0], lr_raw, contrast_low, contrast_high)
        ax[0, 0].set_xticks([])
        ax[0, 0].set_yticks([])
        ax[0, 0].set_title(r"$x$")
        inset = ax[0, 0].inset_axes([0.65, 0.65, 0.4, 0.4])
        _imshow_scaled(inset, lr_raw[lr_inset], inset_contrast_low, inset_contrast_high)
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_color("red")
            spine.set_linewidth(1)
        _draw_inset_box(ax[0, 0], lr_x0, lr_y0, lr_inset_size)

        _imshow_scaled(ax[0, 1], hr_true, contrast_low, contrast_high)
        ax[0, 1].set_xticks([])
        ax[0, 1].set_yticks([])
        ax[0, 1].set_title(r"$y_0$")
        inset = ax[0, 1].inset_axes([0.65, 0.65, 0.4, 0.4])
        _imshow_scaled(inset, hr_true[hr_inset], inset_contrast_low, inset_contrast_high)
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_color("red")
            spine.set_linewidth(1)
        _draw_inset_box(ax[0, 1], hr_x0, hr_y0, hr_inset_size)

        _imshow_scaled(ax[1, 0], mean, contrast_low, contrast_high)
        ax[1, 0].set_xticks([])
        ax[1, 0].set_yticks([])
        ax[1, 0].set_title(r"$\langle \hat{y} \rangle$")
        inset = ax[1, 0].inset_axes([0.65, 0.65, 0.4, 0.4])
        _imshow_scaled(inset, mean[hr_inset], inset_contrast_low, inset_contrast_high)
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_color("red")
            spine.set_linewidth(1)
        _draw_inset_box(ax[1, 0], hr_x0, hr_y0, hr_inset_size)

        _imshow_scaled(ax[1, 1], std, contrast_low, contrast_high)
        std_thresh = float(np.percentile(std, std_overlay_percentile))
        std_mask = std > std_thresh
        ax[1, 1].imshow(std_mask, cmap="bwr", alpha=0.35)
        ax[1, 1].set_xticks([])
        ax[1, 1].set_yticks([])
        ax[1, 1].set_title(r"$\sigma$")
        inset = ax[1, 1].inset_axes([0.65, 0.65, 0.4, 0.4])
        _imshow_scaled(inset, std[hr_inset], inset_contrast_low, inset_contrast_high)
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_color("red")
            spine.set_linewidth(1)
        _draw_inset_box(ax[1, 1], hr_x0, hr_y0, hr_inset_size)

        lr_mean = resize(mean, lr_raw.shape, order=1, preserve_range=True, anti_aliasing=False).astype(np.float32)
        lr_sigma = float(sim_cfg.get("sigma", 1.0))
        lr_mean = gaussian(lr_mean, sigma=lr_sigma, preserve_range=True)
        lr_flat = lr_raw.reshape(-1)
        mean_flat = lr_mean.reshape(-1)
        if np.std(lr_flat) > 0 and np.std(mean_flat) > 0:
            lr_corr = float(np.corrcoef(lr_flat, mean_flat)[0, 1])
        else:
            lr_corr = float("nan")

        plt.tight_layout()
        _save_fig(fig, output_dir, "probe_3a.png")

        fig_lr, ax_lr = plt.subplots(1, 2, figsize=(6, 3))
        _imshow_scaled(ax_lr[0], lr_raw, contrast_low, contrast_high)
        ax_lr[0].set_xticks([])
        ax_lr[0].set_yticks([])
        ax_lr[0].set_title(r"$x_{\mathrm{LR}}$")
        inset = ax_lr[0].inset_axes([0.65, 0.65, 0.4, 0.4])
        _imshow_scaled(inset, lr_raw[lr_inset], inset_contrast_low, inset_contrast_high)
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_color("red")
            spine.set_linewidth(1)
        _draw_inset_box(ax_lr[0], lr_x0, lr_y0, lr_inset_size)

        _imshow_scaled(ax_lr[1], lr_mean, contrast_low, contrast_high)
        ax_lr[1].set_xticks([])
        ax_lr[1].set_yticks([])
        if np.isfinite(lr_corr):
            ax_lr[1].set_title(rf"$\langle \hat{{y}} \rangle_{{\mathrm{{LR}}}}$ (NCC={lr_corr:.2f})")
        else:
            ax_lr[1].set_title(r"$\langle \hat{y} \rangle_{\mathrm{LR}}$")
        inset = ax_lr[1].inset_axes([0.65, 0.65, 0.4, 0.4])
        _imshow_scaled(inset, lr_mean[lr_inset], inset_contrast_low, inset_contrast_high)
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_color("red")
            spine.set_linewidth(1)
        _draw_inset_box(ax_lr[1], lr_x0, lr_y0, lr_inset_size)
        plt.tight_layout()
        if "corr" in probes:
            _save_fig(fig_lr, output_dir, "probe_corr.png")
        else:
            plt.close(fig_lr)

        detect_frame = sample.get("detect_frame")
        spots = sample.get("spots")
        if "detect_overlay" in probes and detect_frame is not None and spots is not None and not spots.empty:
            fig, ax = plt.subplots(figsize=(4, 4))
            _imshow_scaled(ax, detect_frame, contrast_low, contrast_high)
            if "x_mle" in spots.columns and "y_mle" in spots.columns:
                x_coords = spots["x_mle"]
                y_coords = spots["y_mle"]
            else:
                x_coords = spots["x"]
                y_coords = spots["y"]
            ax.scatter(y_coords, x_coords, c="red", s=12, alpha=0.8)
            ax.set_xticks([])
            ax.set_yticks([])
            _save_fig(fig, output_dir, "probe_detect_overlay.png")

    # Figure 3b (sample predictions)
    if sample is not None and sample["preds"] is not None and "3b" in probes:
        samples = [int(i) for i in probe_cfg.get("fig3_samples", [0, 1, 4, 5])]
        pred_count = int(sample["preds"].shape[0])
        desired_count = min(len(samples), pred_count)
        safe_samples = [i for i in samples if 0 <= i < pred_count]
        if len(safe_samples) < desired_count:
            for i in range(pred_count):
                if i not in safe_samples:
                    safe_samples.append(i)
                if len(safe_samples) >= desired_count:
                    break
        if not safe_samples:
            safe_samples = list(range(min(4, pred_count)))
        fig, ax = plt.subplots(1, len(safe_samples), figsize=(2 * len(safe_samples), 2))
        if len(safe_samples) == 1:
            ax = [ax]
        for idx, i in enumerate(safe_samples):
            _imshow_scaled(ax[idx], sample["preds"][i], contrast_low, contrast_high)
            ax[idx].set_title(rf"$\hat{{y}}_{{0,{i}}}$", fontsize=12)
            ax[idx].set_xticks([])
            ax[idx].set_yticks([])
        plt.tight_layout()
        _save_fig(fig, output_dir, "probe_3b.png")

    # Figure 3b map (posterior clustering with GT overlay)
    if "3b_map" in probes and sample is not None and sample.get("preds") is not None:
        lr_raw = sample.get("lr_raw")
        if lr_raw is None:
            lr_raw = sample.get("lr")
        theta = sample.get("theta")
        if lr_raw is not None and theta is not None:
            frame_hr = sample["preds"]
            n_iters_local = frame_hr.shape[0]
            lr_scale = float(frame_hr.shape[1]) / float(lr_raw.shape[0]) if lr_raw.shape[0] else 1.0
            if lr_scale < 1.5:
                lr_scale = 1.0
            detections_per_iter = []
            fit_enabled = bool(detection_cfg.get("fit_enabled", True))
            for c_idx in range(n_iters_local):
                det = _detect_spots(frame_hr[c_idx], detection_cfg)
                if det.empty:
                    detections_per_iter.append(np.empty((0, 2), dtype=float))
                elif fit_enabled and "x_mle" in det.columns and "y_mle" in det.columns:
                    detections_per_iter.append(det[["x_mle", "y_mle"]].to_numpy(dtype=float))
                else:
                    detections_per_iter.append(det[["x", "y"]].to_numpy(dtype=float))

            fig_map = plt.figure(figsize=(18, 10))
            grid = fig_map.add_gridspec(2, 2, height_ratios=[1, 1.1], width_ratios=[1, 1.2])
            ax_cluster = fig_map.add_subplot(grid[0, 0])
            _imshow_scaled(ax_cluster, lr_raw, contrast_low, contrast_high)
            ax_cluster.set_title("3b_map: posterior clusters", fontsize=10)
            ax_cluster.set_xticks([])
            ax_cluster.set_yticks([])
            ax_cluster.set_aspect("equal", adjustable="box")

            gt_lr = theta[:2, :].T

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
                clusters = _cluster_points(lr_points_arr, map_cluster_radius_lr)
                cluster_colors = plt.cm.tab20(np.linspace(0, 1, max(len(clusters), 1)))
                cluster_centers = []
                cluster_means = []
                for c_idx, cluster in enumerate(clusters):
                    pts = lr_points_arr[cluster]
                    center = np.array([float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))])
                    cluster_centers.append(center)
                    ax_cluster.scatter(
                        pts[:, 1],
                        pts[:, 0],
                        s=15,
                        color=cluster_colors[c_idx],
                        alpha=0.9,
                        zorder=5,
                    )
                    counts = np.zeros(n_iters_local, dtype=int)
                    for pt_idx in cluster:
                        counts[lr_point_iters[pt_idx]] += 1
                    n_mode = int(np.bincount(counts).argmax())
                    n_mode = max(1, min(n_mode, len(pts)))
                    means, _, _ = _gmm_fit(pts, n_mode)
                    if len(means):
                        cluster_means.append(means)
                        ax_cluster.scatter(
                            means[:, 1],
                            means[:, 0],
                            s=45,
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
                        subplot_spec=grid[0, 1],
                        wspace=0.3,
                        hspace=0.4,
                    )
                    gmm_grid = GridSpecFromSubplotSpec(
                        n_rows,
                        n_cols,
                        subplot_spec=grid[1, :],
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
                        count_hist = np.bincount(counts)
                        ax_hist.bar(
                            np.arange(len(count_hist)),
                            count_hist,
                            color=cluster_colors[idx],
                            edgecolor="none",
                        )
                        ax_hist.set_title(f"C{idx+1}", fontsize=8)
                        pts = lr_points_arr[cluster]
                        if pts.size == 0:
                            ax_gmm.axis("off")
                            continue
                        n_mode = int(np.bincount(counts).argmax())
                        n_mode = max(1, min(n_mode, len(pts)))
                        means, covs, ll = _gmm_fit(pts, n_mode)
                        patch_size_lr = max(3, map_patch_size_lr)
                        center = cluster_centers[idx]
                        patch = _extract_patch(lr_raw, center[0], center[1], patch_size_lr)
                        _imshow_scaled(ax_gmm, patch, contrast_low, contrast_high)
                        ax_gmm.set_xlim(0, patch_size_lr)
                        ax_gmm.set_ylim(patch_size_lr, 0)
                        ax_gmm.set_aspect("equal", adjustable="box")

                        x0 = center[0] - patch_size_lr / 2.0
                        y0 = center[1] - patch_size_lr / 2.0
                        pts_local = pts - np.array([x0, y0])
                        ax_gmm.scatter(
                            pts_local[:, 1],
                            pts_local[:, 0],
                            s=15,
                            color=cluster_colors[idx],
                            alpha=0.8,
                        )

                        gt_local = gt_lr - np.array([x0, y0])
                        gt_mask = (
                            (gt_local[:, 0] >= 0)
                            & (gt_local[:, 0] <= patch_size_lr)
                            & (gt_local[:, 1] >= 0)
                            & (gt_local[:, 1] <= patch_size_lr)
                        )
                        if np.any(gt_mask):
                            ax_gmm.scatter(
                                gt_local[gt_mask, 1],
                                gt_local[gt_mask, 0],
                                s=28,
                                color="white",
                                marker="x",
                                linewidths=1.2,
                            )

                        if len(means):
                            grid_n = 60
                            xs = np.linspace(0, patch_size_lr, grid_n)
                            ys = np.linspace(0, patch_size_lr, grid_n)
                            xx, yy = np.meshgrid(xs, ys)
                            for k_idx in range(len(means)):
                                var = covs[k_idx]
                                mean_local = means[k_idx] - np.array([x0, y0])
                                diff_x = (xx - mean_local[0])
                                diff_y = (yy - mean_local[1])
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
                        ax_gmm.set_title(f"C{idx+1}, N={n_mode}, LL={ll:.1f}", fontsize=8)

                if gt_lr.size:
                    ax_cluster.scatter(
                        gt_lr[:, 1],
                        gt_lr[:, 0],
                        s=30,
                        color="blue",
                        marker="x",
                        linewidths=1.5,
                        zorder=7,
                    )

            fig_map.tight_layout(pad=0.2)
            _save_fig(fig_map, output_dir, "probe_3b_map.png")

    # Figure 3c (std histograms)
    if "3c" in probes:
        fig3c_bins = int(probe_cfg.get("fig3c_bins", 20))
        pixel_size = float(metrics_cfg.get("pixel_size_nm", 1.0))
        datasets = []
        for density in densities:
            xstd = np.array(std_records[str(density)]["x"], dtype=float) * pixel_size
            ystd = np.array(std_records[str(density)]["y"], dtype=float) * pixel_size
            xstd = xstd[xstd > 0]
            ystd = ystd[ystd > 0]
            datasets.append((xstd, ystd))
        all_values = np.concatenate([ds for pair in datasets for ds in pair if ds.size]) if datasets else np.array([])
        if all_values.size:
            x_min, x_max = float(np.min(all_values)), float(np.max(all_values))
            fig, axes = plt.subplots(2, len(densities), figsize=(3 * len(densities), 4), sharey='row', sharex='col')
            if len(densities) == 1:
                axes = np.array([[axes[0]], [axes[1]]])
            colors = ["red", "blue", "gray"]
            for i, (xstd, ystd) in enumerate(datasets):
                color = colors[i % len(colors)]
                for ax, std, label in zip(
                    axes[:, i],
                    [xstd, ystd],
                    [r"$\sqrt{\mathrm{Var}(\theta_u)}$", r"$\sqrt{\mathrm{Var}(\theta_v)}$"],
                ):
                    if std.size == 0:
                        ax.axis("off")
                        continue
                    hist, bins = np.histogram(std, bins=fig3c_bins, density=True)
                    ax.bar(bins[:-1], hist, width=np.diff(bins), color=color, edgecolor="black")
                    mean_value = float(np.mean(std))
                    ax.axvline(mean_value, color="black", linestyle="--", label=rf"$\mu={mean_value:.2f}$")
                    if ax.get_subplotspec().is_first_row():
                        ax.set_title(rf"$\rho={densities[i]}$", fontsize=14)
                    ax.set_xlabel(label + r"\ (nm)", fontsize=12)
                    ax.set_xlim(x_min, x_max)
                    ax.legend(frameon=False, fontsize=10)
            axes[0, 0].set_ylabel(r"$\mathrm{Probability}$", fontsize=12)
            axes[1, 0].set_ylabel(r"$\mathrm{Probability}$", fontsize=12)
            plt.tight_layout()
            _save_fig(fig, output_dir, "probe_3c.png")

    # Detection overlays per density
    if "detect_by_density" in probes:
        fig, axes = plt.subplots(1, len(densities), figsize=(4 * len(densities), 4))
        if len(densities) == 1:
            axes = [axes]
        for idx, density in enumerate(densities):
            ax = axes[idx]
            sample = sample_by_density.get(str(density))
            if sample is None:
                ax.axis("off")
                continue
            detect_frame = sample.get("detect_frame")
            spots = sample.get("spots")
            if detect_frame is None or spots is None or spots.empty:
                ax.axis("off")
                continue
            _imshow_scaled(ax, detect_frame, contrast_low, contrast_high)
            if "x_mle" in spots.columns and "y_mle" in spots.columns:
                x_coords = spots["x_mle"]
                y_coords = spots["y_mle"]
            else:
                x_coords = spots["x"]
                y_coords = spots["y"]
            ax.scatter(y_coords, x_coords, c="red", s=12, alpha=0.8)
            ax.set_title(rf"$\rho={density}$", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        _save_fig(fig, output_dir, "probe_detect_by_density.png")

    # Figure 2bc (errors vs photons)
    if "2bc" in probes:
        pixel_size = float(metrics_cfg.get("pixel_size_nm", 1.0))
        bins = np.array(metrics_cfg.get("error_bins", list(np.arange(500, 1000, 100))))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
        for density, color in zip(densities, ["red", "blue", "gray"]):
            errs = error_records[str(density)]
            if not errs:
                continue
            n0_vals = np.array([e["N0"] for e in errs])
            x_err = np.array([e["x_err"] for e in errs]) * pixel_size
            y_err = np.array([e["y_err"] for e in errs]) * pixel_size
            def bin_std(values):
                stds = []
                for i in range(len(bins) - 1):
                    mask = (n0_vals >= bins[i]) & (n0_vals < bins[i + 1])
                    if not np.any(mask):
                        stds.append(np.nan)
                    else:
                        stds.append(float(np.std(values[mask])))
                return np.array(stds)
            x_std = bin_std(x_err)
            y_std = bin_std(y_err)
            x_mask = np.isfinite(x_std)
            y_mask = np.isfinite(y_std)
            if np.any(x_mask):
                ax1.plot(bins[:-1][x_mask], x_std[x_mask], "x", color=color, label=rf"$\rho = {density}$")
            if np.any(y_mask):
                ax2.plot(bins[:-1][y_mask], y_std[y_mask], "x", color=color, label=rf"$\rho = {density}$")
        ax1.set_xscale("log")
        ax2.set_xscale("log")
        ax1.set_xlabel(r"$\mathrm{Photons}$", fontsize=16)
        ax2.set_xlabel(r"$\mathrm{Photons}$", fontsize=16)
        ax1.set_ylabel(r"$\sigma_u$ (nm)", fontsize=16)
        ax2.set_ylabel(r"$\sigma_v$ (nm)", fontsize=16)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False, fontsize=12)
        ax1.grid()
        ax2.grid()
        fig.tight_layout()
        _save_fig(fig, output_dir, "probe_2bc.png")

    # Figure 2de (precision/recall)
    if "2de" in probes:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
        density_values = [int(d) for d in densities]
        precision_means = []
        precision_stds = []
        recall_means = []
        recall_stds = []
        for density in densities:
            precision, recall = density_metrics[str(density)]
            precision_means.append(np.mean(precision) if precision.size else 0.0)
            precision_stds.append(np.std(precision) if precision.size else 0.0)
            recall_means.append(np.mean(recall) if recall.size else 0.0)
            recall_stds.append(np.std(recall) if recall.size else 0.0)
        ax1.errorbar(density_values, precision_means, yerr=precision_stds, fmt='x', capsize=5, capthick=1, color='black')
        ax2.errorbar(density_values, recall_means, yerr=recall_stds, fmt='x', capsize=5, capthick=1, color='black')
        ax1.set_xlabel(r'$\rho$', fontsize=16)
        ax2.set_xlabel(r'$\rho$', fontsize=16)
        ax1.set_ylabel(r"$\mathrm{Precision}$", fontsize=16)
        ax2.set_ylabel(r"$\mathrm{Recall}$", fontsize=16)
        ax1.set_xticks(density_values)
        ax2.set_xticks(density_values)
        ax1.grid()
        ax2.grid()
        fig.tight_layout()
        _save_fig(fig, output_dir, "probe_2de.png")


if __name__ == "__main__":
    main()
