import argparse
import math
import os
from typing import Dict, List, Tuple
from skimage.io import imread, imsave
from tifffile import imread as tiff_read, imwrite
from skimage.transform import resize
from skimage.measure import profile_line
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpecFromSubplotSpec
import yaml

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from cvdm.configs_pkg.utils import (
    create_data_config,
    create_eval_config,
    create_model_config,
    load_config_from_yaml,
)
from cvdm.models.joint_model import instantiate_cvdm
from cvdm.utils.inference_utils import ddpm_obtain_sr_img
from cvdm.psf.mle2d import PipelineMLE2D
from cvdm.utils.training_utils import prepare_dataset, prepare_model_input


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path", help="Path to the configuration file", required=True
    )

    args = parser.parse_args()

    print("Num CPUs Available: ", len(tf.config.list_physical_devices("CPU")))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

    config = load_config_from_yaml(args.config_path)
    model_config = create_model_config(config)
    print(model_config)
    task = config.get("task")
    assert task in [
        "SMLM",
        "biosr_sr",
        "imagenet_sr",
        "biosr_phase",
        "imagenet_phase",
        "hcoco_phase",
        "other",
    ], "Possible tasks are: biosr_sr, imagenet_sr, biosr_phase, imagenet_phase, hcoco_phase, other"

    dataset_names = config.get("datasets")
    test_cfg = config.get("test", {})
    test_prefixes = test_cfg.get("prefixes")
    dataset_items: List[Tuple[str, str, str]]
    if test_prefixes:
        data_dir = test_cfg.get("data_dir")
        results_dir = test_cfg.get("results_dir")
        if not data_dir or not results_dir:
            raise ValueError("test.data_dir and test.results_dir must be set when using test.prefixes.")
        dataset_items = [
            (name, os.path.join(data_dir, name), os.path.join(results_dir, name))
            for name in test_prefixes
        ]
    elif dataset_names:
        data_base_path = config.get("data", {}).get("dataset_base_path")
        eval_base_path = config.get("eval", {}).get("output_base_path")
        if not data_base_path or not eval_base_path:
            raise ValueError(
                "When using 'datasets', set data.dataset_base_path and eval.output_base_path in config."
            )
        dataset_items = [
            (
                name,
                os.path.join(data_base_path, name),
                os.path.join(eval_base_path, name),
            )
            for name in dataset_names
        ]
    else:
        dataset_items = [(None, config["data"]["dataset_path"], config["eval"]["output_path"])]

    diff_inp = model_config.diff_inp

    for dataset_idx, (dataset_name, dataset_path, output_path) in enumerate(dataset_items, start=1):
        dataset_label = dataset_name or "single"
        print(f"\n=== Evaluating dataset {dataset_idx}/{len(dataset_items)}: {dataset_label} ===")

        per_config = dict(config)
        if "data" in per_config:
            per_config["data"] = dict(per_config["data"])
        if "eval" in per_config:
            per_config["eval"] = dict(per_config["eval"])
        if "test" in per_config:
            per_config["test"] = dict(per_config["test"])
        if "data" in per_config:
            per_config["data"]["dataset_path"] = dataset_path
            per_config["data"].pop("dataset_base_path", None)
        if "eval" in per_config:
            per_config["eval"]["output_path"] = output_path
            per_config["eval"].pop("output_base_path", None)
        data_config = create_data_config(per_config) if not test_prefixes else None
        eval_config = create_eval_config(per_config)
        generation_timesteps = eval_config.generation_timesteps

        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "config.yaml"), "w", encoding="utf-8") as handle:
            yaml.safe_dump(per_config, handle, sort_keys=False)
        if test_prefixes:
            _run_experimental_stack(
                dataset_path,
                output_path,
                model_config,
                generation_timesteps,
                eval_config.n_iters,
                test_cfg,
            )
        else:
            print("Creating model...")
            dataset, x_shape, y_shape = prepare_dataset(task, data_config, training=False)
            noise_model, joint_model, schedule_model, mu_model = instantiate_cvdm(
                lr=0.0,
                generation_timesteps=generation_timesteps,
                cond_shape=x_shape,
                out_shape=y_shape,
                model_config=model_config,
            )
            if model_config.load_weights is not None:
                joint_model.load_weights(model_config.load_weights)
            if model_config.load_mu_weights is not None and mu_model is not None:
                mu_model.load_weights(model_config.load_mu_weights)
            print("Getting data...")
            batch_size = data_config.batch_size
            dataset = dataset.batch(batch_size, drop_remainder=True)
            step = 0
            total_batches = dataset.cardinality().numpy() if hasattr(dataset, 'cardinality') else None
            print(f"Starting evaluation loop. Total batches: {total_batches if total_batches is not None else 'unknown'}")
            batch_iter = tqdm(
                enumerate(dataset),
                total=None if total_batches in (None, -2) else int(total_batches),
                desc=f"Batches ({dataset_label})",
            )
            for batch_idx, batch in batch_iter:
                batch_x, batch_y = batch
                print(f"Processing batch {batch_idx+1}{f' / {total_batches}' if total_batches is not None else ''} (step={step})")
                model_input = prepare_model_input(batch_x, batch_y, diff_inp=diff_inp)
                joint_model.evaluate(model_input, np.zeros_like(batch_y), verbose=0)
                print("Saving at: " + output_path)
                n_iters = eval_config.n_iters
                print(f"Batch shape: {batch_x.shape}")
                for sample in range(n_iters):
                    print(f"  Saving sample {sample+1} of {n_iters} in batch {batch_idx+1}")
                    pred_diff, _, _ = ddpm_obtain_sr_img(
                        batch_x,
                        generation_timesteps,
                        noise_model,
                        schedule_model,
                        mu_model,
                        batch_y.shape,
                        store_schedule=False,
                        show_tqdm=True,
                    )
                    pred_diff = np.clip(pred_diff, -1, 1)
                    imsave(output_path + f"/z-{step}-{sample}.tif", np.squeeze(pred_diff))
                    imsave(output_path + f"/x-{step}-{sample}.tif", np.squeeze(batch_x))
                    imsave(output_path + f"/y-{step}-{sample}.tif", np.squeeze(batch_y))
                step += 1


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

        Nk = resp.sum(axis=0) + 1e-8
        weights = Nk / float(n)
        means = (resp.T @ points) / Nk[:, None]
        for idx in range(k):
            diff = points - means[idx]
            covs[idx] = (resp[:, idx][:, None] * diff ** 2).sum(axis=0) / Nk[idx] + 1e-3

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


def _run_experimental_stack(
    dataset_path: str,
    output_path: str,
    model_config,
    generation_timesteps: int,
    n_iters: int,
    test_cfg: Dict,
) -> None:
    stack_name = test_cfg.get("stack_name", "lr-1x.tif")
    stack_path = os.path.join(dataset_path, stack_name)
    n_frames = test_cfg.get("n_frames")
    probes = [p.lower() for p in test_cfg.get("probes", ["2a", "3a"])]
    skip_inference = bool(test_cfg.get("skip_inference", False))
    fig_root = os.path.join(output_path, "figures")
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

    if skip_inference:
        x_stack_path = os.path.join(output_path, "x_stack.tif")
        y_stack_path = os.path.join(output_path, "y_stack.tif")
        z_stack_path = os.path.join(output_path, "z_stack.tif")
        if not (os.path.exists(x_stack_path) and os.path.exists(y_stack_path) and os.path.exists(z_stack_path)):
            raise FileNotFoundError("skip_inference requires x_stack.tif, y_stack.tif, and z_stack.tif")
        x_stack_loaded = tiff_read(x_stack_path)
        y_stack_loaded = tiff_read(y_stack_path)
        z_stack_loaded = tiff_read(z_stack_path)
        if x_stack_loaded.ndim == 2:
            x_stack_loaded = x_stack_loaded[None, ...]
        if y_stack_loaded.ndim == 2:
            y_stack_loaded = y_stack_loaded[None, ...]
        if z_stack_loaded.ndim == 3:
            z_stack_loaded = z_stack_loaded[None, ...]
        lr_stack = x_stack_loaded
        if z_stack_loaded.size:
            debug_path = os.path.join(output_path, "z_stack_debug_frame0_iter0.tif")
            imwrite(debug_path, z_stack_loaded[0, 0].astype(np.float32))
        if n_frames:
            lr_stack = lr_stack[: int(n_frames)]
            y_stack_loaded = y_stack_loaded[: int(n_frames)]
            z_stack_loaded = z_stack_loaded[: int(n_frames)]
    else:
        lr_stack = imread(stack_path)
        if lr_stack.ndim == 2:
            lr_stack = lr_stack[None, ...]
        if n_frames:
            lr_stack = lr_stack[: int(n_frames)]

    input_upsample = int(test_cfg.get("input_upsample", 4))
    subtract_offset = bool(test_cfg.get("subtract_offset", False))
    offset = float(test_cfg.get("offset", 0.0))
    input_centering = test_cfg.get("input_centering", "none")
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
    spot_count = int(test_cfg.get("spot_count", 5))
    spot_patch_size = int(test_cfg.get("spot_patch_size", 16))
    spot_seed = int(test_cfg.get("spot_seed", 0))

    noise_model = schedule_model = mu_model = None
    joint_model = None
    model_ready = False
    x_stack = []
    y_stack = []
    z_stack = []
    rng = np.random.default_rng(nanoruler_rand_seed)
    spot_rng = np.random.default_rng(spot_seed)
    for frame_idx, lr_raw in enumerate(tqdm(lr_stack, desc="Frames")):
        lr_raw = lr_raw.astype(np.float32)
        x = lr_raw
        if subtract_offset:
            x = x - offset
        if skip_inference:
            x_up = y_stack_loaded[frame_idx].astype(np.float32)
        else:
            if input_upsample > 1:
                x_up = resize(
                    x,
                    (x.shape[0] * input_upsample, x.shape[1] * input_upsample),
                    order=1,
                    preserve_range=True,
                    anti_aliasing=False,
                ).astype(np.float32)
            else:
                x_up = x
        if not skip_inference and input_centering == "zscore":
            mean = float(np.mean(x_up))
            std = float(np.std(x_up))
            if std > 0:
                x_up = (x_up - mean) / std
        if skip_inference:
            preds = z_stack_loaded[frame_idx].astype(np.float32)
        else:
            x_in = x_up[None, ..., None]
            if not model_ready:
                cond_shape = tf.TensorShape((x_up.shape[0], x_up.shape[1], 1))
                out_shape = tf.TensorShape((x_up.shape[0], x_up.shape[1], 1))
                noise_model, joint_model, schedule_model, mu_model = instantiate_cvdm(
                    lr=0.0,
                    generation_timesteps=generation_timesteps,
                    cond_shape=cond_shape,
                    out_shape=out_shape,
                    model_config=model_config,
                )
                if model_config.load_weights is not None:
                    joint_model.load_weights(model_config.load_weights)
                if model_config.load_mu_weights is not None and mu_model is not None:
                    mu_model.load_weights(model_config.load_mu_weights)
                model_ready = True

            preds = []
            for _ in range(n_iters):
                pred, _, _ = ddpm_obtain_sr_img(
                    x_in,
                    generation_timesteps,
                    noise_model,
                    schedule_model,
                    mu_model,
                    (1, x_up.shape[0], x_up.shape[1], 1),
                    store_schedule=False,
                    show_tqdm=False,
                )
                preds.append(np.squeeze(pred))
            preds = np.array(preds)
        pred_mean = np.mean(preds, axis=0)
        pred_std = np.std(preds, axis=0)

        if not skip_inference:
            x_stack.append(lr_raw)
            y_stack.append(x_up)
            z_stack.append(preds)

        hr_x0, hr_y0 = _pick_center_inset_coords(pred_mean, inset_hr_size, inset_threshold)
        lr_x0 = max(min(hr_x0 // input_upsample, lr_raw.shape[0] - inset_lr_size), 0)
        lr_y0 = max(min(hr_y0 // input_upsample, lr_raw.shape[1] - inset_lr_size), 0)

        hr_inset = (slice(hr_x0, hr_x0 + inset_hr_size), slice(hr_y0, hr_y0 + inset_hr_size))
        lr_inset = (slice(lr_x0, lr_x0 + inset_lr_size), slice(lr_y0, lr_y0 + inset_lr_size))

        if "2a" in probes:
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
            fig2a.savefig(
                os.path.join(probe_dirs["2a"], f"probe_2a_frame-{frame_idx:04d}.png"),
                dpi=200,
            )
            plt.close(fig2a)

        if "3a" in probes:
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
            fig3a.savefig(
                os.path.join(probe_dirs["3a"], f"probe_3a_frame-{frame_idx:04d}.png"),
                dpi=200,
            )
            plt.close(fig3a)

        if "3b" in probes:
            fig3b_samples = [int(i) for i in test_cfg.get("fig3b_samples", list(range(n_iters)))]
            available = [i for i in fig3b_samples if 0 <= i < preds.shape[0]]
            if not available:
                available = list(range(min(preds.shape[0], n_iters)))
            fig3b, ax3b = plt.subplots(1, len(available), figsize=(2 * len(available), 2))
            if len(available) == 1:
                ax3b = [ax3b]
            for idx, iter_idx in enumerate(available):
                _imshow_scaled(ax3b[idx], preds[iter_idx], contrast_low, contrast_high)
                ax3b[idx].set_title(rf"$\hat{{y}}_{{0,{iter_idx}}}$", fontsize=12)
                ax3b[idx].set_xticks([])
                ax3b[idx].set_yticks([])
            plt.tight_layout()
            fig3b.savefig(
                os.path.join(probe_dirs["3b"], f"probe_3b_frame-{frame_idx:04d}.png"),
                dpi=200,
            )
            plt.close(fig3b)

        if "3b_mark" in probes:
            fig3b_samples = [int(i) for i in test_cfg.get("fig3b_samples", list(range(n_iters)))]
            available = [i for i in fig3b_samples if 0 <= i < preds.shape[0]]
            if not available:
                available = list(range(min(preds.shape[0], n_iters)))
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
            fig3m.savefig(
                os.path.join(probe_dirs["3b_mark"], f"probe_3b_mark_frame-{frame_idx:04d}.png"),
                dpi=200,
            )
            plt.close(fig3m)

        if "3b_rand" in probes:
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
                lr_p1 = np.array([x1 / input_upsample, y1 / input_upsample])
                lr_p2 = np.array([x2 / input_upsample, y2 / input_upsample])
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
                inset_pair = (
                    (x1 - hr_x0, y1 - hr_y0),
                    (x2 - hr_x0, y2 - hr_y0),
                )
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
            fig_rand.savefig(
                os.path.join(probe_dirs["3b_rand"], f"probe_3b_rand_frame-{frame_idx:04d}.png"),
                dpi=200,
            )
            plt.close(fig_rand)

        if "3b_spots" in probes:
            if skip_inference:
                frame_hr = z_stack_loaded[frame_idx]
            else:
                frame_hr = preds
            if frame_hr.size and frame_idx == 0:
                imwrite(
                    os.path.join(output_path, "hr_stack_debug_frame0.tif"),
                    frame_hr.astype(np.float32),
                )
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
                    lr_x0 = lr_cx - lr_size / 2.0
                    lr_y0 = lr_cy - lr_size / 2.0
                    ax_lr.add_patch(
                        patches.Rectangle(
                            (lr_y0, lr_x0),
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

        if "3b_map" in probes:
            if skip_inference:
                frame_hr = z_stack_loaded[frame_idx]
            else:
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

            iter_colors = plt.cm.tab10(np.linspace(0, 1, n_iters_local))
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
                    ax_cluster.scatter(
                        pts[:, 1],
                        pts[:, 0],
                        s=8,
                        color=cluster_colors[c_idx],
                        alpha=0.5,
                        zorder=5,
                    )
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
                        count_hist = np.bincount(counts)
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
                        ax_gmm.scatter(
                            pts_local[:, 1],
                            pts_local[:, 0],
                            s=8,
                            color=cluster_colors[idx],
                            alpha=0.5,
                        )
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
                        if np.isfinite(avg_mean_dist):
                            ax_gmm.set_title(
                                f"N={n_mode}, LL={ll:.1f}, d={avg_mean_dist:.2f}",
                                fontsize=8,
                            )
                        else:
                            ax_gmm.set_title(f"N={n_mode}, LL={ll:.1f}", fontsize=8)

                    if cluster_centers:
                        rng = np.random.default_rng(test_cfg.get("spot_seed", 0) + frame_idx)
                        n_cols = 5
                        if n_iters_local >= n_cols:
                            iter_samples = rng.choice(n_iters_local, size=n_cols, replace=False)
                        else:
                            iter_samples = rng.choice(n_iters_local, size=n_cols, replace=True)
                        montage_fig = plt.figure(
                            figsize=(2 * n_cols, 2 * len(cluster_centers)),
                        )
                        montage_grid = montage_fig.add_gridspec(
                            len(cluster_centers),
                            n_cols,
                            wspace=0.0,
                            hspace=0.0,
                        )
                        for r_idx, center in enumerate(cluster_centers):
                            hr_center = center * lr_scale
                            for c_idx, iter_idx in enumerate(iter_samples):
                                ax = montage_fig.add_subplot(montage_grid[r_idx, c_idx])
                                patch_hr = _extract_patch(
                                    frame_hr[iter_idx],
                                    hr_center[0],
                                    hr_center[1],
                                    spot_patch_size,
                                )
                                _imshow_scaled(ax, patch_hr, contrast_low, contrast_high)
                                ax.set_xticks([])
                                ax.set_yticks([])
                                row_color = cluster_colors[r_idx]
                                for spine in ax.spines.values():
                                    spine.set_edgecolor(row_color)
                                    spine.set_linewidth(1.2)
                        montage_fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
                        montage_fig.savefig(
                            os.path.join(
                                probe_dirs["3b_map"],
                                f"probe_3b_map_montage_frame-{frame_idx:04d}.png",
                            ),
                            dpi=200,
                        )
                        plt.close(montage_fig)

            fig_map.tight_layout(pad=0.2)
            fig_map.savefig(
                os.path.join(probe_dirs["3b_map"], f"probe_3b_map_frame-{frame_idx:04d}.png"),
                dpi=200,
            )
            plt.close(fig_map)


    if not skip_inference and x_stack and y_stack and z_stack:
        x_stack_arr = np.stack(x_stack, axis=0)
        y_stack_arr = np.stack(y_stack, axis=0)
        z_stack_arr = np.stack(z_stack, axis=0)
        imwrite(os.path.join(output_path, "x_stack.tif"), x_stack_arr.astype(np.float32))
        imwrite(os.path.join(output_path, "y_stack.tif"), y_stack_arr.astype(np.float32))
        imwrite(os.path.join(output_path, "z_stack.tif"), z_stack_arr.astype(np.float32))
        if z_stack_arr.size:
            debug_path = os.path.join(output_path, "z_stack_debug_frame0_iter0.tif")
            imwrite(debug_path, z_stack_arr[0, 0].astype(np.float32))


if __name__ == "__main__":
    main()
