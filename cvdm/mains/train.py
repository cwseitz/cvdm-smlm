import argparse
import logging
import os
import uuid
from typing import Callable, Iterator, Tuple

import numpy as np
import tensorflow as tf
import yaml
from skimage.exposure import rescale_intensity
from skimage.io import imsave
from skimage.transform import resize
import tifffile

from cvdm.configs_pkg.utils import (
    create_eval_config,
    create_model_config,
    create_neptune_config,
    create_training_config,
    load_config_from_yaml,
)
from cvdm.models.joint_model import instantiate_cvdm
from cvdm.make.kde import BasicKDE
from cvdm.psf.psf2d.psf2d import lamx, lamy
from cvdm.generators.generators import Nanoruler2D, Uniform2D
from cvdm.data.localization_dataloader import SMLMDataLoader
from cvdm.utils.zoom import custom_zoom
from cvdm.utils.inference_utils import (
    ddpm_obtain_sr_img,
    log_loss,
    log_metrics,
    obtain_output_montage_and_metrics,
    save_output_montage,
    save_weights,
)
from skimage.util import montage
from cvdm.utils.training_utils import prepare_model_input, train_on_batch_cvdm

try:
    import neptune as neptune
except Exception:  # pragma: no cover - optional
    neptune = None


tf.keras.utils.set_random_seed(42)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _resolve_generator(name: str):
    if name == "Nanoruler2D":
        return Nanoruler2D
    if name == "Uniform2D":
        return Uniform2D
    raise ValueError("sim.generator must be 'Nanoruler2D' or 'Uniform2D'")


def _transform_kde(
    kde: np.ndarray,
    sigma: float,
    scale_factor: float,
    enabled: bool,
) -> np.ndarray:
    if not enabled:
        return kde.astype(np.float32)
    if scale_factor != 1.0:
        kde = kde * scale_factor
    shift_value = _kde_half_max(sigma) * scale_factor
    return (kde - shift_value).astype(np.float32)


def _kde_half_max(sigma: float) -> float:
    coord = np.array([[0.0]], dtype=np.float64)
    peak = float(lamx(coord, 0.0, sigma) * lamy(coord, 0.0, sigma))
    return 0.5 * peak


def _make_label(
    label_type: str,
    theta: np.ndarray,
    size: int,
    upsample: int,
    sigma_kde: float,
    scale_factor: float,
    center_enabled: bool,
) -> np.ndarray:
    if label_type == "kde":
        theta_xy = theta[:2, :].T
        kde = BasicKDE(theta_xy).forward(size, upsample=upsample, sigma=sigma_kde)
        return _transform_kde(kde, sigma_kde, scale_factor, center_enabled)
    if label_type == "spikes":
        theta_xy = theta[:2, :]
        nspots = theta_xy.shape[1]
        spikes = np.zeros((size * upsample, size * upsample), dtype=np.float32)
        x_idx = np.clip((theta_xy[0] * upsample).astype(int), 0, size * upsample - 1)
        y_idx = np.clip((theta_xy[1] * upsample).astype(int), 0, size * upsample - 1)
        np.add.at(spikes, (x_idx, y_idx), 1)
        return spikes
    if label_type == "mu_s":
        return np.zeros((size, size), dtype=np.float32)
    raise ValueError("sim.label_type must be 'kde', 'spikes', or 'mu_s'")


def _upsample_image(image: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return image
    if factor & (factor - 1) == 0:
        out = image
        steps = int(np.log2(factor))
        for _ in range(steps):
            out = custom_zoom(out)
        return out
    new_shape = (image.shape[0] * factor, image.shape[1] * factor)
    return resize(image, new_shape, order=1, preserve_range=True, anti_aliasing=False).astype(image.dtype)


def _center_input(image: np.ndarray, mode: str, offset: float) -> np.ndarray:
    if mode == "none":
        return image
    if mode == "zscore":
        mean = float(np.mean(image))
        std = float(np.std(image))
        if std <= 0:
            return image - mean
        return (image - mean) / std
    raise ValueError("input_centering must be 'none' or 'zscore'")


def build_sim_dataset(
    sim_cfg: dict,
    default_size: int,
) -> Tuple[tf.data.Dataset, tf.TensorShape, tf.TensorShape]:
    size = int(sim_cfg.get("size") or default_size)
    generator_cls = _resolve_generator(sim_cfg.get("generator", "Nanoruler2D"))
    generator = generator_cls(size)

    label_type = sim_cfg.get("label_type", "kde")
    upsample = int(sim_cfg.get("label_upsample", 8))
    sigma_kde = float(sim_cfg.get("label_sigma", 3.0))
    label_scale = float(sim_cfg.get("label_scale", 1.0))
    label_centering = bool(sim_cfg.get("label_centering", True))
    input_upsample = int(sim_cfg.get("input_upsample", upsample if label_type != "mu_s" else 1))
    input_centering = sim_cfg.get("input_centering", "zscore")
    sim_offset = float(sim_cfg.get("offset", 0.0))
    if input_centering not in {"none", "zscore"}:
        raise ValueError("sim.input_centering must be 'none' or 'zscore'")
    nspots_min = int(sim_cfg.get("nspots_min", 10))
    nspots_max = int(sim_cfg.get("nspots_max", 50))

    def _iterator() -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        while True:
            nspots = np.random.randint(nspots_min, nspots_max + 1)
            spacing_px_val = sim_cfg.get("spacing_px", 4.0)
            if spacing_px_val is not None:
                spacing_px_val = float(spacing_px_val)

            b0_min = sim_cfg.get("B0_min", None)
            b0_max = sim_cfg.get("B0_max", None)
            if b0_min is not None and b0_max is not None:
                b0_val = float(np.random.uniform(b0_min, b0_max))
            else:
                b0_val = sim_cfg.get("B0", None)

            grf_sigma_min = sim_cfg.get("grf_sigma_min", None)
            grf_sigma_max = sim_cfg.get("grf_sigma_max", None)
            if grf_sigma_min is not None and grf_sigma_max is not None:
                grf_sigma_val = float(np.random.uniform(grf_sigma_min, grf_sigma_max))
            else:
                grf_sigma_val = float(sim_cfg.get("grf_sigma", 0.0))
            grf_seed_val = sim_cfg.get("grf_seed", None)

            adu, spikes, theta = generator.forward(
                nspots=nspots,
                sigma=float(sim_cfg.get("sigma", 0.92)),
                texp=float(sim_cfg.get("texp", 1.0)),
                N0_min=float(sim_cfg.get("N0_min", 500.0)),
                N0_max=float(sim_cfg.get("N0_max", 1000.0)),
                eta=float(sim_cfg.get("eta", 1.0)),
                gain=float(sim_cfg.get("gain", 1.0)),
                B0=b0_val,
                nframes=1,
                offset=float(sim_cfg.get("offset", 100.0)),
                var=float(sim_cfg.get("var", 5.0)),
                spacing_px=spacing_px_val,
                spacing_nm=sim_cfg.get("spacing_nm", None),
                pixel_size_nm=sim_cfg.get("pixel_size_nm", None),
                edgew=float(sim_cfg.get("edgew", 5.0)),
                position_sigma=float(sim_cfg.get("position_sigma", 0.0)),
                pattern=sim_cfg.get("pattern", "uniform"),
                parent_rate=sim_cfg.get("parent_rate", None),
                parent_count=sim_cfg.get("parent_count", None),
                children_sigma=float(sim_cfg.get("children_sigma", 1.0)),
                children_min=int(sim_cfg.get("children_min", 0)),
                children_pmf=sim_cfg.get("children_pmf", None),
                burst_prob=sim_cfg.get("burst_prob", None),
                halo_alpha=float(sim_cfg.get("halo_alpha", 0.0)),
                halo_sigma=float(sim_cfg.get("halo_sigma", 0.0)),
                grf_alpha=float(sim_cfg.get("grf_alpha", 0.0)),
                grf_sigma=grf_sigma_val,
                grf_seed=grf_seed_val,
            )
            if adu.ndim == 3:
                adu = adu[0]
                spikes = spikes[0]
            x = adu.astype(np.float32)
            if input_upsample > 1:
                x = _upsample_image(x, input_upsample)
            x = _center_input(x, input_centering, sim_offset)
            x = x[..., None]
            if label_type == "mu_s":
                texp = float(sim_cfg.get("texp", 1.0))
                eta = float(sim_cfg.get("eta", 1.0))
                mu_s = generator._mu_s(theta, texp=texp, eta=eta)
                y = mu_s.astype(np.float32)[..., None]
            else:
                y = _make_label(
                    label_type,
                    theta,
                    size,
                    upsample,
                    sigma_kde,
                    label_scale,
                    label_centering,
                )
                y = y.astype(np.float32)[..., None]
            yield x, y

    x_shape = tf.TensorShape([size * input_upsample, size * input_upsample, 1])
    if label_type == "mu_s":
        y_shape = tf.TensorShape([size, size, 1])
    else:
        y_shape = tf.TensorShape([size * upsample, size * upsample, 1])

    dataset = tf.data.Dataset.from_generator(
        _iterator,
        output_types=(tf.float32, tf.float32),
        output_shapes=(x_shape, y_shape),
    )
    return dataset, x_shape, y_shape


def build_val_dataset(
    val_cfg: dict,
    default_size: int,
) -> Tuple[tf.data.Dataset, tf.TensorShape, tf.TensorShape]:
    n_samples = val_cfg.get("n_samples", None)
    im_size = int(val_cfg.get("im_size") or default_size)
    input_upsample = int(val_cfg.get("input_upsample", 1))
    input_centering = val_cfg.get("input_centering", "zscore")
    if input_centering not in {"none", "zscore"}:
        raise ValueError("val.input_centering must be 'none' or 'zscore'")

    stack = tifffile.imread(val_cfg["path"]).astype(np.float32)
    if stack.ndim == 2:
        stack = stack[None, ...]
    if n_samples is not None and n_samples > 0:
        stack = stack[:n_samples]

    def _iterator() -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        for frame in stack:
            if frame.shape[0] != im_size or frame.shape[1] != im_size:
                frame = frame[:im_size, :im_size]
            if input_upsample > 1:
                frame = _upsample_image(frame, input_upsample)
            frame = _center_input(frame, input_centering, 0.0)
            if frame.ndim == 2:
                frame = np.expand_dims(frame, -1)
            yield frame, frame

    dataloader = _iterator
    x_shape = tf.TensorShape([im_size * input_upsample, im_size * input_upsample, 1])
    y_shape = tf.TensorShape([im_size * input_upsample, im_size * input_upsample, 1])

    dataset = tf.data.Dataset.from_generator(
        dataloader,
        output_types=(tf.float32, tf.float32),
        output_shapes=(x_shape, y_shape),
    )
    return dataset, x_shape, y_shape


def load_val_stack(val_cfg: dict, default_size: int) -> np.ndarray:
    n_samples = val_cfg.get("n_samples", None)
    im_size = int(val_cfg.get("im_size") or default_size)
    input_upsample = int(val_cfg.get("input_upsample", 1))
    input_centering = val_cfg.get("input_centering", "zscore")
    if input_centering not in {"none", "zscore"}:
        raise ValueError("val.input_centering must be 'none' or 'zscore'")

    stack = tifffile.imread(val_cfg["path"]).astype(np.float32)
    if stack.ndim == 2:
        stack = stack[None, ...]
    if n_samples is not None and n_samples > 0:
        stack = stack[:n_samples]

    processed = []
    for frame in stack:
        if frame.shape[0] != im_size or frame.shape[1] != im_size:
            frame = frame[:im_size, :im_size]
        if input_upsample > 1:
            frame = _upsample_image(frame, input_upsample)
        frame = _center_input(frame, input_centering, 0.0)
        if frame.ndim == 2:
            frame = np.expand_dims(frame, -1)
        processed.append(frame)
    return np.stack(processed, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True, help="Path to YAML config")
    parser.add_argument("--neptune-token", help="API token for Neptune")
    args = parser.parse_args()

    logger.info("Num CPUs Available: %s", len(tf.config.list_physical_devices("CPU")))
    logger.info("Num GPUs Available: %s", len(tf.config.list_physical_devices("GPU")))

    config = load_config_from_yaml(args.config_path)
    model_config = create_model_config(config)
    training_config = create_training_config(config)
    eval_config = create_eval_config(config)
    neptune_config = create_neptune_config(config)

    task = config.get("task", "SMLM")
    logger.info("Using task: %s", task)

    sim_cfg = config["sim"]
    val_cfg = config.get("val", {})
    val_enabled = bool(val_cfg.get("enabled", True))
    val_no_gt = bool(val_cfg.get("no_gt", True))

    logger.info("Building simulated training dataset...")
    data_cfg = config.get("data", {})
    batch_size = int(data_cfg.get("batch_size", 1))
    im_size = int(data_cfg.get("im_size", 64))

    dataset, x_shape, y_shape = build_sim_dataset(sim_cfg, im_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    if val_enabled:
        logger.info("Building validation dataset...")
        val_dataset, val_x_shape, val_y_shape = build_val_dataset(val_cfg, im_size)
        val_dataset = val_dataset.batch(batch_size, drop_remainder=True)
        val_stack = load_val_stack(val_cfg, im_size)

        if x_shape != val_x_shape or y_shape != val_y_shape:
            logger.warning(
                "Train/val shapes differ: train %s/%s vs val %s/%s",
                x_shape,
                y_shape,
                val_x_shape,
                val_y_shape,
            )
    else:
        logger.info("Validation disabled; skipping val dataset setup.")
        val_dataset = None
        val_stack = np.array([])

    logger.info("Creating model...")
    models = instantiate_cvdm(
        lr=training_config.lr,
        generation_timesteps=eval_config.generation_timesteps,
        cond_shape=x_shape,
        out_shape=y_shape,
        model_config=model_config,
    )
    noise_model, joint_model, schedule_model, mu_model = models

    if model_config.load_weights is not None:
        joint_model.load_weights(model_config.load_weights)
    if model_config.load_mu_weights is not None and mu_model is not None:
        mu_model.load_weights(model_config.load_mu_weights)

    run = None
    if args.neptune_token is not None and neptune_config is not None and neptune is not None:
        run = neptune.init_run(
            api_token=args.neptune_token,
            name=neptune_config.name,
            project=neptune_config.project,
        )
        run["config.yaml"].upload(args.config_path)

    output_root = eval_config.output_path
    run_id = str(uuid.uuid4())
    output_subdir = config.get("output_subdir", "sim_on_the_fly")
    output_path = os.path.join(output_root, output_subdir)
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, "config.yaml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    steps_per_epoch = int(config.get("steps_per_epoch", 100))
    val_steps = int(config.get("val_steps", 10))

    log_freq = eval_config.log_freq
    checkpoint_freq = eval_config.checkpoint_freq
    image_freq = eval_config.image_freq
    val_freq = eval_config.val_freq
    diff_inp = model_config.diff_inp
    sim_cfg = config.get("sim", {})
    sim_size = int(sim_cfg.get("size") or im_size)
    sim_generator_cls = _resolve_generator(sim_cfg.get("generator", "Nanoruler2D"))
    sim_generator = sim_generator_cls(sim_size)
    sim_label_upsample = int(sim_cfg.get("label_upsample", 4))
    sim_label_sigma = float(sim_cfg.get("label_sigma", 1.0))
    sim_input_upsample = int(sim_cfg.get("input_upsample", sim_label_upsample))
    sim_nspots_min = int(sim_cfg.get("nspots_min", 10))
    sim_nspots_max = int(sim_cfg.get("nspots_max", 50))
    sim_preview_count = int(config.get("sim_preview_count", 100))
    sim_preview_path = config.get("sim_preview_path", None)

    if sim_preview_count > 0:
        preview_path = sim_preview_path or os.path.join(output_path, "sim_preview.tif")
        preview_stack = []
        for _ in range(sim_preview_count):
            nspots = np.random.randint(sim_nspots_min, sim_nspots_max + 1)
            spacing_px_val = sim_cfg.get("spacing_px", 4.0)
            if spacing_px_val is not None:
                spacing_px_val = float(spacing_px_val)
            b0_min = sim_cfg.get("B0_min", None)
            b0_max = sim_cfg.get("B0_max", None)
            if b0_min is not None and b0_max is not None:
                b0_val = float(np.random.uniform(b0_min, b0_max))
            else:
                b0_val = sim_cfg.get("B0", None)
            grf_sigma_min = sim_cfg.get("grf_sigma_min", None)
            grf_sigma_max = sim_cfg.get("grf_sigma_max", None)
            if grf_sigma_min is not None and grf_sigma_max is not None:
                grf_sigma_val = float(np.random.uniform(grf_sigma_min, grf_sigma_max))
            else:
                grf_sigma_val = float(sim_cfg.get("grf_sigma", 0.0))
            grf_seed_val = sim_cfg.get("grf_seed", None)

            adu, _, _ = sim_generator.forward(
                nspots=nspots,
                sigma=float(sim_cfg.get("sigma", 0.92)),
                texp=float(sim_cfg.get("texp", 1.0)),
                N0_min=float(sim_cfg.get("N0_min", 500.0)),
                N0_max=float(sim_cfg.get("N0_max", 1000.0)),
                eta=float(sim_cfg.get("eta", 1.0)),
                gain=float(sim_cfg.get("gain", 1.0)),
                B0=b0_val,
                nframes=1,
                offset=float(sim_cfg.get("offset", 100.0)),
                var=float(sim_cfg.get("var", 5.0)),
                spacing_px=spacing_px_val,
                spacing_nm=sim_cfg.get("spacing_nm", None),
                pixel_size_nm=sim_cfg.get("pixel_size_nm", None),
                edgew=float(sim_cfg.get("edgew", 5.0)),
                position_sigma=float(sim_cfg.get("position_sigma", 0.0)),
                pattern=sim_cfg.get("pattern", "uniform"),
                parent_rate=sim_cfg.get("parent_rate", None),
                parent_count=sim_cfg.get("parent_count", None),
                children_sigma=float(sim_cfg.get("children_sigma", 1.0)),
                children_min=int(sim_cfg.get("children_min", 0)),
                children_pmf=sim_cfg.get("children_pmf", None),
                burst_prob=sim_cfg.get("burst_prob", None),
                halo_alpha=float(sim_cfg.get("halo_alpha", 0.0)),
                halo_sigma=float(sim_cfg.get("halo_sigma", 0.0)),
                grf_alpha=float(sim_cfg.get("grf_alpha", 0.0)),
                grf_sigma=grf_sigma_val,
                grf_seed=grf_seed_val,
            )
            if adu.ndim == 3:
                adu = adu[0]
            x = adu.astype(np.float32)
            if sim_input_upsample > 1:
                x = _upsample_image(x, sim_input_upsample)
            x = _center_input(x, sim_cfg.get("input_centering", "zscore"), 0.0)
            preview_stack.append(x.astype(np.float32))

        preview_stack = np.stack(preview_stack, axis=0)
        tifffile.imwrite(preview_path, preview_stack, dtype=np.float32)
        logger.info("Saved sim preview stack to %s", preview_path)

    logger.info("Starting training...")
    cumulative_loss = np.zeros(6 if model_config.zmd else 5)
    step = 0
    for ep in range(training_config.epochs):
        for batch in dataset.take(steps_per_epoch):
            batch_x, batch_y = batch
            logger.info("Epoch %s | Step %s", ep, step)
            cmap = "gray" if task in ["SMLM", "biosr_phase", "imagenet_phase"] else None
            cumulative_loss += train_on_batch_cvdm(
                batch_x, batch_y, joint_model, diff_inp=diff_inp
            )

            if step % log_freq == 0:
                avg_loss = cumulative_loss / (step + 1)
                log_loss(run=run, avg_loss=avg_loss, prefix="train")
                logger.info("Train loss @ step %s: %s", step, np.array2string(avg_loss, precision=6))

            if step % checkpoint_freq == 0:
                save_weights(
                    run=run,
                    model=joint_model,
                    mu_model=mu_model,
                    step=step,
                    output_path=output_path,
                    run_id=run_id,
                )

            if step % image_freq == 0:
                output_montage, metrics = obtain_output_montage_and_metrics(
                    batch_x,
                    batch_y.numpy(),
                    noise_model,
                    schedule_model,
                    mu_model,
                    eval_config.generation_timesteps,
                    diff_inp,
                    task,
                )
                log_metrics(run, metrics, prefix="train")
                save_output_montage(
                    run=run,
                    output_montage=output_montage,
                    step=step,
                    output_path=output_path,
                    run_id=run_id,
                    prefix="train",
                    cmap=cmap,
                )

            if val_enabled and step % val_freq == 0:
                val_loss = np.zeros(6 if model_config.zmd else 5)
                for val_batch in val_dataset.take(val_steps):
                    val_x, val_y = val_batch
                    model_input = prepare_model_input(val_x, val_y, diff_inp=diff_inp)
                    val_loss += joint_model.evaluate(
                        model_input, np.zeros_like(val_y), verbose=0
                    )

                log_loss(run=run, avg_loss=val_loss, prefix="val")
                logger.info("Val loss @ step %s: %s", step, np.array2string(val_loss, precision=6))
                random_batch = val_dataset.take(1)
                for val_x, val_y in random_batch:
                    if val_no_gt:
                        pred_diff, gamma_vec, _ = ddpm_obtain_sr_img(
                            val_x,
                            eval_config.generation_timesteps,
                            noise_model,
                            schedule_model,
                            mu_model,
                            val_y.shape,
                        )
                        if diff_inp:
                            pred_y = np.clip(pred_diff + val_x, -1, 1)
                        else:
                            pred_y = np.clip(pred_diff, -1, 1)
                        gamma_mid = np.clip(gamma_vec[..., eval_config.generation_timesteps // 2], -1, 1)
                        concat = np.concatenate((pred_y, val_x, gamma_mid), axis=2)
                        output_montage = montage(np.squeeze(concat), channel_axis=None)
                        save_output_montage(
                            run=run,
                            output_montage=output_montage,
                            step=step,
                            output_path=output_path,
                            run_id=run_id,
                            prefix="val",
                            cmap=cmap,
                        )
                        image_dir = os.path.join(output_path, "images")
                        os.makedirs(image_dir, exist_ok=True)
                        pred_path = os.path.join(image_dir, f"val_pred_{step}_{run_id}.png")
                        pred_img = np.squeeze(pred_y[0])
                        pred_img = rescale_intensity(pred_img, out_range=np.uint8).astype(np.uint8)
                        imsave(pred_path, pred_img)
                    else:
                        output_montage, metrics = obtain_output_montage_and_metrics(
                            val_x,
                            val_y.numpy(),
                            noise_model,
                            schedule_model,
                            mu_model,
                            eval_config.generation_timesteps,
                            diff_inp,
                            task,
                        )
                        log_metrics(run, metrics, prefix="val")
                        save_output_montage(
                            run=run,
                            output_montage=output_montage,
                            step=step,
                            output_path=output_path,
                            run_id=run_id,
                            prefix="val",
                            cmap=cmap,
                        )

                if val_stack.size:
                    preds = []
                    for start in range(0, val_stack.shape[0], batch_size):
                        val_x_full = val_stack[start : start + batch_size]
                        pred_diff, _, _ = ddpm_obtain_sr_img(
                            val_x_full,
                            eval_config.generation_timesteps,
                            noise_model,
                            schedule_model,
                            mu_model,
                            val_x_full.shape,
                        )
                        if diff_inp:
                            pred_y_full = np.clip(pred_diff + val_x_full, -1, 1)
                        else:
                            pred_y_full = np.clip(pred_diff, -1, 1)
                        preds.append(pred_y_full)
                    pred_stack = np.concatenate(preds, axis=0)
                    image_dir = os.path.join(output_path, "images")
                    os.makedirs(image_dir, exist_ok=True)
                    stack_path = os.path.join(image_dir, f"val_pred_stack_{step}_{run_id}.tif")
                    tifffile.imwrite(stack_path, np.squeeze(pred_stack).astype(np.float32))

                def _sample_sim_example() -> Tuple[np.ndarray, np.ndarray]:
                    nspots = np.random.randint(sim_nspots_min, sim_nspots_max + 1)
                    spacing_px_val = sim_cfg.get("spacing_px", 4.0)
                    if spacing_px_val is not None:
                        spacing_px_val = float(spacing_px_val)
                    b0_min = sim_cfg.get("B0_min", None)
                    b0_max = sim_cfg.get("B0_max", None)
                    if b0_min is not None and b0_max is not None:
                        b0_val = float(np.random.uniform(b0_min, b0_max))
                    else:
                        b0_val = sim_cfg.get("B0", None)

                    grf_sigma_min = sim_cfg.get("grf_sigma_min", None)
                    grf_sigma_max = sim_cfg.get("grf_sigma_max", None)
                    if grf_sigma_min is not None and grf_sigma_max is not None:
                        grf_sigma_val = float(np.random.uniform(grf_sigma_min, grf_sigma_max))
                    else:
                        grf_sigma_val = float(sim_cfg.get("grf_sigma", 0.0))
                    grf_seed_val = sim_cfg.get("grf_seed", None)

                    adu, spikes, theta = sim_generator.forward(
                        nspots=nspots,
                        sigma=float(sim_cfg.get("sigma", 0.92)),
                        texp=float(sim_cfg.get("texp", 1.0)),
                        N0_min=float(sim_cfg.get("N0_min", 500.0)),
                        N0_max=float(sim_cfg.get("N0_max", 1000.0)),
                        eta=float(sim_cfg.get("eta", 1.0)),
                        gain=float(sim_cfg.get("gain", 1.0)),
                        B0=b0_val,
                        nframes=1,
                        offset=float(sim_cfg.get("offset", 100.0)),
                        var=float(sim_cfg.get("var", 5.0)),
                        spacing_px=spacing_px_val,
                        spacing_nm=sim_cfg.get("spacing_nm", None),
                        pixel_size_nm=sim_cfg.get("pixel_size_nm", None),
                        edgew=float(sim_cfg.get("edgew", 5.0)),
                        position_sigma=float(sim_cfg.get("position_sigma", 0.0)),
                        pattern=sim_cfg.get("pattern", "uniform"),
                        parent_rate=sim_cfg.get("parent_rate", None),
                        parent_count=sim_cfg.get("parent_count", None),
                        children_sigma=float(sim_cfg.get("children_sigma", 1.0)),
                        children_min=int(sim_cfg.get("children_min", 0)),
                        children_pmf=sim_cfg.get("children_pmf", None),
                        burst_prob=sim_cfg.get("burst_prob", None),
                        halo_alpha=float(sim_cfg.get("halo_alpha", 0.0)),
                        halo_sigma=float(sim_cfg.get("halo_sigma", 0.0)),
                        grf_alpha=float(sim_cfg.get("grf_alpha", 0.0)),
                        grf_sigma=grf_sigma_val,
                        grf_seed=grf_seed_val,
                    )
                    if adu.ndim == 3:
                        adu = adu[0]
                    x = adu.astype(np.float32)
                    if sim_input_upsample > 1:
                        x = _upsample_image(x, sim_input_upsample)
                    x = _center_input(x, input_centering, sim_offset)
                    x = x[..., None]
                    y = _make_label(
                        "kde",
                        theta,
                        sim_size,
                        sim_label_upsample,
                        sim_label_sigma,
                        label_scale,
                        label_centering,
                    )
                    y = y.astype(np.float32)[..., None]
                    return x, y

                sim_rows = []
                for _ in range(3):
                    sim_x, sim_y = _sample_sim_example()
                    sim_x_batch = np.expand_dims(sim_x, axis=0)
                    sim_y_batch = np.expand_dims(sim_y, axis=0)
                    preds = []
                    for _ in range(10):
                        pred_diff, _, _ = ddpm_obtain_sr_img(
                            sim_x_batch,
                            eval_config.generation_timesteps,
                            noise_model,
                            schedule_model,
                            mu_model,
                            sim_y_batch.shape,
                        )
                        if diff_inp:
                            pred_y = np.clip(pred_diff + sim_x_batch, -1, 1)
                        else:
                            pred_y = np.clip(pred_diff, -1, 1)
                        preds.append(np.squeeze(pred_y[0]))
                    preds = np.stack(preds, axis=0)
                    sample_img = preds[0]
                    mean_img = preds.mean(axis=0)
                    var_img = preds.var(axis=0)
                    gt_img = np.squeeze(sim_y)
                    sim_rows.append([gt_img, sample_img, mean_img, var_img])

                if sim_rows:
                    h, w = sim_rows[0][0].shape
                    montage_canvas = np.zeros((3 * h, 4 * w), dtype=np.float32)
                    for i, row in enumerate(sim_rows):
                        for j, img in enumerate(row):
                            if img.shape != (h, w):
                                img = resize(img, (h, w), order=1, preserve_range=True, anti_aliasing=False)
                            montage_canvas[i * h : (i + 1) * h, j * w : (j + 1) * w] = img
                    save_output_montage(
                        run=run,
                        output_montage=montage_canvas,
                        step=step,
                        output_path=output_path,
                        run_id=run_id,
                        prefix="val_sim",
                        cmap=cmap,
                    )

            step += 1

    if run is not None:
        run.stop()


if __name__ == "__main__":
    main()
