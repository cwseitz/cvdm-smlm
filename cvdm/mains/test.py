import argparse
import os
from typing import Dict, List, Tuple
from skimage.io import imread, imsave
from tifffile import imwrite
from skimage.transform import resize
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
    skip_inference = bool(test_cfg.get("skip_inference", False))
    if skip_inference:
        return

    lr_stack = imread(stack_path)
    if lr_stack.ndim == 2:
        lr_stack = lr_stack[None, ...]
    if n_frames:
        lr_stack = lr_stack[: int(n_frames)]

    input_upsample = int(test_cfg.get("input_upsample", 4))
    subtract_offset = bool(test_cfg.get("subtract_offset", False))
    offset = float(test_cfg.get("offset", 0.0))
    input_centering = test_cfg.get("input_centering", "none")

    noise_model = schedule_model = mu_model = None
    joint_model = None
    model_ready = False
    x_stack = []
    y_stack = []
    z_stack = []
    for lr_raw in tqdm(lr_stack, desc="Frames"):
        lr_raw = lr_raw.astype(np.float32)
        x = lr_raw
        if subtract_offset:
            x = x - offset
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
        if input_centering == "zscore":
            mean = float(np.mean(x_up))
            std = float(np.std(x_up))
            if std > 0:
                x_up = (x_up - mean) / std

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

        x_stack.append(lr_raw)
        y_stack.append(x_up)
        z_stack.append(preds)

    if x_stack and y_stack and z_stack:
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
