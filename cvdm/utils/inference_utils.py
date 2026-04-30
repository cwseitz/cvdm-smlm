# Citation: Della Maggiora, Gabriel, Luis Alberto Croquevielle, Nikita Deshpande, Harry Horsley, Thomas Heinis, and Artur Yakimovich. "Conditional Variational Diffusion Models." ICLR 2023.

import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from neptune import Run
from neptune.types import File
from skimage.util import montage
from tensorflow.keras.models import Model
from tqdm import tqdm

from cvdm.utils.metrics_utils import calculate_metrics


def ddpm_obtain_sr_img(
    x: np.ndarray,
    timesteps_test: int,
    noise_model: Model,
    schedule_model: Model,
    mu_model: Optional[Model],
    out_shape: Optional[Tuple[int, ...]] = None,
    store_schedule: bool = True,
    show_tqdm: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if out_shape == None:
        out_shape = x.shape
    assert out_shape is not None
    pred_sr = np.random.normal(0, 1, out_shape).astype(np.float32)
    if mu_model is not None:
        mu_pred = mu_model.predict_on_batch(x)[0]

    if store_schedule:
        alpha_vec = np.zeros(out_shape + (timesteps_test,), dtype=np.float32)
        t_iter = tqdm(range(timesteps_test), desc="Schedule", leave=False) if show_tqdm else range(timesteps_test)
        for t in t_iter:
            t_inp = np.clip(
                np.ones(out_shape, dtype=np.float32)
                * np.reshape(np.float32(t / timesteps_test), (1, 1, 1, 1)),
                0,
                0.99999,
            )
            sch_params_t = schedule_model.predict_on_batch([x, t_inp])
            alpha_t = np.clip(1 - sch_params_t[1] / timesteps_test, 1e-6, 0.99999).astype(
                np.float32
            )
            alpha_vec[..., t] = alpha_t
        gamma_vec = np.cumprod(alpha_vec, axis=-1)
        gamma_vec = np.clip(gamma_vec, 1e-10, 0.99999).astype(np.float32)
    else:
        # Memory-efficient path: compute gamma_T once, then step backward using
        # gamma_{t-1} = gamma_t / alpha_t. Avoids storing (H,W,C,T) arrays.
        alpha_vec = np.empty((0,), dtype=np.float32)
        gamma_vec = np.empty((0,), dtype=np.float32)
        gamma_t = np.ones(out_shape, dtype=np.float32)
        t_iter = tqdm(range(timesteps_test), desc="Gamma init", leave=False) if show_tqdm else range(timesteps_test)
        for t in t_iter:
            t_inp = np.clip(
                np.ones(out_shape, dtype=np.float32)
                * np.reshape(np.float32(t / timesteps_test), (1, 1, 1, 1)),
                0,
                0.99999,
            )
            sch_params_t = schedule_model.predict_on_batch([x, t_inp])
            alpha_t = np.clip(1 - sch_params_t[1] / timesteps_test, 1e-6, 0.99999).astype(
                np.float32
            )
            gamma_t *= alpha_t
        gamma_t = np.clip(gamma_t, 1e-10, 0.99999).astype(np.float32)
    count = 0
    pred_noise = 0
    t_iter = tqdm(range(timesteps_test, 1, -1), desc="Denoise", leave=False) if show_tqdm else range(timesteps_test, 1, -1)
    for t in t_iter:
        z: Union[float, np.ndarray] = np.random.normal(0, 1, out_shape).astype(np.float32)
        if t == 1:
            z = 0
        if store_schedule:
            alpha_t = alpha_vec[..., t - 1]
            beta_t = 1 - alpha_t
            gamma_t_step = gamma_vec[..., t - 1]
            gamma_tm1 = gamma_vec[..., t - 2]
        else:
            t_inp = np.clip(
                np.ones(out_shape, dtype=np.float32)
                * np.reshape(np.float32((t - 1) / timesteps_test), (1, 1, 1, 1)),
                0,
                0.99999,
            )
            sch_params_t = schedule_model.predict_on_batch([x, t_inp])
            alpha_t = np.clip(1 - sch_params_t[1] / timesteps_test, 1e-6, 0.99999).astype(
                np.float32
            )
            beta_t = 1 - alpha_t
            gamma_t_step = gamma_t
            gamma_tm1 = np.clip(gamma_t_step / alpha_t, 1e-10, 0.99999).astype(np.float32)
        beta_factor = (1 - gamma_tm1) * beta_t / (1 - gamma_t_step)
        if count > 0:
            pred_sr = (
                np.sqrt(gamma_t_step) * pred_sr
                + np.sqrt(1 - gamma_t_step - beta_factor) * pred_noise
                + np.sqrt(beta_factor) * z
            )
        if mu_model is not None:
            pred_noise = noise_model.predict_on_batch([pred_sr, x, mu_pred, gamma_t_step])
        else:
            pred_noise = noise_model.predict_on_batch([pred_sr, x, gamma_t_step])
        pred_sr = (pred_sr - np.sqrt(1 - gamma_t_step) * pred_noise) / np.sqrt(gamma_t_step)
        if not store_schedule:
            gamma_t = gamma_tm1
        count += 1
    if mu_model is not None:
        sigma = 0.5
        pred_diff = sigma * pred_sr + mu_pred
    else:
        pred_diff = pred_sr
    return pred_diff, gamma_vec, alpha_vec


def create_output_montage(
    pred_y: np.ndarray,
    gamma_vec: np.ndarray,
    y: np.ndarray,
    x: Optional[np.ndarray],
) -> np.ndarray:
    if pred_y.shape[3] > 1:
        channel_axis = 3
    else:
        channel_axis = None

    if x is not None:
        concatenated_images = np.concatenate(
            (pred_y, y, x, gamma_vec),
            axis=2,
        )
    else:
        concatenated_images = np.concatenate(
            (pred_y, y, gamma_vec),
            axis=2,
        )
    print(concatenated_images.shape)
    image: np.ndarray = montage(
        np.squeeze(concatenated_images),
        channel_axis=channel_axis,
    )
    return image


def log_loss(run: Optional[Run], avg_loss: np.ndarray, prefix: str) -> None:
    if run is not None:
        run[f"{prefix}_loss_sum"].log(avg_loss[0])
        run[f"{prefix}_loss_delta_noise"].log(avg_loss[1])
        run[f"{prefix}_loss_beta"].log(avg_loss[2])
        run[f"{prefix}_loss_KL"].log(avg_loss[3])
        run[f"{prefix}_loss_gamma"].log(avg_loss[4])
        if len(avg_loss) == 6:
            run[f"{prefix}_loss_mean"].log(avg_loss[5])
    else:
        loss_labels = [
            "Loss Sum",
            "Delta Noise Loss",
            "Beta Loss",
            "KL Loss",
            "Gamma Loss",
        ]
        formatted_losses = [
            f"{label}: {loss:.6f}" for label, loss in zip(loss_labels, avg_loss[:5])
        ]
        for loss in formatted_losses:
            print(loss)
        if len(avg_loss) == 6:
            print(f"Mean Loss: {avg_loss[5]:.6f}")


def log_metrics(
    run: Optional[Run], metrics_dict: Dict[str, float], prefix: str
) -> None:
    if run is not None:
        for metric_name, metric_value in metrics_dict.items():
            run[f"{prefix}_" + metric_name].log(metric_value)
    else:
        print(f"{prefix.capitalize()} Metrics:")
        for metric_name, metric_value in metrics_dict.items():
            print(f"{metric_name}: {metric_value:.6f}")


def save_weights(
    run: Optional[Run],
    model: Model,
    mu_model: Optional[Model],
    step: int,
    output_path: str,
    run_id: str,
) -> None:
    weights_dir = f"{output_path}/weights"
    os.makedirs(weights_dir, exist_ok=True)

    model_weights_path = f"{weights_dir}/model_{str(step)}_{run_id}.h5"
    model.save_weights(model_weights_path, True)

    if run is not None:
        run[f"model_weights/model_{str(step)}.h5"].upload(model_weights_path)

    if mu_model is not None:
        mu_model_weights_path = f"{weights_dir}/mu_model_{str(step)}_{run_id}.h5"
        mu_model.save_weights(mu_model_weights_path, True)

        if run is not None:
            run[f"mu_model_weights/mu_model_{str(step)}.h5"].upload(
                mu_model_weights_path
            )


def save_output_montage(
    run: Optional[Run],
    output_montage: np.ndarray,
    step: int,
    output_path: str,
    run_id: str,
    prefix: str,
    cmap: Optional[str] = None,
) -> None:
    output_dir = f"{output_path}/images"
    os.makedirs(output_dir, exist_ok=True)

    image_path = f"{output_dir}/{prefix}_output_{str(step)}_{run_id}.png"
    plt.imsave(image_path, output_montage, cmap=cmap)

    if run is not None:
        run[f"{prefix}_images"].append(
            File(image_path),
            description=f"Step {step}, {prefix}",
        )


def obtain_output_montage_and_metrics(
    batch_x: np.ndarray,
    batch_y: np.ndarray,
    noise_model: Model,
    schedule_model: Model,
    mu_model: Optional[Model],
    generation_timesteps: int,
    diff_inp: bool,
    task: str,
) -> Tuple[np.ndarray, Dict]:

    pred_diff, gamma_vec, _ = ddpm_obtain_sr_img(
        batch_x,
        generation_timesteps,
        noise_model,
        schedule_model,
        mu_model,
        batch_y.shape,
    )
    if diff_inp:
        pred_y = np.clip(pred_diff + batch_x, -1, 1)
    else:
        pred_y = np.clip(pred_diff, -1, 1)

    metrics = calculate_metrics(pred_y, batch_y)
    if task in ["biosr_sr", "imagenet_sr"]:
        gamma_vec = np.clip(gamma_vec[..., generation_timesteps // 2], -1, 1)
        montage_x = batch_x
    else:
        gamma_vec = np.clip(gamma_vec[..., 0:1, generation_timesteps // 2], -1, 1)
        montage_x = None
    output_montage = create_output_montage(
        pred_y,
        gamma_vec,
        batch_y,
        montage_x,
    )
    if task in ["biosr_sr", "imagenet_sr"]:
        output_montage = (output_montage * 127.5 + 127.5).astype(np.uint8)
    return output_montage, metrics
