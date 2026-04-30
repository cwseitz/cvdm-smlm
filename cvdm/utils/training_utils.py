# Citation: Della Maggiora, Gabriel, Luis Alberto Croquevielle, Nikita Deshpande, Harry Horsley, Thomas Heinis, and Artur Yakimovich. "Conditional Variational Diffusion Models." ICLR 2023.

from typing import Any, Callable, Iterator, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

from cvdm.configs_pkg.data_config import DataConfig
from cvdm.data.localization_dataloader import SMLMDataLoader

def prepare_dataset(
    task: str, data_config: DataConfig, training: bool
) -> Tuple[tf.data.Dataset, tf.TensorShape, tf.TensorShape]:
    dataloader: Callable[
        [], Iterator[Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]]
    ]
    if task == "SMLM":
        dataloader = SMLMDataLoader(
            path=data_config.dataset_path,
            n_samples=data_config.n_samples,
            im_size=data_config.im_size,
        )
        x_channels = 1
        y_channels = x_channels
    else:
        raise NotImplementedError("Only SMLM task is supported by this dataloader setup.")

    x_shape = tf.TensorShape([data_config.im_size, data_config.im_size, x_channels])
    y_shape = tf.TensorShape([data_config.im_size, data_config.im_size, y_channels])
    dataset = tf.data.Dataset.from_generator(
        dataloader,
        output_types=(tf.float32, tf.float32),
        output_shapes=(
            x_shape,
            y_shape,
        ),
    )
    return dataset, x_shape, y_shape


def prepare_model_input(
    x: np.ndarray, y: np.ndarray, diff_inp: bool = False
) -> List[np.ndarray]:
    if diff_inp:
        dfy = y - x
    else:
        dfy = y
    s_tdx = np.random.uniform(0, 1, (x.shape[0], 1, 1, 1))
    ft = s_tdx * np.ones_like(dfy)
    return [dfy, x, ft]


def train_on_batch_cvdm(
    batch_x: np.ndarray, batch_y: np.ndarray, joint_model: Model, diff_inp: bool = False
) -> np.ndarray:
    model_input = prepare_model_input(batch_x, batch_y, diff_inp)
    loss = joint_model.train_on_batch(model_input, np.zeros_like(batch_y))
    return np.array(loss)


def taylor_expand_gamma(x, n=1):
    gamma_t_n = 0
    for i in range(n):
        gamma_t_n += tf.math.pow(x[0], i + 1)

    return gamma_t_n


def time_grad(x: List[np.ndarray]) -> float:
    gamma_t = x[0]
    timestep = x[1]

    d_gamma: float = tf.gradients(
        gamma_t, timestep, unconnected_gradients=tf.UnconnectedGradients.ZERO
    )[0]
    return d_gamma
