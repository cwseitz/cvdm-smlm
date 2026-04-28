from typing import Iterator, Tuple
import numpy as np
from cvdm.utils.data_utils import center_crop
from skimage.io import imread, imsave


class SMLMDataLoader:
    def __init__(
        self,
        path: str,
        n_samples: int,
        im_size: int,
    ) -> None:
        self._x = imread(f"{path}/lr.tif")
        self._y = imread(f"{path}/hr.tif")
        if n_samples is not None and n_samples > 0:
            self._x = self._x[:n_samples]
            self._y = self._y[:n_samples]
        self._im_size = im_size
        self._n_samples: int = self._x.shape[0] if n_samples is None or n_samples <= 0 else min(n_samples, self._x.shape[0])

    def __len__(self) -> int:
        return self._n_samples

    def get_channels(self) -> Tuple[int, int]:
        return self._x.shape[-1], self._y.shape[-1]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        x, y = self._x[idx], self._y[idx]
        if len(x.shape) == 2:
            x = np.expand_dims(x, -1)
            y = np.expand_dims(y, -1)
        return x, y

    def __call__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        for i in range(self.__len__()):
            yield self.__getitem__(i)
