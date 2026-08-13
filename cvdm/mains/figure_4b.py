import argparse
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import yaml
from skimage.io import imread
from skimage.restoration import rolling_ball


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 4b for real tube data.")
    parser.add_argument("--config", required=True, type=str, help="Path to figure4_tubes YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    paths_cfg = config["paths"]
    fig_cfg = config["figure_4b"]

    path_hd = paths_cfg["high_density_dir"]
    path_ls = paths_cfg["long_sequence_dir"]
    output_dir = paths_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    hd_idx = int(fig_cfg.get("hd_idx", 0))
    ls_idx = int(fig_cfg.get("ls_idx", 0))
    ls_sum_idx = int(fig_cfg.get("ls_sum_idx", 0))

    hd_1x = _read_frame(os.path.join(path_hd, "lr-1x-crop.tif"), hd_idx)
    ls_1x = _read_frame(os.path.join(path_ls, "lr-1x.tif"), ls_idx)
    ls_sum_1x = _read_frame(os.path.join(path_ls, "lr-1x-sum.tif"), ls_sum_idx)

    hd_4x = imread(os.path.join(path_hd, "eval", f"z-{hd_idx}-0.tif")).astype(np.float32)
    ls_4x = imread(os.path.join(path_ls, "eval", f"z-{ls_idx}-0.tif")).astype(np.float32)
    ls_sum_4x = imread(os.path.join(path_ls, "eval", f"z-{ls_sum_idx}-0.tif")).astype(np.float32)

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
    plt.show()


if __name__ == "__main__":
    main()
