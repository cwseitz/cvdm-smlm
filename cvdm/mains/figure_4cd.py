import argparse
import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import yaml
from skimage.io import imread


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _profile_norm(row: np.ndarray, start: int, stop: int) -> np.ndarray:
    vals = row[start:stop].astype(np.float32)
    denom = float(np.max(vals)) if vals.size else 1.0
    if denom <= 0:
        denom = 1.0
    return vals / denom


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 4c/4d panels for real tube data.")
    parser.add_argument("--config", required=True, type=str, help="Path to figure4_tubes YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    paths_cfg = config["paths"]
    fig_cfg = config["figure_4cd"]

    path_hd = paths_cfg["high_density_dir"]
    path_ls = paths_cfg["long_sequence_dir"]
    output_dir = paths_cfg["output_dir"]
    summary_output_dir = paths_cfg.get("summary_output_dir", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(summary_output_dir, exist_ok=True)

    summed_hd = imread(os.path.join(path_hd, "SUM_lr-1x.tif"))
    summed_ls = imread(os.path.join(path_ls, "SUM_lr-1x.tif"))

    ls_cvdm = imread(os.path.join(path_ls, "eval", "render-cvdm.tif")).astype(np.float32)
    ls_thunder = imread(os.path.join(path_ls, "thunderstorm", "render.tif")).astype(np.float32)
    ls_thunder_multi = imread(os.path.join(path_ls, "thunderstorm-multi", "render.tif")).astype(np.float32)

    hd_cvdm = imread(os.path.join(path_hd, "eval", "render-cvdm.tif")).astype(np.float32)
    hd_thunder = imread(os.path.join(path_hd, "thunderstorm", "render-crop.tif")).astype(np.float32)

    ls_roll = fig_cfg.get("ls_cvdm_roll", [0, 1])
    ls_cvdm = np.roll(ls_cvdm, int(ls_roll[1]), axis=int(ls_roll[0]))

    hd_roll_axis0 = int(fig_cfg.get("hd_cvdm_roll_axis0", 5))
    hd_roll_axis1 = int(fig_cfg.get("hd_cvdm_roll_axis1", 4))
    hd_cvdm = np.roll(hd_cvdm, hd_roll_axis0, axis=0)
    hd_cvdm = np.roll(hd_cvdm, hd_roll_axis1, axis=1)

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

    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    ax[0].imshow(summed_hd, cmap="gray", vmin=0.0)
    ax[1].imshow(hd_thunder, cmap="gray", vmin=0.0)
    ax[2].imshow(hd_cvdm, cmap="gray")
    for axi in ax.ravel():
        axi.set_xticks([])
        axi.set_yticks([])

    panel_hd_name = fig_cfg.get("output_panel_hd", "figure-11-1-2.png")
    plt.savefig(os.path.join(output_dir, panel_hd_name), dpi=200)
    plt.show()

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
    plt.savefig(os.path.join(summary_output_dir, out_line), dpi=300)

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
    plt.savefig(os.path.join(summary_output_dir, out_crop), dpi=300)


if __name__ == "__main__":
    main()
