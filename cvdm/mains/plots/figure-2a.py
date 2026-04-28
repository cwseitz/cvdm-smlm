import os
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BASE_DIR = "/projects/data_solutions/raiders/pathai/seitzcx/exps/exp_CVDM"
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

DATASET_NAMES = [
    "cvdm_sz64_n1000_N100-101_08a3fce4",
    "cvdm_sz64_n1000_N200-201_f828753b",
    "cvdm_sz64_n1000_N500-501_8b705769",
]


def _lr_path(dataset_name):
    return os.path.join(DATA_DIR, dataset_name, "lr-1x.tif")


def _result_path(dataset_name, filename):
    return os.path.join(RESULTS_DIR, dataset_name, filename)


lr100 = imread(_lr_path(DATASET_NAMES[0]))[0]
hr100_true = imread(_result_path(DATASET_NAMES[0], "y-0-0.tif"))
hr100 = imread(_result_path(DATASET_NAMES[0], "z-0-0.tif"))

lr200 = imread(_lr_path(DATASET_NAMES[1]))[0]
hr200_true = imread(_result_path(DATASET_NAMES[1], "y-0-0.tif"))
hr200 = imread(_result_path(DATASET_NAMES[1], "z-0-0.tif"))

lr500 = imread(_lr_path(DATASET_NAMES[2]))[0]
hr500_true = imread(_result_path(DATASET_NAMES[2], "y-0-0.tif"))
hr500 = imread(_result_path(DATASET_NAMES[2], "z-0-0.tif"))

hr100[hr100 < 0] = 0
hr200[hr200 < 0] = 0
hr500[hr500 < 0] = 0

densities = [100, 200, 500]
fig, axes = plt.subplots(3, 3, figsize=(9, 9))
plt.subplots_adjust(wspace=0.02, hspace=0.1)

hr_inset_coords_list = [(15, 40), (50, 50), (100, 100)]

lr_inset_size = 15
hr_inset_size = 60

for row, (lr, hr_true, hr, hr_inset_coords) in enumerate(zip(
        [lr100, lr200, lr500], 
        [hr100_true, hr200_true, hr500_true], 
        [hr100, hr200, hr500], 
        hr_inset_coords_list)):

    lr_inset_coords = (hr_inset_coords[0] // 4, hr_inset_coords[1] // 4)

    axes[row, 0].imshow(lr, cmap='gray')
    axes[row, 0].set_ylabel(f'$\\rho$={densities[row]}', fontsize=16)
    axes[row, 0].set_xticks([])
    axes[row, 0].set_yticks([])


    inset = axes[row, 0].inset_axes([0.65, 0.65, 0.4, 0.4])
    inset.imshow(lr[lr_inset_coords[0]:lr_inset_coords[0]+lr_inset_size, lr_inset_coords[1]:lr_inset_coords[1]+lr_inset_size], cmap='gray', interpolation='nearest')
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color('red')
        spine.set_linewidth(1)

    axes[row, 1].imshow(hr_true, cmap='gray')
    axes[row, 1].set_xticks([])
    axes[row, 1].set_yticks([])

    inset = axes[row, 1].inset_axes([0.65, 0.65, 0.4, 0.4])
    inset.imshow(hr_true[hr_inset_coords[0]:hr_inset_coords[0]+hr_inset_size, hr_inset_coords[1]:hr_inset_coords[1]+hr_inset_size], cmap='gray', interpolation='nearest')
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color('red')
        spine.set_linewidth(1)

    axes[row, 2].imshow(hr, cmap='gray')
    axes[row, 2].set_xticks([])
    axes[row, 2].set_yticks([])

    inset = axes[row, 2].inset_axes([0.65, 0.65, 0.4, 0.4])
    inset.imshow(hr[hr_inset_coords[0]:hr_inset_coords[0]+hr_inset_size, hr_inset_coords[1]:hr_inset_coords[1]+hr_inset_size], cmap='gray', interpolation='nearest')
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color('red')
        spine.set_linewidth(1)

axes[0, 0].set_title(r'$x$', fontsize=16)
axes[0, 1].set_title(r'$y_0$', fontsize=16)
axes[0, 2].set_title(r'$\hat{y}_0$', fontsize=16)

plt.savefig(os.path.join(RESULTS_DIR, "figure-2a.png"), dpi=300)
plt.show()

