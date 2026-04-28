import numpy as np
from skimage.io import imread
from glob import glob
import matplotlib.pyplot as plt

path = '/N/slate/cwseitz/cvdm/Sim/4x/N-100_N0-500-1000/eval_data/N100-1/'
lr100 = imread(path + 'lr-1x.tif')[0]
hr100_true = imread(path + 'y-0-0.tif')
hr100_files = glob(path + 'z-0-*.tif')
hr100 = np.array([imread(f) for f in hr100_files])
hr100_avg = np.mean(hr100, axis=0)
hr100_std = np.std(hr100, axis=0)

inset_coords = (50, 50)
lr_inset_size = 15
hr_inset_size = 60

lr_inset_coords = (
    slice(inset_coords[0] // 4, inset_coords[0] // 4 + lr_inset_size),
    slice(inset_coords[1] // 4, inset_coords[1] // 4 + lr_inset_size),
)
hr_inset_coords = (
    slice(inset_coords[0], inset_coords[0] + hr_inset_size),
    slice(inset_coords[1], inset_coords[1] + hr_inset_size),
)

fig, ax = plt.subplots(2, 2, figsize=(5,5))

ax[0, 0].imshow(lr100, cmap='gray')
ax[0, 0].set_xticks([])
ax[0, 0].set_yticks([])
ax[0, 0].set_title(r'$x$', fontsize=14)
inset = ax[0, 0].inset_axes([0.65, 0.65, 0.4, 0.4])
inset.imshow(lr100[lr_inset_coords], cmap='gray')
inset.set_xticks([])
inset.set_yticks([])
for spine in inset.spines.values():
    spine.set_color('red')
    spine.set_linewidth(1)

ax[0, 1].imshow(hr100_true, cmap='gray')
ax[0, 1].set_xticks([])
ax[0, 1].set_yticks([])
ax[0, 1].set_title(r'$y_0$', fontsize=14)
inset = ax[0, 1].inset_axes([0.65, 0.65, 0.4, 0.4])
inset.imshow(hr100_true[hr_inset_coords], cmap='gray')
inset.set_xticks([])
inset.set_yticks([])
for spine in inset.spines.values():
    spine.set_color('red')
    spine.set_linewidth(1)

ax[1, 0].imshow(hr100_avg, cmap='gray')
ax[1, 0].set_xticks([])
ax[1, 0].set_yticks([])
ax[1, 0].set_title(r'$\langle \hat{y}_0 \rangle$', fontsize=14)
inset = ax[1, 0].inset_axes([0.65, 0.65, 0.4, 0.4])
inset.imshow(hr100_avg[hr_inset_coords], cmap='gray')
inset.set_xticks([])
inset.set_yticks([])
for spine in inset.spines.values():
    spine.set_color('red')
    spine.set_linewidth(1)

ax[1, 1].imshow(hr100_std, cmap='gray')
ax[1, 1].set_xticks([])
ax[1, 1].set_yticks([])
ax[1, 1].set_title(r'$\sqrt{\mathrm{Var}(\hat{y}_0)}$', fontsize=14)
inset = ax[1, 1].inset_axes([0.65, 0.65, 0.4, 0.4])
inset.imshow(hr100_std[hr_inset_coords], cmap='gray')
inset.set_xticks([])
inset.set_yticks([])
for spine in inset.spines.values():
    spine.set_color('red')
    spine.set_linewidth(1)

plt.subplots_adjust(wspace=0.2,hspace=0.3)
exp = '/projects/data_solutions/raiders/pathai/seitzcx/exps/exp_CVDM/results'
plt.savefig(exp+'figure-3a.png',dpi=300)
plt.show()
