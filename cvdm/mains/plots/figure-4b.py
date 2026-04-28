import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.restoration import rolling_ball
from scipy.ndimage import median_filter
from matplotlib.ticker import ScalarFormatter, MultipleLocator

path_hd = '/N/slate/cwseitz/cvdm/Tubes/Real/Real_High_Density/'
path_ls = '/N/slate/cwseitz/cvdm/Tubes/Real/Real_Long_Sequence/'

hd_idx = 0; ls_idx = 0; ls_sum_idx = 0
hd_1x = imread(path_hd + 'lr-1x-crop.tif')[hd_idx]
ls_1x = imread(path_ls + 'lr-1x.tif')[ls_idx]
ls_sum_1x = imread(path_ls + 'lr-1x-sum.tif')[ls_sum_idx]

hd_4x = imread(path_hd + f'eval/z-{hd_idx}-0.tif')
ls_4x = imread(path_ls + f'eval/z-{ls_idx}-0.tif')
ls_sum_4x = imread(path_ls + f'eval/z-{ls_sum_idx}-0.tif')

hd_4x[hd_4x < 0.0] = 0
ls_sum_4x[ls_sum_4x < 0.0] = 0
hd_4x -= rolling_ball(hd_4x,radius=5.0)
ls_sum_4x -= rolling_ball(ls_sum_4x,radius=5.0)
#hd_4x = median_filter(hd_4x,size=2)
#ls_sum_4x = median_filter(ls_sum_4x,size=2)
hd_4x[:5, :] = 0
ls_sum_4x[:5, :] = 0
hd_4x[:, :5] = 0
ls_sum_4x[:, :5] = 0

fig, ax = plt.subplots(2, 2, figsize=(5,5))

ax[0, 0].imshow(ls_sum_1x, cmap='gray',vmin=0.0)
ax[0, 1].imshow(hd_1x, cmap='gray',vmin=0.0)
ax[1, 0].imshow(ls_sum_4x, cmap='gray')
ax[1, 1].imshow(hd_4x, cmap='gray')

for axi in ax.ravel():
    axi.set_aspect(1.0)
    axi.set_xticks([])
    axi.set_yticks([])

ax[0, 0].set_ylabel('$x$', fontsize=14, labelpad=10)
ax[1, 0].set_ylabel('$\hat{y}_{0}$', fontsize=14, labelpad=10)

hr_inset_coords_list = [(40,12),(8,16)]
lr_inset_size = 15
hr_inset_size = 60
lr_inset_coords_list = [(10,3),(2,4)]

inset = ax[0,0].inset_axes([0.65, 0.65, 0.4, 0.4])
lr_inset_coords = lr_inset_coords_list[0]
inset.imshow(ls_sum_1x[lr_inset_coords[0]:lr_inset_coords[0]+lr_inset_size, lr_inset_coords[1]:lr_inset_coords[1]+lr_inset_size], cmap='gray', interpolation='nearest')
inset.set_xticks([])
inset.set_yticks([])
for spine in inset.spines.values():
    spine.set_color('red')
    spine.set_linewidth(1)

inset = ax[0,1].inset_axes([0.65, 0.65, 0.4, 0.4])
lr_inset_coords = lr_inset_coords_list[1]
inset.imshow(hd_1x[lr_inset_coords[0]:lr_inset_coords[0]+lr_inset_size, lr_inset_coords[1]:lr_inset_coords[1]+lr_inset_size], cmap='gray', interpolation='nearest')
inset.set_xticks([])
inset.set_yticks([])
for spine in inset.spines.values():
    spine.set_color('red')
    spine.set_linewidth(1)

inset = ax[1,0].inset_axes([0.65, 0.65, 0.4, 0.4])
hr_inset_coords = hr_inset_coords_list[0]
inset.imshow(ls_sum_4x[hr_inset_coords[0]:hr_inset_coords[0]+hr_inset_size, hr_inset_coords[1]:hr_inset_coords[1]+hr_inset_size], cmap='gray', interpolation='nearest')
inset.set_xticks([])
inset.set_yticks([])
for spine in inset.spines.values():
    spine.set_color('red')
    spine.set_linewidth(1)

inset = ax[1,1].inset_axes([0.65, 0.65, 0.4, 0.4])
hr_inset_coords = hr_inset_coords_list[1]
inset.imshow(hd_4x[hr_inset_coords[0]:hr_inset_coords[0]+hr_inset_size, hr_inset_coords[1]:hr_inset_coords[1]+hr_inset_size], cmap='gray', interpolation='nearest')
inset.set_xticks([])
inset.set_yticks([])
for spine in inset.spines.values():
    spine.set_color('red')
    spine.set_linewidth(1)

plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.1, hspace=0.1)
plt.savefig('/N/slate/cwseitz/cvdm/Tubes/Real/figure-10.png',dpi=200)

plt.show()
