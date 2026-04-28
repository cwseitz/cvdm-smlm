import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

path_hd = '/N/slate/cwseitz/cvdm/Tubes/Real/Real_High_Density/'
path_ls = '/N/slate/cwseitz/cvdm/Tubes/Real/Real_Long_Sequence/'

summed_hd = imread(path_hd + 'SUM_lr-1x.tif')
summed_ls = imread(path_ls + 'SUM_lr-1x.tif')

ls_cvdm = imread(path_ls + 'eval/render-cvdm.tif')
ls_thunder = imread(path_ls + 'thunderstorm/render.tif')
ls_thunder_multi = imread(path_ls + 'thunderstorm-multi/render.tif')

hd_cvdm = imread(path_hd + 'eval/render-cvdm.tif')
hd_thunder = imread(path_hd + 'thunderstorm/render-crop.tif') #multi

ls_cvdm = np.roll(ls_cvdm,1,axis=1)
hd_cvdm = np.roll(hd_cvdm,5,axis=0)
hd_cvdm = np.roll(hd_cvdm,4,axis=1)

fig, ax = plt.subplots(1,4,figsize=(10,4))
ax[0].imshow(summed_ls, cmap='gray', vmin=0.0)
ax[1].imshow(ls_thunder, cmap='gray', vmin=0.0,vmax=40.0)
ax[2].imshow(ls_thunder_multi,cmap='gray',vmin=0.0,vmax=30.0)
ax[3].imshow(ls_cvdm, cmap='gray', vmin=0.0)
for axi in ax.ravel():
    axi.set_xticks([])
    axi.set_yticks([])

plt.savefig('/N/slate/cwseitz/cvdm/Tubes/Real/figure-11-1-1.png',dpi=200)

fig, ax = plt.subplots(1,3,figsize=(8,4))
ax[0].imshow(summed_hd, cmap='gray', vmin=0.0)
ax[1].imshow(hd_thunder, cmap='gray', vmin=0.0)
ax[2].imshow(hd_cvdm,cmap='gray')
for axi in ax.ravel():
    axi.set_xticks([])
    axi.set_yticks([])

plt.savefig('/N/slate/cwseitz/cvdm/Tubes/Real/figure-11-1-2.png',dpi=200)
plt.show()

fig_line, ax_line = plt.subplots(2,1,figsize=(10,5))

x_hr = np.arange(50,80)
y_combined_r = ls_cvdm[120, x_hr]
y_combined_r /= y_combined_r.max()
y_combined_b = ls_thunder[120, x_hr]
y_combined_b /= y_combined_b.max()
y_combined_g = ls_thunder_multi[120, x_hr]
y_combined_g /= y_combined_g.max()
pixel_size = 25.0  # nm

x_hr -= 40
ax_line[0].plot((x_hr-x_hr[0]) * pixel_size, y_combined_r, 'r-', marker='o', label='CVDM (LS-SUM)')
ax_line[0].plot((x_hr-x_hr[0]) * pixel_size, y_combined_b, 'b-', marker='o', label='ThunderSTORM (LS)')
ax_line[0].plot((x_hr-x_hr[0]) * pixel_size, y_combined_g, color='cyan', marker='o', label='ThunderSTORM (LS-SUM)')
ax_line[0].legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.6, 1.5), ncol=3, frameon=False)

ax_line[0].set_xlabel('Distance (nm)',fontsize=12)
ax_line[0].set_ylabel('Intensity (a.u.)',fontsize=12)
ax_line[0].spines['top'].set_visible(False)
ax_line[0].spines['right'].set_visible(False)

x_hr = np.arange(83,113)
y_combined_r = hd_cvdm[182, x_hr]
y_combined_r /= y_combined_r.max()
y_combined_b = hd_thunder[182, x_hr]
y_combined_b /= y_combined_b.max()
pixel_size = 25.0  # nm

x_hr -= 40
ax_line[1].plot((x_hr-x_hr[0]) * pixel_size, y_combined_r, 'r-', marker='o', label='CVDM (HD)')
ax_line[1].plot((x_hr-x_hr[0]) * pixel_size, y_combined_b, 'b-', marker='o', label='ThunderSTORM (HD)')
ax_line[1].legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, 1.5), ncol=3, frameon=False)

ax_line[1].set_xlabel('Distance (nm)',fontsize=12)
ax_line[1].set_ylabel('Intensity (a.u.)',fontsize=12)
ax_line[1].spines['top'].set_visible(False)
ax_line[1].spines['right'].set_visible(False)

plt.tight_layout()
exp = '/projects/data_solutions/raiders/pathai/seitzcx/exps/exp_CVDM/results'
plt.savefig(exp+'figure-4c.png',dpi=300)

fig,ax=plt.subplots(1,2,figsize=(6,3))
ax[0].imshow(hd_thunder[162:212,73:123],cmap='gray')
ax[1].imshow(hd_cvdm[162:212,73:123],cmap='gray')
for axi in ax.ravel():
    axi.set_xticks([])
    axi.set_yticks([])
plt.savefig(exp+'figure-4d.png',dpi=300)


