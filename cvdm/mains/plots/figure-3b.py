import numpy as np
from skimage.io import imread
from glob import glob
import matplotlib.pyplot as plt

path = '/N/slate/cwseitz/cvdm/Sim/4x/N-100_N0-500-1000/eval_data/N100-1/'
hr100_files = glob(path + 'z-0-*.tif')
hr100 = np.array([imread(f) for f in hr100_files])

inset_coords = (50, 50)
hr_inset_size = 60

hr_inset_coords = (
    slice(inset_coords[0], inset_coords[0] + hr_inset_size),
    slice(inset_coords[1], inset_coords[1] + hr_inset_size),
)

fig, ax = plt.subplots(2, 2, figsize=(4,4))

samples = [0, 1, 4, 5]
for idx, i in enumerate(samples):
    row, col = divmod(idx, 2)
    ax[row, col].imshow(hr100[i][hr_inset_coords], cmap='gray')
    ax[row, col].set_xticks([])
    ax[row, col].set_yticks([])
    ax[row, col].set_title(rf'$\hat{{y}}_{{0,{i}}}$', fontsize=14)

plt.subplots_adjust(wspace=0.25, hspace=0.25)
exp = '/projects/data_solutions/raiders/pathai/seitzcx/exps/exp_CVDM/results'
plt.savefig(exp+'figure-3b.png',dpi=300)
plt.show()

