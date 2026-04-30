import numpy as np
import matplotlib.pyplot as plt
from miniSMLM import generators, localize, psf
from cvdm.make import BasicKDE

config = {
    "sigma": 0.92,
    "N0": 200,
    "B0": 0,
    "eta": 1.0,
    "texp": 1.0,
    "gain": 1.0,
    "offset": 100.0,
    "var": 5.0
}

nparticles=30;radius=20.0;npixels=50
disc2d = generators.Disc2D(npixels,npixels)
adu,_,theta = disc2d.forward(radius,nparticles,**config,show=True)
nx,ny = adu.shape

kde = BasicKDE(theta[:2,:].T)
render0 = kde.forward(nx,upsample=4)
render0 = render0/render0.max()
render1 = render0 + np.random.normal(0,scale=0.1,size=render0.shape)
render2 = render1 + np.random.normal(0,scale=0.2,size=render0.shape)
render3 = render2 + np.random.normal(0,scale=0.3,size=render0.shape)
render4 = np.random.normal(0,scale=0.5,size=render0.shape)
fig,ax=plt.subplots(1,5,figsize=(10,2))
ax[0].imshow(render0,cmap='gray')
ax[1].imshow(render1,cmap='gray')
ax[2].imshow(render2,cmap='gray')
ax[3].imshow(render3,cmap='gray')
ax[4].imshow(render4,cmap='gray')
for axi in ax.ravel():
    axi.set_xticks([])
    axi.set_yticks([])
    axi.spines[['left','right','top','bottom']].set_visible(False)
plt.tight_layout()
plt.subplots_adjust(hspace=0.0)
exp = '/projects/data_solutions/raiders/pathai/seitzcx/exps/exp_CVDM/results'
plt.savefig(exp+'figure-1b.png',dpi=300)

