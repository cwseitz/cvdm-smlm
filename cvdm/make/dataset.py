import numpy as np
import matplotlib.pyplot as plt
from ..generators import *
from .kde import BasicKDE
from skimage.io import imsave
from skimage.exposure import rescale_intensity

class TrainDataset:
    """Training dataset object"""
    def __init__(self,ngenerate):
        self.ngenerate = ngenerate
        self.X_type = np.int16
        self.Z_type = np.float32
    def make_dataset(
        self,
        generator,
        upsample=8,
        sigma_kde=3.0,
        N_min=10,
        N_max=1000,
        show=False,
        generator_params=None,
    ):
        pad = upsample // 2
        if generator_params is None:
            generator_params = {}
        Xs = []; Zs = []; Ss = []; thetas = []
        for n in range(self.ngenerate):
            print(f'Generating sample {n}')
            nspots = np.random.randint(N_min,N_max)
            args = [nspots]
            G = generator.forward(*args, **generator_params)
            theta = G[2][:2,:].T
            S = G[1]; X = G[0]
            nx,ny = X.shape
            Z = BasicKDE(theta).forward(nx,upsample=upsample,sigma=sigma_kde)
            Z = rescale_intensity(Z,out_range=self.Z_type)
            Xs.append(X); Zs.append(Z); Ss.append(S)
            thetas.append(G[2])
            if show:
                fig,ax=plt.subplots(1,2)
                ax[0].imshow(X,cmap='gray')
                ax[1].imshow(Z,cmap='gray',vmin=0.0,vmax=1.0)
                plt.show()
        Ss = np.array(Ss,dtype=np.int16)
        return (np.array(Xs),np.array(Zs),Ss,thetas)
 

