import numpy as np
import matplotlib.pyplot as plt
import cv2
from ..psf.psf2d.psf2d import *
from scipy.stats import poisson

class Density:
    def __init__(self):
        pass
        
class Disc(Density):
    """Uniform distribution on a disc"""
    def __init__(self,radius):
        super().__init__()
        self.radius=radius
    def sample(self,n):
        theta = np.random.uniform(0,2*np.pi,n)
        radius = self.radius*np.sqrt(np.random.uniform(0, 1, n))
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        return x, y

class Uniform(Density):
    """Uniform distribution"""
    def __init__(self,size,edgew=5.0):
        super().__init__()
        self.size=size
        self.edgew = edgew
    def sample(self,n):
        x = np.random.uniform(self.edgew,self.size-self.edgew,n)
        y = np.random.uniform(self.edgew,self.size-self.edgew,n)
        return x, y
       
class Generator:
    def __init__(self,nx,ny):
        self.nx = nx
        self.ny = ny
    def show(self,theta,adu,noise,muS,S):
        fig,ax=plt.subplots(1,3,figsize=(10,4))
        ax[0].scatter(theta[1,:],theta[0,:],color='black',s=5,marker='o')
        ax[1].imshow(muS,cmap='gray')
        ax[2].imshow(adu,cmap='gray')
        ax[0].set_xticks([]); ax[0].set_yticks([])
        ax[1].set_xticks([]); ax[1].set_yticks([])
        ax[2].set_xticks([]); ax[2].set_yticks([])
        ax[0].set_xlim([0,adu.shape[0]])
        ax[0].set_ylim([0,adu.shape[1]])
        ax[0].set_aspect(1.0)
        ax[0].invert_yaxis()
        plt.show()

        
    def _gaussian_blur(self, image: np.ndarray, sigma: float) -> np.ndarray:
        if sigma <= 0:
            return image
        ksize = int(np.ceil(sigma * 6))
        ksize = ksize + 1 if ksize % 2 == 0 else ksize
        return cv2.GaussianBlur(image.astype(np.float32), (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

    def _sample_grf(self, sigma: float, rng: np.random.Generator) -> np.ndarray:
        if sigma <= 0:
            return np.zeros((self.nx, self.ny), dtype=np.float32)
        white = rng.normal(0.0, 1.0, size=(self.nx, self.ny)).astype(np.float32)
        field = self._gaussian_blur(white, sigma)
        field = field - float(field.mean())
        field_std = float(field.std())
        if field_std > 0:
            field = field / field_std
        return field

    def sample_frames(
        self,
        theta,
        nframes,
        texp,
        eta,
        B0,
        gain,
        offset,
        var,
        show=False,
        halo_alpha: float = 0.0,
        halo_sigma: float = 0.0,
        grf_alpha: float = 0.0,
        grf_sigma: float = 0.0,
        grf_seed: int | None = None,
    ):
        _adu = []; _spikes = []
        rng = np.random.default_rng(grf_seed)
        for n in range(nframes):
            muS = self._mu_s(theta,texp=texp,eta=eta)
            S = self.shot_noise(muS)
            muB = self._mu_b(B0) if B0 is not None else 0
            if halo_alpha != 0.0 and halo_sigma > 0:
                halo = halo_alpha * self._gaussian_blur(muS, halo_sigma)
                muB = muB + halo
            if grf_alpha != 0.0 and grf_sigma > 0:
                grf = self._sample_grf(grf_sigma, rng)
                muB = muB + grf_alpha * grf
            if isinstance(muB, np.ndarray):
                muB = np.maximum(muB, 0.0)
            elif muB is None:
                muB = 0
            B = self.shot_noise(muB) if isinstance(muB, np.ndarray) or muB != 0 else 0
            read_noise = self.read_noise(offset=offset,var=var)
            adu = gain*(S+B) + read_noise
            adu = np.clip(adu,0,None)
            adu = np.squeeze(adu)
            spikes = self.spikes(theta)
            _adu.append(adu); _spikes.append(spikes)
            if show:
                self.show(theta,adu,read_noise,muS,S)
            
        adu = np.squeeze(np.array(_adu))
        spikes = np.squeeze(np.array(_spikes))
        return adu,spikes
    def _mu_s(self,theta,texp=1.0,eta=1.0,patch_hw=3):
        x = np.arange(0,2*patch_hw); y = np.arange(0,2*patch_hw)
        X,Y = np.meshgrid(x,y,indexing='ij')
        mu = np.zeros((self.nx,self.ny),dtype=np.float32)
        ntheta,nspots = theta.shape
        for n in range(nspots):
            x0,y0,sigma,N0 = theta[:,n]
            i0 = eta*N0*texp
            patchx, patchy = int(round(x0))-patch_hw, int(round(y0))-patch_hw
            x0p = x0-patchx; y0p = y0-patchy
            this_mu = i0*lamx(X,x0p,sigma)*lamy(Y,y0p,sigma)
            x_start = max(patchx, 0)
            y_start = max(patchy, 0)
            x_end = min(patchx + 2 * patch_hw, self.nx)
            y_end = min(patchy + 2 * patch_hw, self.ny)
            if x_start >= x_end or y_start >= y_end:
                continue
            local_x_start = x_start - patchx
            local_y_start = y_start - patchy
            local_x_end = local_x_start + (x_end - x_start)
            local_y_end = local_y_start + (y_end - y_start)
            mu[x_start:x_end, y_start:y_end] += this_mu[local_x_start:local_x_end, local_y_start:local_y_end]
        return mu

    def _mu_b(self,B0):
        rate = B0*np.ones((self.nx,self.ny))
        return rate
       
    def shot_noise(self,rate):
        """Universal for all types of detectors"""
        electrons = np.random.poisson(lam=rate)
        return electrons
                
    def read_noise(self,offset=100.0,var=5.0):
        """Gaussian readout noise"""
        noise = np.random.normal(offset,np.sqrt(var),size=(self.nx,self.ny))
        return noise
        
    def spikes(self,theta,upsample=4):
        new_nx = self.nx * upsample
        new_ny = self.ny * upsample
        theta = theta[:2, :, np.newaxis, np.newaxis]
        x_vals = np.linspace(0, new_nx, new_nx, endpoint=False)
        y_vals = np.linspace(0, new_ny, new_ny, endpoint=False)

        x_indices = np.floor(theta[0] * upsample).astype(int)
        y_indices = np.floor(theta[1] * upsample).astype(int)

        x_indices = np.clip(x_indices, 0, new_nx - 1)
        y_indices = np.clip(y_indices, 0, new_ny - 1)

        spikes = np.zeros((new_nx, new_ny), dtype=int)
        np.add.at(spikes, (x_indices, y_indices), 1)

        return spikes


        

    
        

