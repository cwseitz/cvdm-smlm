import numpy as np
import matplotlib.pyplot as plt
from ..psf.psf2d.psf2d import *
from scipy.stats import multivariate_normal

class BasicKDE:
    def __init__(self,theta):
        self.theta = theta
    def forward(self,npixels,upsample=10,sigma=1.0):
        patchw = int(round(3*sigma))
        theta = upsample*(self.theta)
        kde = np.zeros((upsample*npixels,upsample*npixels),dtype=np.float32)
        ns,nd = theta.shape
        x = np.arange(0,2*patchw); y = np.arange(0,2*patchw)
        X,Y = np.meshgrid(x,y,indexing='ij')
        for n in range(ns):
            x0,y0 = theta[n,:]
            patchx, patchy = int(round(x0))-patchw, int(round(y0))-patchw
            x0p = x0-patchx; y0p = y0-patchy
            lam = lamx(X,x0p,sigma)*lamy(Y,y0p,sigma)
            kde_xmin = patchx; kde_xmax = patchx+2*patchw
            kde_ymin = patchy; kde_ymax = patchy+2*patchw
            x_min = max(kde_xmin, 0)
            y_min = max(kde_ymin, 0)
            x_max = min(kde_xmax, kde.shape[0])
            y_max = min(kde_ymax, kde.shape[1])
            if x_min < x_max and y_min < y_max:
                lam_xmin = x_min - kde_xmin
                lam_ymin = y_min - kde_ymin
                lam_xmax = lam_xmin + (x_max - x_min)
                lam_ymax = lam_ymin + (y_max - y_min)
                kde[x_min:x_max, y_min:y_max] += lam[lam_xmin:lam_xmax, lam_ymin:lam_ymax]
        return kde
        
class DistKDE:
    def __init__(self,theta):
        self.theta = theta
    def forward(self,npixels,upsample=10,sigma=1.0,xyvar=1.0,xylim=4.0,nsamples=1000):
        patchw = int(round(5*sigma))
        theta = upsample*(self.theta)
        kde = np.zeros((upsample*npixels,upsample*npixels),dtype=np.float32)
        ns,nd = theta.shape
        x = np.arange(0,2*patchw); y = np.arange(0,2*patchw)
        X,Y = np.meshgrid(x,y,indexing='ij')
        mu = [patchw, patchw]; cov = [[xyvar,0],[0,xyvar]]
        for n in range(ns):
            x0,y0 = theta[n,:]
            patchx, patchy = int(round(x0))-patchw, int(round(y0))-patchw
            x0p = x0-patchx; y0p = y0-patchy
            lams = []
            norm = multivariate_normal(mu,cov)
            for m in range(nsamples):
                this_x0p,this_y0p = norm.rvs()
                lam = lamx(X,this_x0p,sigma)*lamy(Y,this_y0p,sigma)
                lams.append(lam)
            lams = np.array(lams);
            avg = np.average(lams,axis=0)
            var = np.var(lams,axis=0)
            kde_xmin = patchx; kde_xmax = patchx+2*patchw
            kde_ymin = patchy; kde_ymax = patchy+2*patchw
            kde[kde_xmin:kde_xmax,kde_ymin:kde_ymax] += var
        return kde
