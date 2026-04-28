import numpy as np
from numpy.linalg import inv
from .psf2d import *
import matplotlib.pyplot as plt

def crlb2d(theta,cam_params,npix=10,sigma=1.0,B0=0):
    ntheta = len(theta)
    x0,y0,N0 = theta
    x = np.arange(0,npix); y = np.arange(0,npix)
    X,Y = np.meshgrid(x,y,indexing='ij')
    lam = lamx(X,x0,sigma)*lamy(Y,y0,sigma)
    mu = N0*lam + B0
    J = jac1(X,Y,theta,cam_params,sigma=sigma)
    I = np.zeros((ntheta,ntheta))
    for n in range(ntheta):
       for m in range(ntheta):
           I[n,m] = np.sum(J[n]*J[m]/mu)
    return np.sqrt(np.diagonal(inv(I)))
