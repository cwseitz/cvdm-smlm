import numpy as np
from .psf2d import *

def jaciso2d(theta,adu,cam_params,sigma=1.0):
    nx,ny = adu.shape
    ntheta = len(theta)
    x0,y0,N0 = theta
    X,Y = np.meshgrid(np.arange(0,nx),np.arange(0,ny),indexing='ij')
    J1 = jac1(X,Y,theta,cam_params,sigma=sigma)
    J1 = J1.reshape((ntheta,nx**2))
    J2 = jac2(adu,X,Y,theta,cam_params,sigma=sigma)
    J = J1 @ J2
    return J
