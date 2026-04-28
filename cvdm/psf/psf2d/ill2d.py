import numpy as np
import warnings
from .psf2d import *

def isologlike2d(theta,adu,cam_params,sigma=1.0):
    nx,ny = adu.shape
    x0,y0,N0 = theta
    eta,texp,gain,offset,var = cam_params
    X,Y = np.meshgrid(np.arange(0,nx),np.arange(0,ny),indexing='ij')
    lam = lamx(X,x0,sigma)*lamy(Y,y0,sigma)
    i0 = gain*eta*texp*N0
    muprm = i0*lam + var
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    stirling = adu * np.nan_to_num(np.log(adu)) - adu
    p = adu*np.log(muprm)
    warnings.filterwarnings("default", category=RuntimeWarning)
    p = np.nan_to_num(p)
    nll = stirling + muprm - p
    nll = np.sum(nll)
    return nll


def isologlike2d_aniso(theta, adu, cam_params):
    nx, ny = adu.shape
    x0, y0, N0, sigma_x, sigma_y = theta
    eta, texp, gain, offset, var = cam_params
    X, Y = np.meshgrid(np.arange(0, nx), np.arange(0, ny), indexing='ij')
    lam = lamx(X, x0, sigma_x) * lamy(Y, y0, sigma_y)
    i0 = gain * eta * texp * N0
    muprm = i0 * lam + var
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    stirling = adu * np.nan_to_num(np.log(adu)) - adu
    p = adu * np.log(muprm)
    warnings.filterwarnings("default", category=RuntimeWarning)
    p = np.nan_to_num(p)
    nll = stirling + muprm - p
    nll = np.sum(nll)
    return nll


def isologlike2d_rot(theta, adu, cam_params):
    nx, ny = adu.shape
    x0, y0, N0, sigma_x, sigma_y, angle = theta
    eta, texp, gain, offset, var = cam_params
    X, Y = np.meshgrid(np.arange(0, nx), np.arange(0, ny), indexing='ij')
    Xc = X - x0
    Yc = Y - y0
    cos_t = np.cos(angle)
    sin_t = np.sin(angle)
    xr = cos_t * Xc + sin_t * Yc
    yr = -sin_t * Xc + cos_t * Yc
    sigma_x = max(float(sigma_x), 1e-6)
    sigma_y = max(float(sigma_y), 1e-6)
    g = np.exp(-0.5 * ((xr / sigma_x) ** 2 + (yr / sigma_y) ** 2)) / (2.0 * np.pi * sigma_x * sigma_y)
    i0 = gain * eta * texp * N0
    muprm = i0 * g + var
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    stirling = adu * np.nan_to_num(np.log(adu)) - adu
    p = adu * np.log(muprm)
    warnings.filterwarnings("default", category=RuntimeWarning)
    p = np.nan_to_num(p)
    nll = stirling + muprm - p
    nll = np.sum(nll)
    return nll
