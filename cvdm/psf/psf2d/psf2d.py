from numpy import exp, pi, sqrt
from scipy.special import erf
import matplotlib.pyplot as plt
import numpy as np

def lamx(X,x0,sigma):
    alpha = sqrt(2)*sigma
    return 0.5*(erf((X+0.5-x0)/alpha)-erf((X-0.5-x0)/alpha))
    
def lamy(Y,y0,sigma):
    alpha = sqrt(2)*sigma
    return 0.5*(erf((Y+0.5-y0)/alpha)-erf((Y-0.5-y0)/alpha))

def dudn0(X,Y,x0,y0,sigma):
    return lamx(X,x0,sigma)*lamy(Y,y0,sigma)
     
def dudx0(X,Y,x0,y0,sigma):
    A = 1/(sqrt(2*pi)*sigma)
    return A*lamy(Y,y0,sigma)*(exp(-(X-0.5-x0)**2/(2*sigma**2))-exp(-(X+0.5-x0)**2/(2*sigma**2)))
    
def dudy0(X,Y,x0,y0,sigma):
    A = 1/(sqrt(2*pi)*sigma)
    return A*lamx(X,x0,sigma)*(exp(-(Y-0.5-y0)**2/(2*sigma**2))-exp(-(Y+0.5-y0)**2/(2*sigma**2)))

#share dudsx() and dudsy() with psf3d for convenience

def dudsx(X,Y,x0,y0,sigma_x,sigma_y):
    A = 1/(sqrt(2*pi)*sigma_x**2)
    return A*lamy(Y,y0,sigma_y)*((X-x0-0.5)*exp(-(X-0.5-x0)**2/(2*sigma_x**2))-(X-x0+0.5)*exp(-(X+0.5-x0)**2/(2*sigma_x**2)))
    
def dudsy(X,Y,x0,y0,sigma_x,sigma_y):
    A = 1/(sqrt(2*pi)*sigma_y**2)
    return A*lamx(X,x0,sigma_x)*((Y-y0-0.5)*exp(-(Y-0.5-y0)**2/(2*sigma_y**2))-(Y-y0+0.5)*exp(-(Y+0.5-y0)**2/(2*sigma_y**2)))

def duds0(X,Y,x0,y0,sigma):
    return dudsx(X,Y,x0,y0,sigma,sigma)+dudsy(X,Y,x0,y0,sigma,sigma)
    
def jac1(X,Y,theta,cam_params,sigma=1.0):
    x0,y0,N0 = theta
    #x0,y0,sigma,N0 = theta
    eta,texp,gain,offset,var = cam_params
    i0 = N0*eta*gain*texp
    j_x0 = i0*dudx0(X,Y,x0,y0,sigma)
    j_y0 = i0*dudy0(X,Y,x0,y0,sigma)
    j_s0 = i0*duds0(X,Y,x0,y0,sigma)
    j_n0 = (i0/N0)*dudn0(X,Y,x0,y0,sigma)
    jac = np.array([j_x0, j_y0, j_n0], dtype=np.float64)
    #jac = np.array([j_x0, j_y0, j_s0, j_n0], dtype=np.float64)
    return jac
    
def jac2(adu,X,Y,theta,cam_params,sigma=1.0):
    x0,y0,N0 = theta
    #x0,y0,sigma,N0 = theta
    eta,texp,gain,offset,var = cam_params
    i0 = N0*eta*gain*texp
    lam = lamx(X,x0,sigma)*lamy(Y,y0,sigma)
    mu = i0*lam + var
    jac2 = 1 - adu/mu
    return jac2.flatten()

