import numpy as np
from skimage.io import imread,imsave

def custom_zoom(X):
    m, n = X.shape
    zoomed = np.zeros((2*m,2*n))
    zoomed[::2,::2] = X
    zoomed = np.pad(zoomed,((1,1),(1,1)),mode='constant')
    nx,ny = zoomed.shape
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            if i % 2 == 0 and j % 2 != 0:
                zoomed[i,j] = (zoomed[i-1,j]+zoomed[i+1,j])/2
            if i % 2 != 0 and j % 2 == 0:
                zoomed[i,j] = (zoomed[i,j-1]+zoomed[i,j+1])/2
            if i % 2 == 0 and j % 2 == 0:
                zoomed[i,j] = (zoomed[i-1,j-1]+zoomed[i+1,j+1])/4
                zoomed[i,j] += (zoomed[i+1,j-1]+zoomed[i-1,j+1])/4
    zoomed = zoomed[1:-1,1:-1]
    return zoomed
