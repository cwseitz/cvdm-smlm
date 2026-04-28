import numpy as np
from skimage.restoration import rolling_ball
from os.path import basename
from skimage.io import imread,imsave
from itertools import groupby
import pandas as pd
import matplotlib.pyplot as plt

def errors2d(xy,xy_est,tol=5.0):
    def match_on_xy(xy_est,xy,tol=5.0):
        #print(xy_est,xy)
        dist = np.linalg.norm(xy[:2,:].T-xy_est, axis=1)
        a = dist <= tol
        b = np.any(a)
        if b:
            idx = np.argmin(dist)
            c = np.squeeze(xy[:2,idx])
            N0 = xy[3,idx]
            xerr,yerr = xy_est[0]-c[0],xy_est[1]-c[1]
            xy = np.delete(xy,idx,axis=1)
        else:
            xerr,yerr,idx,N0 = None,None,None,None
        return xy,b,xerr,yerr,idx,N0

    num_found,_ = xy_est.shape
    _,nspots = xy.shape
    bfound = []; all_x_err = []; all_y_err = []
    all_label = []; all_N0 = []

    for n in range(num_found):
        this_xy_est = xy_est[n,:]
        if xy.shape[1] > 0:
            xy,bool,xerr,yerr,label,N0 = match_on_xy(this_xy_est,xy,tol=tol)
            bfound.append(bool)
            if xerr is not None and yerr is not None:
                all_x_err.append(xerr)
                all_y_err.append(yerr)   
                all_label.append(label)
                all_N0.append(N0)
  
    inter = np.sum(np.array(bfound).astype(int)) #intersection
    fp = len(bfound)-sum(bfound) #num found - matched
    union = nspots+fp
    fn = nspots - inter
    all_x_err = np.array(all_x_err)
    all_y_err = np.array(all_y_err)
    all_label = np.array(all_label)
    all_N0 = np.array(all_N0)
    return all_x_err, all_y_err, all_label, all_N0, inter, union, fp, fn


def prepare_data(path,xfiles,yfiles,zfiles):
    Xstack = []; Ystack = []; Zstack = []
    for xfile,yfile,zfile in zip(xfiles,yfiles,zfiles):
        X = imread(path+'/'+xfile)
        Y = imread(path+'/'+yfile)
        Z = imread(path+'/'+zfile)
        Xstack.append(X); Ystack.append(Y); Zstack.append(Z)
    Xstack = np.array(Xstack); Ystack = np.array(Ystack); Zstack = np.array(Zstack)
    return Xstack, Ystack, Zstack

def get_errors(spots,coordsgt,upsample=4,tol=5.0):
    errordfs = []; set_metrics = []
    coordsgt[0,:] *= upsample
    coordsgt[1,:] *= upsample
    for frame in spots['frame'].unique():
        coords = spots.loc[spots['frame']==frame]
        coords = coords[['x_mle','y_mle']].values
        all_x_err, all_y_err, all_label, all_N0, inter, union, fp, fn =\
        errors2d(coordsgt,coords,tol=tol)
        errordf = pd.DataFrame({'x_err':all_x_err,'y_err':all_y_err,'label':all_label,'N0':all_N0})
        errordfs.append(errordf)
        set_metrics.append([inter,union,fp,fn])
    errordfs_out = pd.concat(errordfs)
    return errordfs_out,set_metrics


def sort_and_group(files):
    files = [basename(file) for file in files]
    sort = sorted(files, key=lambda x: (int(x.split('-')[1]), int(x.split('-')[2].split('.')[0])))
    sort = {k: list(v) for k, v in groupby(sort, key=lambda x: int(x.split('-')[1]))}
    return sort

def show_images(X1x,Y,Z,coords,coordsgt,upsample=4):
    fig,ax=plt.subplots(1,3)
    ax[0].imshow(X1x,cmap='gray')
    ax[0].scatter(coordsgt[:,1]/upsample,coordsgt[:,0]/upsample,
              marker='x',color='red',s=3)
    ax[1].scatter(coordsgt[:,1],coordsgt[:,0],
              marker='x',color='red',s=3)
    ax[2].scatter(coordsgt[:,1],coordsgt[:,0],
              marker='x',color='red',s=3)
    ax[2].scatter(coords[:,1],coords[:,0],
              marker='x',color='blue',s=3)
    ax[1].imshow(Y,cmap='gray')
    ax[2].imshow(Z,cmap='gray')
    plt.show()
