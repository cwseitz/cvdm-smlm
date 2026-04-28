import pandas as pd
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path
from sklearn.cluster import DBSCAN
from cvdm.psf import LoGDetector
from cvdm.psf.psf2d import MLE2D_BFGS, MLE2D_BFGS_ANISO, MLE2D_BFGS_ROT
from numpy.linalg import inv
import matplotlib.cm as cm
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

class PipelineMLE2D:
    """A collection of functions for maximum likelihood localization"""
    def __init__(self,stack):
        self.stack = stack
    def localize(self,plot_spots=False,plot_fit=False,tmax=None,threshold=0.1,min_sigma=0.75,max_sigma=1.5,n_jobs=1,max_fit_distance=None,cam_params=None,sigma_psf=1.0,fit_enabled=True,max_iters=100,show_tqdm=True,progress_desc=None,patchw=3,fit_model="aniso",sigma_x_init=None,sigma_y_init=None,theta_init=None):
        nt,nx,ny = self.stack.shape
        if tmax is not None: nt = tmax
        spotst = []
        frame_iter = range(nt)
        if show_tqdm:
            frame_iter = tqdm(frame_iter, desc=progress_desc or "Frames")
        for n in frame_iter:
            framed = self.stack[n]
            log = LoGDetector(framed,threshold=threshold,min_sigma=min_sigma,max_sigma=max_sigma)
            spots = log.detect() #image coordinates
            if fit_enabled:
                spots = self.fit(
                    framed,
                    spots,
                    patchw=patchw,
                    max_iters=max_iters,
                    plot_fit=plot_fit,
                    n_jobs=n_jobs,
                    cam_params=cam_params,
                    sigma_psf=sigma_psf,
                    fit_model=fit_model,
                    sigma_x_init=sigma_x_init,
                    sigma_y_init=sigma_y_init,
                    theta_init=theta_init,
                )
            else:
                spots = spots.copy()
            if max_fit_distance is not None and not spots.empty:
                dx = spots['x_mle'] - spots['x']
                dy = spots['y_mle'] - spots['y']
                dist = np.sqrt(dx * dx + dy * dy)
                spots = spots.loc[dist <= max_fit_distance]
            if plot_spots:
                fig, ax = plt.subplots()
                ax.imshow(framed, cmap='gray')
                x_plot = pd.to_numeric(spots['x'], errors='coerce')
                y_plot = pd.to_numeric(spots['y'], errors='coerce')
                det_mask = x_plot.notna() & y_plot.notna()
                ax.scatter(y_plot[det_mask], x_plot[det_mask], c='red', s=15, alpha=0.7, label='Detections')
                if 'x_mle' in spots.columns and 'y_mle' in spots.columns:
                    x_fit = pd.to_numeric(spots['x_mle'], errors='coerce')
                    y_fit = pd.to_numeric(spots['y_mle'], errors='coerce')
                    fit_mask = x_fit.notna() & y_fit.notna()
                    ax.scatter(y_fit[fit_mask], x_fit[fit_mask], c='blue', s=15, alpha=0.7, label='Fits')
                print(f"Frame {n} spots dataframe:")
                print(spots)
                ax.invert_yaxis()
                ax.legend(frameon=False)
                plt.show()
            spots = spots.assign(frame=n)
            spotst.append(spots)
        spotst = pd.concat(spotst)
        return spotst

    @staticmethod
    def _fit_spot(i, x0, y0, adu, patchw, max_iters, plot_fit, cam_params, sigma_psf):
        theta0 = np.array([patchw, patchw, 1.0])
        opt = MLE2D_BFGS(theta0, adu, cam_params=cam_params, sigma_psf=sigma_psf)  # cartesian coordinates with top-left origin
        theta_mle, loglike, conv, err = opt.optimize(max_iters=max_iters, plot_fit=plot_fit)
        dx = theta_mle[0] - patchw
        dy = theta_mle[1] - patchw
        return i, x0 + dx, y0 + dy, theta_mle[2], conv

    @staticmethod
    def _fit_spot_aniso(i, x0, y0, adu, patchw, max_iters, plot_fit, cam_params, sigma_x_init, sigma_y_init, sigma_psf):
        sx0 = sigma_x_init if sigma_x_init is not None else sigma_psf
        sy0 = sigma_y_init if sigma_y_init is not None else sigma_psf
        theta0 = np.array([patchw, patchw, 1.0, sx0, sy0])
        opt = MLE2D_BFGS_ANISO(theta0, adu, cam_params=cam_params)
        theta_mle, loglike, conv, err = opt.optimize(max_iters=max_iters, plot_fit=plot_fit)
        dx = theta_mle[0] - patchw
        dy = theta_mle[1] - patchw
        return i, x0 + dx, y0 + dy, theta_mle[2], theta_mle[3], theta_mle[4], conv

    @staticmethod
    def _fit_spot_rot(i, x0, y0, adu, patchw, max_iters, plot_fit, cam_params, sigma_x_init, sigma_y_init, theta_init, sigma_psf):
        sx0 = sigma_x_init if sigma_x_init is not None else sigma_psf
        sy0 = sigma_y_init if sigma_y_init is not None else sigma_psf
        t0 = theta_init if theta_init is not None else 0.0
        theta0 = np.array([patchw, patchw, 1.0, sx0, sy0, t0])
        opt = MLE2D_BFGS_ROT(theta0, adu, cam_params=cam_params)
        theta_mle, loglike, conv, err = opt.optimize(max_iters=max_iters, plot_fit=plot_fit)
        dx = theta_mle[0] - patchw
        dy = theta_mle[1] - patchw
        return i, x0 + dx, y0 + dy, theta_mle[2], theta_mle[3], theta_mle[4], theta_mle[5], conv

    def fit(self,frame,spots,patchw=3,max_iters=100,plot_fit=False,n_jobs=1,cam_params=None,sigma_psf=1.0,fit_model="aniso",sigma_x_init=None,sigma_y_init=None,theta_init=None):
        if n_jobs is None or n_jobs < 2:
            for i in spots.index:
                x0 = int(spots.at[i,'x']) #image coordinates (row)
                y0 = int(spots.at[i,'y']) #image coordinates (column)
                adu = frame[x0-patchw:x0+patchw+1,y0-patchw:y0+patchw+1]
                adu = np.clip(adu,0,None)
                if fit_model == "rot":
                    i, x_mle, y_mle, n0, sx, sy, theta, conv = self._fit_spot_rot(
                        i, x0, y0, adu, patchw, max_iters, plot_fit, cam_params, sigma_x_init, sigma_y_init, theta_init, sigma_psf
                    )
                    spots.at[i, 'x_mle'] = x_mle
                    spots.at[i, 'y_mle'] = y_mle
                    spots.at[i, 'N0'] = n0
                    spots.at[i, 'sigma_x'] = sx
                    spots.at[i, 'sigma_y'] = sy
                    spots.at[i, 'theta'] = theta
                    spots.at[i, 'conv'] = conv
                elif fit_model == "aniso":
                    i, x_mle, y_mle, n0, sx, sy, conv = self._fit_spot_aniso(
                        i, x0, y0, adu, patchw, max_iters, plot_fit, cam_params, sigma_x_init, sigma_y_init, sigma_psf
                    )
                    spots.at[i, 'x_mle'] = x_mle
                    spots.at[i, 'y_mle'] = y_mle
                    spots.at[i, 'N0'] = n0
                    spots.at[i, 'sigma_x'] = sx
                    spots.at[i, 'sigma_y'] = sy
                    spots.at[i, 'conv'] = conv
                else:
                    i, x_mle, y_mle, n0, conv = self._fit_spot(i, x0, y0, adu, patchw, max_iters, plot_fit, cam_params, sigma_psf)
                    spots.at[i, 'x_mle'] = x_mle
                    spots.at[i, 'y_mle'] = y_mle
                    spots.at[i, 'N0'] = n0
                    spots.at[i, 'conv'] = conv
            return spots

        if plot_fit:
            print("plot_fit=True is not supported with parallel fitting; disabling plot_fit.")
            plot_fit = False

        tasks = []
        for i in spots.index:
            x0 = int(spots.at[i,'x']) #image coordinates (row)
            y0 = int(spots.at[i,'y']) #image coordinates (column)
            adu = frame[x0-patchw:x0+patchw+1,y0-patchw:y0+patchw+1]
            adu = np.clip(adu,0,None)
            if fit_model == "rot":
                tasks.append((i, x0, y0, adu, patchw, max_iters, plot_fit, cam_params, sigma_x_init, sigma_y_init, theta_init, sigma_psf))
            elif fit_model == "aniso":
                tasks.append((i, x0, y0, adu, patchw, max_iters, plot_fit, cam_params, sigma_x_init, sigma_y_init, sigma_psf))
            else:
                tasks.append((i, x0, y0, adu, patchw, max_iters, plot_fit, cam_params, sigma_psf))

        results = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            if fit_model == "rot":
                futures = [executor.submit(self._fit_spot_rot, *task) for task in tasks]
            elif fit_model == "aniso":
                futures = [executor.submit(self._fit_spot_aniso, *task) for task in tasks]
            else:
                futures = [executor.submit(self._fit_spot, *task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())

        if fit_model == "rot":
            for i, x_mle, y_mle, n0, sx, sy, theta, conv in results:
                spots.at[i, 'x_mle'] = x_mle
                spots.at[i, 'y_mle'] = y_mle
                spots.at[i, 'N0'] = n0
                spots.at[i, 'sigma_x'] = sx
                spots.at[i, 'sigma_y'] = sy
                spots.at[i, 'theta'] = theta
                spots.at[i, 'conv'] = conv
        elif fit_model == "aniso":
            for i, x_mle, y_mle, n0, sx, sy, conv in results:
                spots.at[i, 'x_mle'] = x_mle
                spots.at[i, 'y_mle'] = y_mle
                spots.at[i, 'N0'] = n0
                spots.at[i, 'sigma_x'] = sx
                spots.at[i, 'sigma_y'] = sy
                spots.at[i, 'conv'] = conv
        else:
            for i, x_mle, y_mle, n0, conv in results:
                spots.at[i, 'x_mle'] = x_mle
                spots.at[i, 'y_mle'] = y_mle
                spots.at[i, 'N0'] = n0
                spots.at[i, 'conv'] = conv

        return spots

    def dbscan(self,data,eps=3,min_samples=10,plot=False):
        coords = data[['x_mle','y_mle']].values
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        data['cluster'] = db.labels_
        filtered_count = int((data['cluster'] == -1).sum())
        print(f"DBSCAN filtered out {filtered_count} points")
        if plot:
            fig,ax=plt.subplots()
            ax.imshow(np.sum(self.stack,axis=0),cmap='gray')
            unique_clusters = data['cluster'].unique()
            colors = cm.get_cmap('tab20b', len(unique_clusters))
            for i, cluster in enumerate(unique_clusters):
                cluster_data = data[data['cluster'] == cluster]
                color = colors(i) if cluster != -1 else 'red'
                ax.scatter(cluster_data['y_mle'],cluster_data['x_mle'],c=[color], s=20, alpha=0.7, 
                            label=f'Cluster {cluster}' if cluster != -1 else 'Noise')
                ax.invert_yaxis()
            plt.show()
        data = data.loc[data['cluster'] != -1]
        return data
