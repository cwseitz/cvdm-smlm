import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib_scalebar.scalebar import ScaleBar
from skimage.feature import blob_log
from skimage.util import img_as_float

class LoGDetector:
    def __init__(self,X,min_sigma=2,max_sigma=5,num_sigma=5,threshold=0.5,
                 overlap=0.5,show_scalebar=True,pixel_size=108.3,r_to_sigraw=3,
                 plot_r=True,blob_marker='x',
                 blob_markersize=10,blob_markercolor=(0,0,1,0.8)):

        self.X = X
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.num_sigma = num_sigma
        self.threshold = threshold
        self.overlap = overlap
        self.show_scalebar = show_scalebar
        self.pixel_size = pixel_size
        self.r_to_sigraw = r_to_sigraw
        self.plot_r = plot_r
        self.blob_marker = blob_marker
        self.blob_markersize = blob_markersize
        self.blob_markercolor = blob_markercolor

    def detect(self):

        blobs = blob_log(img_as_float(self.X),
                         min_sigma=self.min_sigma,
                         max_sigma=self.max_sigma,
                         num_sigma=self.num_sigma,
                         threshold=self.threshold,
                         overlap=self.overlap,
                         )

        columns = ['x', 'y', 'sigma', 'r', 'peak']
        self.blobs_df = pd.DataFrame([], columns=columns)
        self.blobs_df['x'] = blobs[:,0]
        self.blobs_df['y'] = blobs[:,1]
        self.blobs_df['sigma'] = blobs[:, 2]
        self.blobs_df['r'] = blobs[:, 2] * self.r_to_sigraw

        self.blobs_df = self.blobs_df[(self.blobs_df['x'] - self.blobs_df['r'] > 0) &
                      (self.blobs_df['x'] + self.blobs_df['r'] + 1 < self.X.shape[0]) &
                      (self.blobs_df['y'] - self.blobs_df['r'] > 0) &
                      (self.blobs_df['y'] + self.blobs_df['r'] + 1 < self.X.shape[1])]

        for i in self.blobs_df.index:
            x = int(self.blobs_df.at[i, 'x'])
            y = int(self.blobs_df.at[i, 'y'])
            r = int(round(self.blobs_df.at[i, 'r']))
            blob = self.X[x-r:x+r+1, y-r:y+r+1]
            self.blobs_df.at[i, 'peak'] = blob.max()

        return self.blobs_df

    def show(self,X=None,ax=None):

       if ax is None:
           fig, ax = plt.subplots(figsize=(6,6))
       if X is None:
           X = self.X
       ax.imshow(X, cmap="gray", aspect='equal')
       ax.scatter(self.blobs_df['y'],self.blobs_df['x'],color='red',marker='x')
       if self.show_scalebar:
           font = {'family': 'arial', 'weight': 'bold','size': 16}
           scalebar = ScaleBar(self.pixel_size, 'nm', location = 'upper right',
               font_properties=font, box_color = 'black', color='white')
           scalebar.length_fraction = .3
           scalebar.height_fraction = .025
           ax.add_artist(scalebar)
