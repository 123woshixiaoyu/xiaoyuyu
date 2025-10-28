#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 11:37:56 2024

@author: magnuso
"""

import os
import matplotlib.pyplot as plt
import numpy as np

from my_benchmarking.benchmark_assignment1 import benchmark_assignment1


def im2segment(im):
    # return a list of true/false images of the same size as im
    # This is a bad test implementation, please change to your code!
    
    nrofsegments = 5 # could vary, probably you should estimate this
    m,n = im.shape # image size
    S = []
    for kk in range(nrofsegments):
        S =  S + [np.random.rand(m,n)<0.5] # this is not a good segmentation method...
    
    return S


if __name__ == "__main__":
    thisdir = os.path.dirname(os.path.realpath(__file__))
    datadir = os.path.join(thisdir, 'datasets', 'short1')

    # read example image
    im = plt.imread(os.path.join(datadir, 'im1.jpg')) 
    
    # read ground truth numbers
    gt_file = open(os.path.join(datadir, 'im1.txt'),'r') 
    gt = gt_file.read()
    gt = gt[:-1] # remove newline character
    gt_file.close()
    
    # show image with ground truth
    plt.imshow(im)
    plt.title(gt)
    
    # segment example image
    S = im2segment(im)
    
    # Plot all the segments
    fig,axs = plt.subplots(len(S),1) # create n x 1 subplots 
    
    for Si,axi in zip(S,axs):  # loop over segments and subplots
        axi.imshow(Si,cmap = 'gray',vmin = 0, vmax = 1.0)
    
    
    
    # Benchmark your segmentation routine on all images
    
    debug = True
    stats = benchmark_assignment1(im2segment,datadir,debug)
    if stats != 0:
        print(f'Total mean Jaccard score is {np.mean(stats[0]):.2}')
        
