#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 11:37:56 2024

@author: magnuso
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

from local_benchmarking.benchmark_assignment1 import benchmark_assignment1
from main_smp import im2segment

if __name__ == "__main__":

    # read example image
    im = plt.imread('im1.jpg')
    
    # read ground truth numbers
    gt_file = open('datasets/short1/im1.txt','r')
    gt = gt_file.read()
    gt = gt[:-1] # remove newline character
    gt_file.close()
    
    # show image with ground truth
    plt.imshow(im)
    plt.title(gt)
    
    # segment example image
    S = im2segment(im)

    fig, axs = plt.subplots(len(S), 1)

    # 确保 axs 可迭代
    if len(S) == 1:
        axs = [axs]

    for Si, axi in zip(S, axs):
        axi.imshow(Si, cmap='gray', vmin=0, vmax=1.0)

    # Benchmark your segmentation routine on all images
    
    datadir = os.path.join('datasets','short1')
    debug = True
    stats = benchmark_assignment1(im2segment,datadir,debug)
    if stats != 0:
        print(f'Total mean Jaccard score is {np.mean(stats[0]):.2}')
        
