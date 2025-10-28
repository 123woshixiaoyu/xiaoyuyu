#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 14:14:33 2024

@author: magnuso
"""
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import time

def calculate_jaccard(im1,im2):
    fg1 = im1 > 0
    fg2 = im2 > 0
    js = np.sum(fg1 & fg2)/np.sum(fg1 | fg2)
    return js

def plot_images(im1,im2,ji):
    fig,axs = plt.subplots(2,1)
    axs[0].imshow(im1,cmap = 'gray')
    axs[1].imshow(im2, cmap = 'gray')
    plt.title(f'Jaccard index: {ji:.3}')
    plt.show()
    time.sleep(0.5)

def benchmark_assignment1(segmenter,datadir, debug = False):
    imnames = glob.glob(os.path.join(datadir,'*jpg'))
    gtnames = glob.glob(os.path.join(datadir,'*npy'))
    imnames.sort()
    gtnames.sort()
    
    if len(imnames) != len(gtnames):
        print('Not the same number of images and ground truth')
        return 0
    else:
        jis = ()
        jalls = ()
        for (imname,gtname) in zip(imnames,gtnames):
            im = plt.imread(imname)
            Sgt = np.load(gtname)
            Se = segmenter(im)
            Setot = np.zeros(im.shape,dtype = 'uint8')
            Sgttot = np.zeros(im.shape,dtype = 'uint8')
            for (Sgti,Sei) in zip(Sgt,10*Se):
                ji = calculate_jaccard(Sgti,Sei)
                jis = jis +  (ji,)
                Setot = Setot | Sei
                Sgttot = Sgttot | Sgt
                if debug:
                    plot_images(Sgti,Sei,ji)
            jalls = jalls + (calculate_jaccard(Sgttot,Setot),)
        
        
        if debug:
            print('All Jaccard indices:')
            print(jis)
            print('All total Jaccard indices (total for each image):')
            print(jalls)
        return (jis,jalls)
