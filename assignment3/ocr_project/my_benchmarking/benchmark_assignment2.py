#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 17:05:47 2024

@author: magnuso
"""


import glob
import os
import numpy as np
import matplotlib.pyplot as plt



def benchmark_assignment2(seg2feat,datadir, debug = False):
    gtnames = glob.glob(os.path.join(datadir,'*txt'))
    segnames = glob.glob(os.path.join(datadir,'*npy'))
    segnames.sort()
    gtnames.sort()
    
    if len(segnames) != len(gtnames):
        print('Not the same number of segments and ground truth text')
        return 0
    else:
        allX = np.zeros((0,0))
        allY = str()
        for (segname,gtname) in zip(segnames,gtnames):
            Sgt = np.load(segname)
           
            gt_file = open(gtname,'r') 
            gt = gt_file.read()
            gt = gt[:-1] # remove newline character
            gt_file.close() 
            
            if len(gt) == len(Sgt):
                allY = allY + gt
                for i in range(len(gt)):
                    f = seg2feat(Sgt[i])
                    allX = np.hstack((allX,f)) if allX.size else f
                    
                    
           
           
        
        
        if debug:
            U,_ ,_ = np.linalg.svd(allX)
            allX_red = np.transpose(U[:,0:2])@allX
            plt.plot()
            for i in range(len(allY)):
                plt.text(allX_red[0,i],allX_red[1,i],allY[i])
              
            plt.xlim(allX_red[0,:].min(),allX_red[0,:].max())
            plt.ylim(allX_red[1,:].min(),allX_red[1,:].max())
            plt.show()
            
        return (allX,allY)
