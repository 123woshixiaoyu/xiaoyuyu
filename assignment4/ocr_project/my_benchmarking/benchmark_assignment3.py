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
                Sgttot = Sgttot | Sgti
                if debug:
                    plot_images(Sgti,Sei,ji)
            jalls = jalls + (calculate_jaccard(Sgttot,Setot),)
        
        
        if debug:
            print('All Jaccard indices:')
            print(jis)
            print('All total Jaccard indices (total for each image):')
            print(jalls)
        return (jis,jalls)



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
            
            # gtnrs = [int(l) for l in gt]
            
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





def benchmark_assignment3(im2seg, seg2feat, feat2class, classdata, datadir,mode = 2):
    gtnames = glob.glob(os.path.join(datadir,'*txt'))
    segnames = glob.glob(os.path.join(datadir,'*npy'))
    imnames = glob.glob(os.path.join(datadir,'*jpg'))
    segnames.sort()
    gtnames.sort()
    imnames.sort()
    
    
    
    if len(segnames) != len(gtnames):
        print('Not the same number of segments and ground truth text')
        return 0
    else:
        confmat = np.zeros((10,10))
        allX = np.zeros((0,0))
        allY = str()
        jis = ()
        jalls = ()
        # alljs = []
        # alljfg = []
        allres = []
        # nrbcorrect = 0
        for (imname,segname,gtname) in zip(imnames,segnames,gtnames):
            
            # load data
            Sgt = np.load(segname)
            gt_file = open(gtname,'r') 
            gt = gt_file.read()
            gt = gt[:-1] # remove newline character
            gt_file.close()             
            gtnrs = [int(l) for l in gt]
            im = plt.imread(imname)
            
            
            # segment image and check segmentation
            Se = im2seg(im)
            Setot = np.zeros(im.shape,dtype = 'uint8')
            Sgttot = np.zeros(im.shape,dtype = 'uint8')
            for (Sgti,Sei) in zip(Sgt,10*Se):
                ji = calculate_jaccard(Sgti,Sei)
                jis = jis +  (ji,)
                Setot = Setot | Sei
                Sgttot = Sgttot | Sgti
                if mode>=1:
                    plot_images(Sgti,Sei,ji)
            jalls = jalls + (calculate_jaccard(Sgttot,Setot),)
            
            # extract features
            if len(gt) == len(Sgt):
                allY = allY + gt
                for i in range(len(gt)):
                    f = seg2feat(Sgt[i])
                    allX = np.hstack((allX,f)) if allX.size else f
                    
            
            for (gti,Sei) in zip(gtnrs,10*Se):
                f2 = seg2feat(Sei)
                ci = min(max(0,int(feat2class(f2,classdata))),9)
                confmat[gti,ci] += 1
                resi = gti == ci
                allres.append(resi)
                
            
            if mode >=2:
                # plot results
                pass
            
        hitrate = np.sum(allres)/len(allres)
        
        return hitrate,confmat,allres,jis,jalls,allX,allY
        
        
        
        
     
       
      
        
        
      
        
     
    