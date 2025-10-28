#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 09:33:12 2024

@author: magnuso
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt

def colorSpaceTransform(inputColor, fromSpace2toSpace, firstInChain, lastInChain):
    # Check if inputColor is first part of transform chain. If so,
    # transform layout
    dim = inputColor.shape
    if firstInChain:
        # Transform HxWx3 image to 3xHW for easier processing
        inputColor = inputColor.transpose((2, 0, 1)).reshape((dim[2], dim[0] * dim[1]))

    if fromSpace2toSpace == 'srgb2linrgb':
        limit = 0.04045
        allAboveLimit = inputColor > limit
        transformedColor = np.zeros(inputColor.shape)
        transformedColor[allAboveLimit] = np.power(((inputColor[allAboveLimit] + 0.055) / 1.055) , 2.4)
        transformedColor[~allAboveLimit] = inputColor[~allAboveLimit] / 12.92

    elif fromSpace2toSpace == 'linrgb2srgb':
        limit = 0.0031308
        allAboveLimit = inputColor > limit;
        transformedColor = np.zeros(inputColor.shape)
        transformedColor[allAboveLimit] = 1.055 * np.power(inputColor[allAboveLimit] , (1.0 / 2.4)) - 0.055
        transformedColor[~allAboveLimit] = 12.92 * inputColor[~allAboveLimit]

    elif (fromSpace2toSpace == 'linrgb2xyz') | (fromSpace2toSpace == 'xyz2linrgb'):
        # Source: https://www.image-engineering.de/library/technotes/958-how-to-convert-between-srgb-and-ciexyz
        # Assumes D65 standard illuminant
        a11 = 10135552 / 24577794
        a12 = 8788810 / 24577794
        a13 = 4435075 / 24577794
        a21 = 2613072 / 12288897
        a22 = 8788810 / 12288897
        a23 = 887015 / 12288897
        a31 = 1425312 / 73733382
        a32 = 8788810 / 73733382
        a33 = 70074185 / 73733382
        A = np.array([[a11,a12,a13],
             [a21,a22,a23],
             [a31,a32,a33]])
        if fromSpace2toSpace == 'linrgb2xyz':
            transformedColor = A @ inputColor
        else:
            Ai = np.linalg.inv(A)
            transformedColor = Ai @ inputColor
        

    elif fromSpace2toSpace == 'xyz2ycxcz':
        reference_illuminant = colorSpaceTransform(np.ones((1,1,3)), 'linrgb2xyz', 1, 1)
        reference_illuminant = reference_illuminant.transpose((2, 0, 1)).reshape((3, 1))
        reference_illuminant_matrix = np.tile(reference_illuminant, (1, inputColor.size // 3))
        inputColor = inputColor / reference_illuminant_matrix
        Y = 116 * inputColor[1, :] - 16
        Cx = 500 * (inputColor[0,:] - inputColor[1, :])
        Cz = 200 * (inputColor[1, :] - inputColor[2, :])
        transformedColor = np.array([Y,Cx,Cz])

    elif fromSpace2toSpace =='ycxcz2xyz':
        Yy = (inputColor[0, :] + 16) / 116
        Cx = inputColor[1,:] / 500
        Cz = inputColor[2,:] / 200

        X = Yy + Cx
        Y = Yy
        Z = Yy - Cz
        transformedColor = np.array([X,Y,Z])

        reference_illuminant = colorSpaceTransform(np.ones((1,1,3)), 'linrgb2xyz', 1, 1)
        reference_illuminant = reference_illuminant.transpose((2, 0, 1)).reshape((3, 1))
        reference_illuminant_matrix = np.tile(reference_illuminant, (1, transformedColor.size // 3))
        transformedColor = transformedColor * reference_illuminant_matrix
    elif fromSpace2toSpace =='xyz2lab':
        reference_illuminant = colorSpaceTransform(np.ones((1,1,3)), 'linrgb2xyz', 1, 1)
        reference_illuminant = reference_illuminant.transpose((2, 0, 1)).reshape((3, 1))
        reference_illuminant_matrix = np.tile(reference_illuminant, (1, inputColor.size // 3))
        inputColor = inputColor / reference_illuminant_matrix
        delta = 6 / 29
        limit = 0.008856
        allAboveLimit = inputColor > limit
        inputColor[allAboveLimit] = np.power(inputColor[allAboveLimit],(1/3))
        inputColor[~allAboveLimit] = inputColor[~allAboveLimit] / (3 * delta * delta) + 4 / 29
        L = 116 * inputColor[1, :] - 16
        a = 500 * (inputColor[0,:] - inputColor[1, :])
        b = 200 * (inputColor[1, :] - inputColor[2, :])
        transformedColor = np.array([L,a,b])

    elif fromSpace2toSpace == 'lab2xyz':
        L = (inputColor[0, :] + 16) / 116
        a = inputColor[1,:] / 500
        b = inputColor[2,:] / 200

        X = L + a
        Y = L
        Z = L - b
        transformedColor = np.array([X,Y,Z])

        delta = 6/29
        allAboveDelta = transformedColor > delta
        transformedColor[allAboveDelta] = np.power(transformedColor[allAboveDelta],3)
        transformedColor[~allAboveDelta] = 3 * delta*delta * (transformedColor[~allAboveDelta] - 4 / 29)

        reference_illuminant = colorSpaceTransform(np.ones((3, 1)), 'linrgb2xyz', 1, 1)
        reference_illuminant = reference_illuminant.transpose((2,0,1)).reshape((3, 1))
        reference_illuminant_matrix = np.tile(reference_illuminant, (1, transformedColor.size // 3))
        transformedColor = transformedColor * reference_illuminant_matrix;

    elif fromSpace2toSpace =='srgb2xyz':
        transformedColor = colorSpaceTransform(inputColor, 'srgb2linrgb', 1, 0)
        transformedColor = colorSpaceTransform(transformedColor, 'linrgb2xyz', 0, 0)
        lastInChain = 1
    elif fromSpace2toSpace =='srgb2ycxcz':
        transformedColor = colorSpaceTransform(inputColor, 'srgb2linrgb', 1, 0)
        transformedColor = colorSpaceTransform(transformedColor, 'linrgb2xyz', 0, 0)
        transformedColor = colorSpaceTransform(transformedColor, 'xyz2ycxcz', 0, 0)
        lastInChain = 1
    elif fromSpace2toSpace == 'linrgb2ycxcz':
        transformedColor = colorSpaceTransform(inputColor, 'linrgb2xyz', 1, 0)
        transformedColor = colorSpaceTransform(transformedColor, 'xyz2ycxcz', 0, 0)
        lastInChain = 1
    elif fromSpace2toSpace == 'srgb2lab':
        transformedColor = colorSpaceTransform(inputColor, 'srgb2linrgb', 1, 0)
        transformedColor = colorSpaceTransform(transformedColor, 'linrgb2xyz', 0, 0)
        transformedColor = colorSpaceTransform(transformedColor, 'xyz2lab', 0, 0)
        lastInChain = 1
    elif fromSpace2toSpace == 'linrgb2lab':
        transformedColor = colorSpaceTransform(inputColor, 'linrgb2xyz', 1, 0)
        transformedColor = colorSpaceTransform(transformedColor, 'xyz2lab', 0, 0)
        lastInChain = 1
    elif fromSpace2toSpace =='ycxcz2linrgb':
        transformedColor = colorSpaceTransform(inputColor, 'ycxcz2xyz', 1, 0)
        transformedColor = colorSpaceTransform(transformedColor, 'xyz2linrgb', 0, 0)
        lastInChain = 1
    elif fromSpace2toSpace == 'lab2srgb':
        transformedColor = colorSpaceTransform(inputColor, 'lab2xyz', 1, 0)
        transformedColor = colorSpaceTransform(transformedColor, 'xyz2linrgb', 0, 0)
        transformedColor = colorSpaceTransform(transformedColor, 'linrgb2srgb', 0, 0)
        lastInChain = 1
    elif fromSpace2toSpace == 'ycxcz2lab':
        transformedColor = colorSpaceTransform(inputColor, 'ycxcz2xyz', 1, 0)
        transformedColor = colorSpaceTransform(transformedColor, 'xyz2lab', 0, 0)
        lastInChain = 1
    else:
        print('The color transform is not defined!')
        print(fromSpace2toSpace)
        transformedColor = inputColor

    # Transform back to HxWx3 layout if transform is last in chain (HxWx1 in case of grayscale)
    if lastInChain:
        transformedColor = transformedColor.reshape((transformedColor.shape[0], dim[0], dim[1])).transpose((1, 2, 0))
    
    return transformedColor






def  generateSpatialFilter(PixelsPerDegree, channel):
    g = lambda x, a1, b1, a2, b2: a1 * np.sqrt(np.pi / b1) * np.exp(-np.pi*np.pi * x / b1) + a2 * np.sqrt(np.pi / b2) * np.exp(- np.pi*np.pi * x / b2) 
    # Square of x cancels with the sqrt in the distance calculation

    # Set parameters based on which channel the filter will be used for
    a1_A = 1
    b1_A = 0.0047
    a2_A = 0
    b2_A = 1e-5 # avoid division by 0
    a1_rg = 1
    b1_rg = 0.0053
    a2_rg = 0
    b2_rg = 1e-5 # avoid division by 0
    a1_by = 34.1
    b1_by = 0.04
    a2_by = 13.5
    b2_by = 0.025
    if channel == 'A': # Achromatic CSF
        a1 = a1_A
        b1 = b1_A
        a2 = a2_A
        b2 = b2_A
    elif channel == 'RG': # Red-Green CSF
        a1 = a1_rg
        b1 = b1_rg
        a2 = a2_rg
        b2 = b2_rg
    elif channel == 'BY': # Blue-Yellow CSF
        a1 = a1_by
        b1 = b1_by
        a2 = a2_by
        b2 = b2_by
    else:
        print('You can not filter this channel! Check generateSpatialFilter.m.')
    

    # Determine evaluation domain
    maxScaleParameter = max((b1_A, b2_A, b1_rg, b2_rg, b1_by, b2_by))
    radius = np.ceil(3 * np.sqrt(maxScaleParameter / (2 * np.pi*np.pi)) * PixelsPerDegree)
    xx, yy = np.meshgrid(np.arange(-radius,radius+1),np.arange(-radius,radius+1))
    deltaX = 1 / PixelsPerDegree
    xx = xx * deltaX
    yy = yy * deltaX
    d = xx * xx + yy * yy

    # Generate filter and normalize sum to 1
    s = g(d, a1, b1, a2, b2)
    s = s / np.sum(s)
    return s


def spatialFilter(I, s_a, s_rg, s_by):
    # Filters image I using Contrast Sensitivity Functions.
    # Returns linear RGB

    # Apply Gaussian filters
    
    
    ITildeOpponent = np.zeros(I.shape)
    ITildeOpponent[:, :, 0] = scipy.signal.convolve2d(I[:, :, 0], s_a, mode = 'same')
    ITildeOpponent[:, :, 1] = scipy.signal.convolve2d(I[:, :, 1], s_rg, mode = 'same')
    ITildeOpponent[:, :, 2] = scipy.signal.convolve2d(I[:, :, 2], s_by, mode = 'same')
    
    # Transform to linear RGB for clamp
    ITildeLinearRGB = colorSpaceTransform(ITildeOpponent, 'ycxcz2linrgb', 0, 0)
     
    
    # Clamp to RGB box
    ITildeLinearRGB = np.maximum(np.minimum(ITildeLinearRGB, 1), 0)
    return ITildeLinearRGB 



def huntAdjustment(I):
    # Applies Hunt adjustment to L*a*b* image I
    
    # Extract luminance component
    L = I[:, :, 0]
    
    # Apply Hunt adjustment
    Ih = np.zeros(I.shape)
    Ih[:, :, 0] = L
    Ih[:, :, 1] = (0.01 * L) * I[:, :, 1]
    Ih[:, :, 2] = (0.01 * L) * I[:, :, 2]
    return Ih




def HyAB(reference, test):
    # Computes HyAB distance between L*a*b* images reference and test
    delta = reference - test
    d = np.abs(delta[:, :, 0]) + np.linalg.norm(delta[:, :, 1:], axis = 2)
    return d



def redistributeErrors(powerDeltaEhyab, cmax):
    # Set redistribution parameters
    pc = 0.4
    pt = 0.95
    cmax = cmax[0]
    
    # Re-map error to 0-1 range. Values between 0 and
    # pccmax are mapped to the range [0, pt],
    # while the rest are mapped to the range (pt, 1]
    
    deltaEc = np.zeros(powerDeltaEhyab.shape)
    pccmax = pc * cmax
    cutoff = powerDeltaEhyab < pccmax
    
     
    #deltaEc[cutoff] = (pt / pccmax) * powerDeltaEhyab[cutoff]
    deltaEc[cutoff] = (pt / pccmax)*powerDeltaEhyab[cutoff]
    
    
    deltaEc[~cutoff] = pt + ((powerDeltaEhyab[~cutoff] - pccmax) / (cmax - pccmax)) * (1.0 - pt)
    return deltaEc


def featureDetection(Iy, PixelsPerDegree, featureType):
    # Finds features of type featureType in image Iy based on current PPD
    
    # Set peak to trough value (2x standard deviations) of human edge
    # detection filter
    w = 0.082
    
    # Compute filter radius
    sd = 0.5 * w * PixelsPerDegree
    radius = np.ceil(3 * sd)

    # Compute 2D Gaussian
    x,y = np.meshgrid(np.arange(-radius,radius+1),np.arange(-radius,radius+1))
    
    
    g = np.exp(-(x*x + y*y) / (2 * sd*sd))
    
    if featureType == 'edge': # Edge detector
        # Compute partial derivative in x-direction
        Gx = -x * g
    else: # Point detector
        # Compute second partial derivative in x-direction
        Gx = (x*x / (sd*sd) - 1) * g
 
 
    # Normalize positive weights to sum to 1 and negative weights to sum to -1
    negativeWeightsSum = -sum(Gx[Gx < 0])
    positiveWeightsSum = sum(Gx[Gx > 0])
    Gx[Gx < 0] = Gx[Gx < 0] / negativeWeightsSum
    Gx[Gx > 0] = Gx[Gx > 0] / positiveWeightsSum
    
    # Symmetry yields the y-direction filter
    Gy = Gx.transpose()
    
    # Detect features
    
    featuresX = scipy.signal.convolve2d(Iy,Gx, mode = 'same')
    featuresY = scipy.signal.convolve2d(Iy,Gy, mode = 'same')
    
    
    features = np.array([featuresX, featuresY]).transpose((1,2,0))
    return features


def myprint(A):
    bnd = 150
    if A.ndim == 3:
        print(np.mean(A[(bnd-1):-bnd,(bnd-1):-bnd,:]))
    else:
        print(np.mean(A[(bnd-1):-bnd,(bnd-1):-bnd]))
        

def computeFLIP(reference, test, PixelsPerDegree = 67):
    assert reference.shape == test.shape , 'ref and test should have same size'
    qc = 0.7
    qf = 0.5
    
    # Transform reference and test to opponent color space
    reference = colorSpaceTransform(reference, 'srgb2ycxcz', 0, 0)
    test = colorSpaceTransform(test, 'srgb2ycxcz', 0, 0)
    
    
    
    # --- Color pipeline ---
    # Spatial filtering
    s_a = generateSpatialFilter(PixelsPerDegree, 'A')
    s_rg = generateSpatialFilter(PixelsPerDegree, 'RG')
    s_by = generateSpatialFilter(PixelsPerDegree, 'BY')
    
  
    filteredReference = spatialFilter(reference, s_a, s_rg, s_by)
    filteredTest = spatialFilter(test, s_a, s_rg, s_by)

    

    # Perceptually Uniform Color Space
    preprocessedReference = huntAdjustment(colorSpaceTransform(filteredReference, 'linrgb2lab', 0, 0))
    preprocessedTest = huntAdjustment(colorSpaceTransform(filteredTest, 'linrgb2lab', 0, 0))
   
    
   
    
    # Color metric
    deltaEhyab = HyAB(preprocessedReference, preprocessedTest)
    huntAdjustedGreen = huntAdjustment(colorSpaceTransform(np.array([0,1,0]).reshape((1,1,3)), 'linrgb2lab', 0, 0))
    huntAdjustedBlue = huntAdjustment(colorSpaceTransform(np.array([0,0,1]).reshape((1,1,3)), 'linrgb2lab', 0, 0))
    cmax = np.power((HyAB(huntAdjustedGreen, huntAdjustedBlue)), qc)
    deltaEc = redistributeErrors(np.power(deltaEhyab, qc), cmax)
   
    

    # --- Feature pipeline ---
    # Extract and normalize achromatic component
    referenceY = (reference[:, :, 0] + 16) / 116
    testY = (test[:, :, 0] + 16) / 116


  
    # Edge and point detection
    edgesReference = featureDetection(referenceY, PixelsPerDegree, 'edge')
    pointsReference = featureDetection(referenceY, PixelsPerDegree, 'point')
    edgesTest = featureDetection(testY, PixelsPerDegree, 'edge')
    pointsTest = featureDetection(testY, PixelsPerDegree, 'point')

  
    
    # Feature metric
    deltaEf = np.power((1 / np.sqrt(2) *              
                  np.maximum(np.abs(np.linalg.norm(edgesReference, axis = 2) - np.linalg.norm(edgesTest, axis = 2)), 
                  np.abs(np.linalg.norm(pointsReference, axis = 2) - np.linalg.norm(pointsTest, axis = 2)))),qf)
    
    
    
    # --- Final error ---
    deltaE = np.power(deltaEc, (1 - deltaEf))
    error = np.mean(np.abs(deltaE))

    return error, deltaE
    
    


if __name__ == '__main__':
    im1 = plt.imread('abbey_correct.jpg').astype(float)/255
    im2 = plt.imread('abbey_badcolor.jpg').astype(float)/255
     
    
    error, deltaE = computeFLIP(im1,im2)
    plt.imshow(deltaE)
    print(error) 
    
    