#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 14:06:01 2024

@author: magnuso
"""

import numpy as np
import scipy
from scipy.sparse.csgraph import maximum_flow
from scipy import sparse
import matplotlib.pyplot as plt

# ---- Grid edges ----
def edges4connected(height, width, only_one_dir=0):
    """4-neighborhood on an HxW grid."""
    N = height * width
    I = np.array([], dtype=np.int64)
    J = np.array([], dtype=np.int64)

    # horizontal (column-major: neighbors differ by +1)
    iis = np.delete(np.arange(N), np.arange(height - 1, N, height))
    jjs = iis + 1
    if not only_one_dir:
        I = np.hstack((I, iis, jjs))
        J = np.hstack((J, jjs, iis))
    else:
        I = np.hstack((I, iis)); J = np.hstack((J, jjs))

    # vertical (neighbors differ by +height)
    iis = np.arange(0, N - height)
    jjs = iis + height
    if not only_one_dir:
        I = np.hstack((I, iis, jjs))
        J = np.hstack((J, jjs, iis))
    else:
        I = np.hstack((I, iis)); J = np.hstack((J, jjs))

    return I, J


def edges8connected(height, width, only_one_dir=0):
    """8-neighborhood on an HxW grid."""
    N = height * width
    I = np.array([], dtype=np.int64)
    J = np.array([], dtype=np.int64)

    # horizontal
    iis = np.delete(np.arange(N), np.arange(height - 1, N, height))
    jjs = iis + 1
    if not only_one_dir:
        I = np.hstack((I, iis, jjs)); J = np.hstack((J, jjs, iis))
    else:
        I = np.hstack((I, iis)); J = np.hstack((J, jjs))

    # diagonals
    jjs = iis + 1 + height
    if not only_one_dir:
        I = np.hstack((I, iis, jjs)); J = np.hstack((J, jjs, iis))
    else:
        I = np.hstack((I, iis)); J = np.hstack((J, jjs))

    jjs = iis + 1 - height
    if not only_one_dir:
        I = np.hstack((I, iis, jjs)); J = np.hstack((J, jjs, iis))
    else:
        I = np.hstack((I, iis)); J = np.hstack((J, jjs))

    mask = (I >= 0) & (I < N) & (J >= 0) & (J < N)
    I, J = I[mask], J[mask]

    # vertical
    iis = np.arange(0, N - height)
    jjs = iis + height
    if not only_one_dir:
        I = np.hstack((I, iis, jjs)); J = np.hstack((J, jjs, iis))
    else:
        I = np.hstack((I, iis)); J = np.hstack((J, jjs))

    return I, J


if __name__ == '__main__':
    # sanity check on a tiny graph
    I = np.array([0, 0, 1, 2], dtype=np.int32)
    J = np.array([1, 2, 3, 3], dtype=np.int32)
    V = np.array([5, 2, 3, 7], dtype=np.int32)
    F = sparse.coo_array((V, (I, J)), shape=(4, 4)).tocsr()
    mf = maximum_flow(F, 0, 3)
    print("[sanity] max flow value =", mf.flow_value)

    # load data
    data = scipy.io.loadmat('heart_data.mat')
    data_chamber = data['chamber_values'].astype(np.float64).ravel()
    data_background = data['background_values'].astype(np.float64).ravel()
    im = data['im'].astype(np.float64)
    M, N = im.shape
    n = M * N

    # Gaussian stats
    m_chamber = float(np.mean(data_chamber))
    s_chamber = float(np.std(data_chamber) + 1e-8)
    m_background = float(np.mean(data_background))
    s_background = float(np.std(data_background) + 1e-8)
    print(f"[stats] chamber:   mu={m_chamber:.4f}, sigma={s_chamber:.4f}")
    print(f"[stats] background: mu={m_background:.4f}, sigma={s_background:.4f}")

    # n-links (contrast-sensitive Potts)
    Ie, Je = edges4connected(M, N)  # or use edges8connected(M, N)
    im_flat = im.flatten()
    diffs = im_flat[Ie.astype(int)] - im_flat[Je.astype(int)]
    sigma_g = float(np.std(im_flat) + 1e-8)
    lam = 0.5
    Ve = lam * np.exp(- (diffs ** 2) / (2.0 * sigma_g ** 2))
    Ve = np.maximum(Ve, 1e-12)

    # t-links (negative log-likelihoods)
    x = im_flat
    nll_chamber = 0.5 * ((x - m_chamber) / s_chamber) ** 2 + np.log(s_chamber)
    nll_back = 0.5 * ((x - m_background) / s_background) ** 2 + np.log(s_background)
    Vs = np.maximum(nll_chamber, 0) + 1e-12   # source
    Vt = np.maximum(nll_back, 0) + 1e-12      # sink

    # assemble graph
    Is1 = np.arange(n, dtype=np.int64); Js1 = np.full(n, n, dtype=np.int64)     # pixel→S
    Is2 = np.full(n, n, dtype=np.int64);  Js2 = np.arange(n, dtype=np.int64)    # S→pixel
    It1 = np.arange(n, dtype=np.int64);   Jt1 = np.full(n, n+1, dtype=np.int64) # pixel→T
    It2 = np.full(n, n+1, dtype=np.int64); Jt2 = np.arange(n, dtype=np.int64)   # T→pixel

    I = np.hstack((Ie, Is1, Is2, It1, It2)).astype(np.int32)
    J = np.hstack((Je, Js1, Js2, Jt1, Jt2)).astype(np.int32)
    V = np.hstack((Ve, Vs, Vs, Vt, Vt))

    # integer capacities
    sf = 10000.0
    V = np.round(V * sf).astype(np.int32)

    F = sparse.coo_array((V, (I, J)), shape=(n + 2, n + 2)).tocsr()

    # min-cut
    mf = maximum_flow(F, n, n + 1)

    # labels (compare flow-to-sink with sink capacity)
    seg_flow = mf.flow
    imflow = seg_flow[0:n, n + 1].reshape((M, N)).toarray().astype(float)
    Vt_cap = V[-n:].astype(float).reshape(M, N)
    imseg = imflow < Vt_cap

    # visualize + save
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(im, cmap='gray'); plt.title('Input image'); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(imseg, cmap='gray'); plt.title('Graph-Cut segmentation'); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(im, cmap='gray'); plt.contour(imseg, colors='r', linewidths=1); plt.title('Overlay'); plt.axis('off')
    plt.tight_layout()
    plt.savefig('task2_result.png', dpi=200, bbox_inches='tight')
    plt.show()

    
    
    
    