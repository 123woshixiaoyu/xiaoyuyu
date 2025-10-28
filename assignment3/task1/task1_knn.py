#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb Sept 30 17:24 2025

@author: Zeyu Liang
"""

import scipy
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def class_train(X, Y, k=5):
    """
    Weighted kNN training: only store the training set (with standardization parameters).
    X: (n, d)  each row is a sample
    Y: (n, 1) or (n,)  labels in {+1, -1}
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(Y).reshape(-1)
    y = np.where(y > 0, 1, -1).astype(np.int32)

    # Compute standardization parameters using only the training set
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    Xs = (X - mu) / sigma

    return {
        "X": Xs,          # standardized training set
        "y": y,           # training labels
        "mu": mu,         # training mean
        "sigma": sigma,   # training std
        "k": int(k)
    }

def classify(x, classification_data):
    """
    Weighted kNN prediction: inverse distance as weight (1 / (dist + eps)).
    x: (d,) single sample
    """
    Xtr = classification_data["X"]      # (n, d)
    ytr = classification_data["y"]      # (n,)
    mu  = classification_data["mu"]
    sg  = classification_data["sigma"]
    k   = classification_data["k"]

    z = (np.asarray(x, dtype=np.float64).reshape(-1) - mu) / sg  # standardize
    # Euclidean distance
    d2 = np.sum((Xtr - z)**2, axis=1)
    # pick k nearest neighbors
    idx = np.argpartition(d2, kth=min(k, len(d2)-1))[:k]
    dsel = np.sqrt(d2[idx])
    eps = 1e-8
    w = 1.0 / (dsel + eps)              # closer neighbors have larger weights
    # weighted vote
    score = np.sum(w * ytr[idx])
    return 1 if score >= 0 else -1

def _save_img(vec, title, path):
    img = np.asarray(vec, dtype=float).reshape(19, 19)  # 361=19*19
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

if __name__ == "__main__":
    # load data
    datadir = './'
    data = scipy.io.loadmat(datadir + 'FaceNonFace.mat')
    X = data['X'].transpose()   # (200,361)
    Y = data['Y'].transpose()   # (200,1)
    nbr_examples = np.size(Y, 0)

    nbr_trials = 100
    acc_tests  = np.zeros((nbr_trials, 1))
    acc_trains = np.zeros((nbr_trials, 1))

    saved_examples = False  # only save sample images once (at the first trial)

    for i in range(nbr_trials):
        # 80/20 stratified split
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, stratify=Y.ravel(), random_state=None
        )
        nbr_train_examples = np.size(Y_train, 0)
        nbr_test_examples  = np.size(Y_test, 0)

        # training
        classification_data = class_train(X_train, Y_train, k=5)

        # prediction (test set)
        predictions_test = np.zeros((nbr_test_examples, 1), dtype=int)
        for j in range(nbr_test_examples):
            predictions_test[j] = classify(X_test[j, :], classification_data)

        # prediction (train set)
        predictions_train = np.zeros((nbr_train_examples, 1), dtype=int)
        for j in range(nbr_train_examples):
            predictions_train[j] = classify(X_train[j, :], classification_data)

        # accuracy
        acc_test  = np.mean(predictions_test.ravel()  == Y_test.ravel())
        acc_train = np.mean(predictions_train.ravel() == Y_train.ravel())
        acc_tests[i]  = acc_test
        acc_trains[i] = acc_train

        # save two sample test images: one face and one non-face (with prediction and correctness)
        if not saved_examples:
            yte = Y_test.ravel().astype(int)
            ypr = predictions_test.ravel().astype(int)

            face_idx    = np.where(yte ==  1)[0]
            nonface_idx = np.where(yte == -1)[0]

            if len(face_idx) > 0:
                i_face = face_idx[0]
                correct = "correct" if ypr[i_face] == yte[i_face] else "wrong"
                title = f"Face | Pred={int(ypr[i_face])} ({correct})"
                _save_img(X_test[i_face], title, "example_face_knn.png")

            if len(nonface_idx) > 0:
                i_nonface = nonface_idx[0]
                correct = "correct" if ypr[i_nonface] == yte[i_nonface] else "wrong"
                title = f"Non-face | Pred={int(ypr[i_nonface])} ({correct})"
                _save_img(X_test[i_nonface], title, "example_nonface_knn.png")

            saved_examples = True

    # print accuracy (mean ± std), same style as SVM
    print(f"Train acc: {acc_trains.mean():.3f} ± {acc_trains.std():.3f}")
    print(f"Test  acc: {acc_tests.mean():.3f} ± {acc_tests.std():.3f}")
