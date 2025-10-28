#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 11:37:56 2024

@author: magnuso
"""
# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
from my_benchmarking.benchmark_assignment3 import benchmark_assignment3
from main_smp import im2segment
from segment2features import segment2features
from assignment3 import features2class


def segment2feature(Si):
    """Convert a segment mask to a 1-D feature vector."""
    f = segment2features(Si)
    return np.asarray(f, dtype=np.float64).reshape(-1)


def feature2class(x, classification_data):
    """Predict class label for feature vector x."""
    return features2class(x, classification_data)


def _load_by_ext(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pkl":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"[clf] loaded: {path}")
        return obj
    if ext == ".npz":
        z = np.load(path, allow_pickle=True)
        if "arr_0" in z and isinstance(z["arr_0"].item(), dict):
            print(f"[clf] loaded: {path}")
            return z["arr_0"].item()
        print(f"[clf] loaded: {path}")
        return {k: z[k] for k in z.files}
    if ext == ".npy":
        obj = np.load(path, allow_pickle=True)
        obj = obj.item() if hasattr(obj, "item") else obj
        print(f"[clf] loaded: {path}")
        return obj
    if ext == ".mat":
        import scipy.io as sio
        obj = sio.loadmat(path)
        print(f"[clf] loaded: {path}")
        return obj
    raise ValueError(f"Unsupported file type: {path}")


def load_classification_data(search_dir):
    """Load pre-trained classification data."""
    envp = os.environ.get("CLASSIFIER_PATH", "").strip()
    if envp and os.path.isfile(envp):
        return _load_by_ext(envp)

    candidates = [
        "classification_data.pkl",
        "classification_data_bootstrap.pkl",
        "classification_data.npz",
        "classification_data.npy",
        "classification_data.mat",
    ]
    for name in candidates:
        p = os.path.join(search_dir, name)
        if os.path.isfile(p):
            return _load_by_ext(p)

    try:
        import assignment3 as a3
        if hasattr(a3, "load_or_train_classification_data"):
            return a3.load_or_train_classification_data(
                search_dir, force_train=False, use_pca=False, pca_dim=0, random_state=42
            )
        if hasattr(a3, "load_pickled_classifier"):
            p = os.path.join(search_dir, "classification_data.pkl")
            if os.path.isfile(p):
                return a3.load_pickled_classifier(p)
    except Exception:
        pass

    raise FileNotFoundError(
        "No pre-trained classification data found. "
        "Place the file in this directory or set CLASSIFIER_PATH."
    )


if __name__ == "__main__":
    thisdir = os.path.dirname(os.path.realpath(__file__))
    classification_data = load_classification_data(thisdir)

    datasets = ["short1", "short2", "home1", "home2", "home3"]
    mode = 0
    results = []

    for ds in datasets:
        datadir = os.path.join(thisdir, "datasets", ds)
        hitrate, confmat, *_ = benchmark_assignment3(
            im2segment, segment2feature, feature2class, classification_data, datadir, mode
        )
        results.append((ds, float(hitrate)))
        print(f"{ds}, Hitrate = {hitrate*100:.2f}%")

    print("\n===== Summary =====")
    for ds, hr in results:
        print(f"{ds:>6s} : {hr*100:6.2f}%")
