# assignment3.py  —  OCR pipeline main (clean version)
# - uses your im2segment() and segment2features()
# - trains a classifier with StandardScaler + PCA + LogisticRegression
# - runs the provided benchmark_assignment3 (segmentation -> features -> classification)
# assignment3.py — OCR pipeline (fixed to unify feature dim = 14)
# - StandardScaler + PCA + LogisticRegression
# - Trains on ocrsegment*.npy after robust reshaping
# - Ensures train features have the same dim (14) as your segment2features()

# assignment3.py
import os
import pickle
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from my_benchmarking.benchmark_assignment3 import benchmark_assignment3

from main_smp import im2segment
from segment2features import segment2features  # output (28,1)

# ----------------------------
# Feature wrapper
# ----------------------------
def segment2feature(Si):
    f = segment2features(Si)          # (28,1)
    return np.asarray(f, dtype=np.float64).reshape(-1)  # (28,)

# ----------------------------
# Classifier training
# ----------------------------
def class_train(X_feats, y, use_pca=False, pca_dim=0, random_state=42):
    X_feats = np.asarray(X_feats, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    assert X_feats.ndim == 2, f"X_feats must be 2D, got {X_feats.shape}"
    assert X_feats.shape[0] == y.shape[0], f"N mismatch: X={X_feats.shape}, y={y.shape}"

    scaler = StandardScaler().fit(X_feats)
    Xz = scaler.transform(X_feats)

    pca = None
    if use_pca:
        d = Xz.shape[1]
        k = max(1, min(int(pca_dim), d)) if pca_dim else min(20, d)
        pca = PCA(n_components=k, random_state=random_state)
        Xz = pca.fit_transform(Xz)

    clf = LogisticRegression(
        solver="lbfgs",
        multi_class="auto",
        max_iter=1000,
        random_state=random_state
    )
    clf.fit(Xz, y)

    return {
        "scaler": scaler,
        "pca": pca,
        "clf": clf,
        "feat_dim": X_feats.shape[1],
    }

# ----------------------------
# Single sample prediction
# ----------------------------
def features2class(x, classification_data):
    x = np.asarray(x, dtype=np.float64).reshape(1, -1)
    expect_d = classification_data.get("feat_dim", x.shape[1])
    if x.shape[1] != expect_d:
        raise ValueError(f"Feature dim mismatch: got {x.shape[1]}, expect {expect_d}")

    z = classification_data["scaler"].transform(x)
    if classification_data["pca"] is not None:
        z = classification_data["pca"].transform(z)
    yhat = classification_data["clf"].predict(z)[0]
    return int(yhat)

def feature2class(x, classdata):
    return features2class(x, classdata)

# ----------------------------
# Training from provided npy
# ----------------------------
def train_from_npy_masks(npy_masks, npy_gt, out_pkl,
                         use_pca=False, pca_dim=0, random_state=42):
    X_raw = np.load(npy_masks, allow_pickle=True)
    y = np.load(npy_gt, allow_pickle=True)

    if y.ndim > 1:
        y = y.reshape(-1)
    if X_raw.ndim != 3:
        raise ValueError(f"Expect masks 3D (N,H,W), got {X_raw.shape}")

    feats = [segment2feature(X_raw[i]) for i in range(X_raw.shape[0])]
    X = np.vstack(feats)

    model = class_train(X, y, use_pca=use_pca, pca_dim=pca_dim, random_state=random_state)
    with open(out_pkl, "wb") as f:
        pickle.dump(model, f)
    print(f"[train] Saved classification_data.pkl  X={X.shape} y={y.shape}")
    return model

def load_or_train_classification_data(thisdir, force_train=False,
                                      use_pca=False, pca_dim=0, random_state=42):
    pkl_path = os.path.join(thisdir, "classification_data.pkl")
    if (not force_train) and os.path.isfile(pkl_path):
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    npy_masks = os.path.join(thisdir, "ocrsegmentdata.npy")
    npy_gt    = os.path.join(thisdir, "ocrsegmentgt.npy")
    if not (os.path.isfile(npy_masks) and os.path.isfile(npy_gt)):
        raise FileNotFoundError("Missing ocrsegmentdata.npy / ocrsegmentgt.npy")

    return train_from_npy_masks(
        npy_masks, npy_gt, pkl_path,
        use_pca=use_pca, pca_dim=pca_dim, random_state=random_state
    )

# ----------------------------
# main
# ----------------------------
if __name__ == "__main__":
    thisdir = os.path.dirname(os.path.realpath(__file__))

    #datadir = os.path.join(thisdir, 'datasets', 'short1')
    datadir = os.path.join(thisdir, 'datasets', 'home1')

    mode = 0

    classification_data = load_or_train_classification_data(
        thisdir,
        force_train=False,
        use_pca=False,
        pca_dim=0,
        random_state=42
    )

    hitrate, confmat, allres, alljs, alljfg, allX, allY = benchmark_assignment3(
        im2segment,
        segment2feature,
        feature2class,
        classification_data,
        datadir,
        mode
    )

    print('Hitrate = ' + str(hitrate * 100) + '%')

