#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb Sept 30 17:24 2025

@author: Zeyu Liang
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 1 — Linear SVM (Pegasos) from scratch (numpy only)
- Loads FaceNonFace.mat (X: 361x200, Y: 1x200 with {+1, -1})
- 100 stratified 80/20 splits
- Reports mean ± std of train/test accuracy

Usage:
  python task1_svm.py --data FaceNonFace.mat --trials 100 --lam 1e-4 --epochs 20 --batch 32
"""

import argparse
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split  # only for data splitting
import matplotlib.pyplot as plt

# ---------- Save 19x19 grayscale images ----------
def _save_img(vec, title, path):
    img = np.asarray(vec, dtype=float).reshape(19, 19)  # 361=19*19
    plt.figure(figsize=(3, 3))
    plt.imshow(img, cmap='gray', interpolation='nearest')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

# ---------- Standardization helpers ----------
def _fit_zscore(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    return mu, sigma

def _apply_zscore(X, mu, sigma):
    return (X - mu) / sigma

def _fit_pca(Xs, p=60):
    # Xs: standardized training data (n,d)
    # re-center for more stable PCA
    c = Xs.mean(axis=0)
    Xc = Xs - c
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:p].T    # (d,p)
    return c, W     # transform: phi(x)= (xs - c) @ W

def _apply_pca(Xs, c, W):
    return (Xs - c) @ W

def pegasos_train(X, y, lam=1e-5, epochs=80, batch_size=32, seed=0,
                  pca_components=60, t0=200, use_avg=True):
    """
    Linear SVM (Pegasos) + z-score + PCA + learning rate warm-start + weight averaging
    X: (n,d), y: {+1,-1}
    """
    rng = np.random.RandomState(seed)
    X = np.asarray(X, dtype=np.float64)
    y = np.where(np.asarray(y).reshape(-1) > 0, 1, -1).astype(np.int32)
    n, d = X.shape

    # z-score standardization
    mu, sigma = _fit_zscore(X)
    Xs = _apply_zscore(X, mu, sigma)

    # PCA dimensionality reduction
    pc = max(1, min(pca_components, d))
    c_pca, W = _fit_pca(Xs, p=pc)
    Z = _apply_pca(Xs, c_pca, W)              # (n, p)
    p = Z.shape[1]

    # add bias term
    Zt = np.hstack([Z, np.ones((n,1))])       # (n, p+1)
    w = np.zeros(p+1, dtype=np.float64)

    t = 0
    bound = 1.0/np.sqrt(lam)
    w_sum = np.zeros_like(w); steps = 0

    for _ in range(epochs):
        idx = rng.permutation(n)
        for i0 in range(0, n, batch_size):
            ib = idx[i0:i0+batch_size]
            if ib.size == 0: continue
            Zb, yb = Zt[ib], y[ib]
            m = ib.size

            t += 1
            eta = 1.0 / (lam * (t0 + t))    # warm-start schedule

            margin = yb * (Zb @ w)
            violated = margin < 1.0
            w = (1.0 - eta*lam)*w
            if np.any(violated):
                w += (eta/m) * (Zb[violated].T @ yb[violated])

            # project only the weight part (exclude bias)
            wn = np.linalg.norm(w[:-1])
            if wn > bound:
                w[:-1] *= (bound/wn)

            if use_avg:
                w_sum += w; steps += 1

    if use_avg and steps > 0:
        w = w_sum / steps

    return {
        "w": w[:-1], "b": float(w[-1]),
        "mu": mu, "sigma": sigma,
        "c_pca": c_pca, "W": W,             # PCA transform
        "lam": lam, "epochs": epochs, "batch_size": batch_size,
        "pca_components": pc, "t0": t0, "use_avg": use_avg
    }

def pegasos_predict(X, model):
    X = np.asarray(X, dtype=np.float64)
    Zs = _apply_zscore(X, model["mu"], model["sigma"])
    Zp = _apply_pca(Zs, model["c_pca"], model["W"])
    scores = Zp @ model["w"] + model["b"]
    return np.where(scores >= 0.0, 1, -1).astype(np.int32)

# ---------- One evaluation split ----------
def eval_once(X, Y, lam=1e-4, epochs=20, batch=32, seed_split=0, seed_train=0):
    # sklearn split expects (n, d) and (n,)
    Xn = X.T if X.shape[0] == 361 else X      # ensure (n, d)
    Yn = Y.ravel().astype(int)
    Yn = np.where(Yn > 0, 1, -1).astype(np.int32)

    Xtr, Xte, ytr, yte = train_test_split(
        Xn, Yn, test_size=0.2, stratify=Yn, random_state=seed_split
    )
    model = pegasos_train(Xtr, ytr, lam=lam, epochs=epochs, batch_size=batch, seed=seed_train)
    ytr_pred = pegasos_predict(Xtr, model)
    yte_pred = pegasos_predict(Xte, model)
    acc_tr = float(np.mean(ytr_pred == ytr))
    acc_te = float(np.mean(yte_pred == yte))
    # also return info for saving images
    return acc_tr, acc_te, Xte, yte, yte_pred

# ---------- Main loop ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="FaceNonFace.mat")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--lam", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mat = loadmat(args.data)
    X = mat["X"]                # expected (361, 200)
    Y = mat["Y"]                # expected (1, 200) or (200, 1)

    rng = np.random.RandomState(args.seed)
    acc_tr_list, acc_te_list = [], []
    saved_examples = False

    for _ in range(args.trials):
        seed_split = int(rng.randint(0, 10_000_000))
        seed_train = int(rng.randint(0, 10_000_000))
        acc_tr, acc_te, Xte, yte, yte_pred = eval_once(
            X, Y, lam=args.lam, epochs=args.epochs, batch=args.batch,
            seed_split=seed_split, seed_train=seed_train
        )
        acc_tr_list.append(acc_tr)
        acc_te_list.append(acc_te)

        # save two test images (one face, one non-face) only once (at the first trial)
        if not saved_examples:
            face_idx    = np.where(yte ==  1)[0]
            nonface_idx = np.where(yte == -1)[0]
            if len(face_idx) > 0:
                i = face_idx[0]
                ok = "correct" if yte_pred[i] == yte[i] else "wrong"
                _save_img(Xte[i], f"SVM Face | Pred={int(yte_pred[i])} ({ok})",
                          "example_face_svm.png")
            if len(nonface_idx) > 0:
                j = nonface_idx[0]
                ok = "correct" if yte_pred[j] == yte[j] else "wrong"
                _save_img(Xte[j], f"SVM Non-face | Pred={int(yte_pred[j])} ({ok})",
                          "example_nonface_svm.png")
            saved_examples = True

    acc_tr = np.array(acc_tr_list)
    acc_te = np.array(acc_te_list)
    print(f"Train acc: {acc_tr.mean():.3f} ± {acc_tr.std():.3f}")
    print(f"Test  acc: {acc_te.mean():.3f} ± {acc_te.std():.3f}")

if __name__ == "__main__":
    main()
