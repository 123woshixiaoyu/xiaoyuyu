#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:51:35 2024

@author: magnuso
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# --------------------------
# 读取 curvedata.mat（适配 xm/ym、x/y、X/Y、Nx2、2xN）
# --------------------------
def load_curvedata():
    # 优先当前目录，也兼容上一层/常见布局
    candidates = [
        "./curvedata.mat",
        "../curvedata.mat",
        "../../curvedata.mat",
    ]
    path = None
    for p in candidates:
        if os.path.exists(p):
            path = p
            break
    if path is None:
        raise FileNotFoundError("curvedata.mat not found near task4.py")

    mat = sio.loadmat(path)
    # 常见命名
    if "xm" in mat and "ym" in mat:
        x = np.ravel(mat["xm"]).astype(float)
        y = np.ravel(mat["ym"]).astype(float)
        return x, y
    if "x" in mat and "y" in mat:
        return np.ravel(mat["x"]).astype(float), np.ravel(mat["y"]).astype(float)
    if "X" in mat and "Y" in mat:
        return np.ravel(mat["X"]).astype(float), np.ravel(mat["Y"]).astype(float)

    # 回退：找 Nx2 或 2xN
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        arr = np.asarray(v, dtype=float)
        if arr.ndim == 2:
            if arr.shape[1] == 2:   # Nx2
                return arr[:, 0], arr[:, 1]
            if arr.shape[0] == 2:   # 2xN
                return arr[0, :], arr[1, :]
    raise ValueError(f"Unsupported curvedata format. Keys: {list(mat.keys())}")

# --------------------------
# 二次曲线 LS: y = a x^2 + b x + c
# --------------------------
def design_matrix(x):
    x = np.asarray(x).reshape(-1)
    return np.column_stack([x**2, x, np.ones_like(x)])

def ls_fit_quadratic(x, y):
    A = design_matrix(x)
    theta, *_ = np.linalg.lstsq(A, y, rcond=None)  # [a,b,c]
    return theta

def predict_quadratic(x, theta):
    a, b, c = theta
    x = np.asarray(x).reshape(-1)
    return a * x**2 + b * x + c

def mse(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean((y_true - y_pred)**2))

# --------------------------
# RANSAC（二次曲线）
# --------------------------
def ransac_quadratic(
    x, y,
    n_iter=500,
    threshold=0.5,        # 依据噪声量级微调：大→更宽松，小→更严格
    min_inlier_ratio=0.3, # 内点比例下限，避免退化
    random_state=0
):
    rng = np.random.RandomState(random_state)
    n = len(x)
    best_inliers = None
    best_count = -1
    best_theta = None

    for _ in range(n_iter):
        # 二次曲线 3 个参数 → 最小采样 3 点
        ids = rng.choice(n, size=3, replace=False)
        try:
            theta = ls_fit_quadratic(x[ids], y[ids])
        except np.linalg.LinAlgError:
            continue

        y_hat = predict_quadratic(x, theta)
        residuals = np.abs(y - y_hat)
        inliers = residuals < threshold
        cnt = int(np.sum(inliers))

        if cnt > best_count:
            best_count = cnt
            best_inliers = inliers
            best_theta = theta

    # 若内点过少，退回全体 LS
    if best_inliers is None or best_count < int(min_inlier_ratio * n):
        theta_ls = ls_fit_quadratic(x, y)
        return theta_ls, np.ones_like(x, dtype=bool), False

    # 对内点做一次 LS 精化
    theta_refined = ls_fit_quadratic(x[best_inliers], y[best_inliers])
    return theta_refined, best_inliers, True

# --------------------------
# 画图
# --------------------------
def plot_results(x, y, theta_ls, theta_ransac, inliers, out_png="task4_curvefit.png"):
    xmin, xmax = float(np.min(x)), float(np.max(x))
    xx = np.linspace(xmin, xmax, 400)
    y_ls = predict_quadratic(xx, theta_ls)
    y_ransac = predict_quadratic(xx, theta_ransac)

    plt.figure(figsize=(7,5))
    # 外点红色，内点蓝色
    plt.scatter(x[~inliers], y[~inliers], s=20, c="tab:red", label="Outliers")
    plt.scatter(x[inliers],  y[inliers],  s=20, c="tab:blue", label="Inliers")
    # 拟合曲线
    plt.plot(xx, y_ls, lw=2, label="LS fit")
    plt.plot(xx, y_ransac, lw=2, linestyle="--", label="RANSAC (LS on inliers)")
    plt.xlabel("x"); plt.ylabel("y")
    plt.title("Quadratic fit: LS vs RANSAC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Saved figure: {out_png}")

# --------------------------
# 主过程
# --------------------------
if __name__ == "__main__":
    x, y = load_curvedata()

    # 1) LS
    theta_ls = ls_fit_quadratic(x, y)
    yhat_ls = predict_quadratic(x, theta_ls)
    mse_ls_all = mse(y, yhat_ls)

    # 2) RANSAC → 内点 → 内点上 LS
    theta_r, inliers, ok = ransac_quadratic(
        x, y,
        n_iter=500,
        threshold=0.5,        # 可按数据噪声调节
        min_inlier_ratio=0.3,
        random_state=0
    )
    yhat_r_all = predict_quadratic(x, theta_r)
    mse_r_all  = mse(y, yhat_r_all)                    # RANSAC 模型在所有点上的 MSE
    mse_r_in   = mse(y[inliers], yhat_r_all[inliers])  # 仅内点 MSE
    n_in       = int(np.sum(inliers))

    a_ls, b_ls, c_ls = theta_ls
    a_r,  b_r,  c_r  = theta_r
    print("LS fit     : a={:.6f}, b={:.6f}, c={:.6f}".format(a_ls, b_ls, c_ls))
    print("  MSE (all points)           : {:.6f}".format(mse_ls_all))
    print("RANSAC fit : a={:.6f}, b={:.6f}, c={:.6f}".format(a_r, b_r, c_r))
    print("  Inliers / Total            : {} / {}".format(n_in, len(x)))
    print("  MSE (all points, RANSAC)   : {:.6f}".format(mse_r_all))
    print("  MSE (inliers only, RANSAC) : {:.6f}".format(mse_r_in))
    print("  RANSAC success?            :", ok)

    plot_results(x, y, theta_ls, theta_r, inliers)
