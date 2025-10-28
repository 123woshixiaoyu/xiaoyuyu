#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
assignment2.py（报告版）
生成报告所需的全部产物：
1) 分割示例图：report_outputs/sample_segments.png
2) 特征向量表示例（im1的前若干字符）：CSV/Markdown，控制台也打印
3) 基准二维投影图（SVD）：report_outputs/benchmark_projection.png

依赖：
    pip install numpy matplotlib pandas
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from main_smp import im2segment
from segment2features import segment2features
from my_benchmarking.benchmark_assignment2 import benchmark_assignment2


# ---------- 配置 ----------
DATASET_NAME = 'short1'           # 数据集文件夹名
SAMPLE_IMAGE = 'im2'              # 用于举例展示与特征表的样例前缀
MAX_EXAMPLES = 6                  # 表格与掩膜展示的样例数量
OUTDIR = 'report_outputs'         # 导出目录
# -------------------------


def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)


def make_sample_segments_figure(img_path, masks, out_path, max_show=6):
    """原图 + 前 max_show 个掩膜的拼图"""
    img = plt.imread(img_path)
    k = min(len(masks), max_show)
    cols = 3
    rows = 1 + int(np.ceil(k / cols))

    plt.figure(figsize=(cols * 3, rows * 3))
    # 原图
    ax = plt.subplot(rows, cols, 1)
    ax.imshow(img, cmap='gray')
    ax.set_title("Original")
    ax.axis('off')

    # 掩膜
    for i in range(k):
        ax = plt.subplot(rows, cols, i + 2)
        ax.imshow(masks[i], cmap='gray')
        ax.set_title(f"Mask {i}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[saved] sample segments figure -> {out_path}")


def features_table_for_dataset(datadir, out_csv, out_md):
    """
    遍历数据集文件夹 datadir 下的所有 imX.npy + imX.txt，
    提取所有实例的特征向量，保存成 CSV/Markdown。
    """
    feat_names = [
        "f1_area_ratio",         # 面积比
        "f2_rect_fill_ratio",    # 填充度/矩形度
        "f3_bbox_aspect",        # 外接框宽高比
        "f4_holes",              # 孔洞数
        "f5_euler",              # Euler 数
        "f6_compactness_p2_over_a",  # 紧致度
        "f7_sym_left_right",     # 左右对称性
        "f8_sym_up_down",        # 上下对称性
        "f9_thickness_norm",     # 归一化笔画粗细
        "f10_proj_row_over_col", # 方向性峰值比
        "f11_hu1_log",           # Hu #1
        "f12_hu2_log",           # Hu #2
        "f13_hu3_log",           # Hu #3
        "f14_hu4_log"            # Hu #4
    ]

    rows = []
    for fname in os.listdir(datadir):
        if not fname.endswith(".npy"):
            continue
        stem = fname[:-4]  # 去掉 .npy
        npy_path = os.path.join(datadir, stem + ".npy")
        txt_path = os.path.join(datadir, stem + ".txt")
        if not os.path.exists(txt_path):
            continue

        # 加载分割掩膜和标签
        Sgt = np.load(npy_path, allow_pickle=True)
        with open(txt_path, 'r') as f:
            gt = f.read().strip()

        n = min(len(Sgt), len(gt))
        for i in range(n):
            fvec = segment2features(Sgt[i]).reshape(-1)
            row = {"image": stem, "idx": i, "digit": gt[i]}
            for name, val in zip(feat_names, fvec):
                row[name] = float(val)
            rows.append(row)

    df = pd.DataFrame(rows, columns=["image", "idx", "digit"] + feat_names)
    df.to_csv(out_csv, index=False)
    with open(out_md, "w", encoding="utf-8") as fmd:
        fmd.write(df.to_markdown(index=False))

    print("\n[Full dataset feature table]")
    print(df.to_string(index=False, max_cols=None, max_rows=50))  # 控制台打印前50行
    print(f"[saved] features CSV  -> {out_csv}")
    print(f"[saved] features MD   -> {out_md}")



def svd_projection_figure(allX, allY, out_path):
    import numpy as np
    import matplotlib.pyplot as plt

    # 统一输入格式
    X = np.asarray(allX)
    labels = list(allY) if not isinstance(allY, list) else allY

    if X.size == 0 or X.shape[1] == 0 or len(labels) == 0:
        print("[report] 空特征或空标签，跳过投影图。")
        return

    # SVD 投影：allX (k,N) -> Z (2,N)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Z = (U[:, :2].T @ X)  # (2, N)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.grid(alpha=0.3)

    # 先画散点，保证即使不写 text 也能看到点
    ax.scatter(Z[0], Z[1], s=8)

    # 再标注类别字符
    for i, ch in enumerate(labels):
        ax.text(Z[0, i], Z[1, i], str(ch), fontsize=8)

    # 让坐标自动适应数据范围
    ax.relim()
    ax.autoscale()

    ax.set_title("2D projection of features (SVD)")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] SVD projection figure -> {out_path}")

def save_svd_projection(allX, allY, out_path):
    """
    allX: (k, N) 特征矩阵（列为样本），allY: 长度 N 的标签（list or str）
    步骤：z-score 标准化 -> SVD -> 散点+文字 -> autoscale -> 保存
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap

    if allX is None or allX.size == 0 or len(allY) == 0:
        print("[svd] empty X or labels, skip.")
        return

    # ---- z-score 标准化（按特征维度）----
    X = allX.astype(np.float64, copy=False)
    mu = X.mean(axis=1, keepdims=True)
    sigma = X.std(axis=1, keepdims=True) + 1e-8
    Xz = (X - mu) / sigma

    # ---- SVD：Xz = U S V^T -> 用 U 的前两列作为基 ----
    U, S, Vt = np.linalg.svd(Xz, full_matrices=False)   # U:(k,k), S:(k,), Vt:(N,N)
    Z = (U[:, :2].T @ Xz)                                # (2, N)

    # ---- 画图并保存（独立 fig，避免被其他图覆盖）----
    labels = list(allY) if not isinstance(allY, list) else allY
    digits = sorted(set(labels))
    cmap = get_cmap('tab10')
    color_map = {d: cmap(i % 10) for i, d in enumerate(digits)}

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.grid(alpha=0.3)

    # 先散点，后文字；启用 autoscale
    for i, ch in enumerate(labels):
        ax.scatter(Z[0, i], Z[1, i], s=18, color=color_map[ch], alpha=0.85)
        ax.text(Z[0, i], Z[1, i], str(ch), fontsize=8, color=color_map[ch])

    ax.relim(); ax.autoscale()
    ax.set_title("2D projection of features (SVD)")
    ax.set_xlabel("Component 1"); ax.set_ylabel("Component 2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[saved] SVD projection -> {out_path}")


if __name__ == "__main__":
    thisdir = os.path.dirname(os.path.realpath(__file__))
    datadir = os.path.join(thisdir, 'datasets', DATASET_NAME)
    ensure_outdir(os.path.join(thisdir, OUTDIR))

    # ---- 1) 分割示例：原图 + 掩膜图 ----
    img_path = os.path.join(datadir, f"{SAMPLE_IMAGE}.jpg")
    im = plt.imread(img_path)
    S_demo = im2segment(im)  # list of masks (H,W)
    fig_segments_path = os.path.join(thisdir, OUTDIR, "sample_segments.png")
    make_sample_segments_figure(img_path, S_demo, fig_segments_path, max_show=MAX_EXAMPLES)

    # ---- 2) 全数据集特征表 ----
    out_csv = os.path.join(thisdir, OUTDIR, "all_features.csv")
    out_md = os.path.join(thisdir, OUTDIR, "all_features.md")
    features_table_for_dataset(datadir, out_csv, out_md)

    # ---- 3) 整个数据集的特征 + 2D投影图（无监督可视化）----
    #    这里不做分类评估，符合作业要求
    # Benchmark your feature extractor routine on all images
    debug = True
    allX, allY = benchmark_assignment2(segment2features, datadir, debug)

    print("[check] allX shape:", None if allX is None else allX.shape, " #labels:", len(allY))
    proj_path = os.path.join(thisdir, OUTDIR, "benchmark_projection.png")
    save_svd_projection(allX, allY, proj_path)
    # === 加检查，确认结果是否为空 ===
    print("[check] allX shape:", None if allX is None else allX.shape)
    print("[check] #labels:", len(allY))

    print("\nAll report artifacts are saved under:", os.path.join(thisdir, OUTDIR))
