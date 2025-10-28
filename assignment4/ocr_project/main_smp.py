#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
im2segment.py
-------------
将训练好的 U-Net 语义分割输出转换为作业要求的“实例掩膜列表”。

返回：List[np.ndarray]，每个掩膜与原图同尺寸，uint8，{0,1}。
"""

import os
import warnings
import numpy as np
from scipy import ndimage

# 只提示一次
warnings.filterwarnings("once", category=UserWarning)

# -----------------------------
# 配置
# -----------------------------
NUM_CLASSES = 11          # 总类别数
MIN_INSTANCE_PIX = 10     # 实例最小像素数（去小噪声）
CONNECTIVITY = 2          # 连通性：1=4邻, 2=8邻
OPENING_KERNEL = 0        # 轻微开运算去毛刺；0=关闭，3/5=核大小
CONF_THRESHOLD = None     # 低置信度像素强制设为背景；None=关闭，或设为 0.45~0.6

WEIGHT_PATHS = [
    os.path.join(os.path.dirname(__file__), "unet_m2nist.pth"),
    os.path.join(os.getcwd(), "unet_m2nist.pth"),
]

# 背景类别索引（默认0；加载权重后将自动探测覆盖）
BG_CLASS = 0

# -----------------------------
# 懒加载模型
# -----------------------------
_model = None
_device = None

def _to_gray01(im: np.ndarray) -> np.ndarray:
    """RGB/灰度转 float32 的 [0,1] 灰度图。"""
    im = im.astype(np.float32)
    if im.ndim == 3 and im.shape[2] == 3:
        # 感知亮度加权比简单均值更稳
        r, g, b = im[..., 0], im[..., 1], im[..., 2]
        im = 0.299 * r + 0.587 * g + 0.114 * b
    # 归一化到 [0,1]
    mn, mx = float(im.min()), float(im.max())
    im = (im - mn) / (mx - mn + 1e-12)
    return im

def _load_model():
    """
    懒加载分割模型。只在第一次调用时加载，并自动探测背景类索引。
    """
    global _model, _device, BG_CLASS
    if _model is not None:
        return _model, _device

    import torch
    import segmentation_models_pytorch as smp

    # 设备选择
    _device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 与训练时一致：resnet18 encoder、灰度单通道、NUM_CLASSES 类
    _model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,   # 灰度训练，未用 imagenet 预训练
        in_channels=1,
        classes=NUM_CLASSES
    ).to(_device)

    # 寻找权重文件
    weight_path = next((p for p in WEIGHT_PATHS if os.path.isfile(p)), None)
    if weight_path is None:
        raise FileNotFoundError(
            "未找到分割权重文件 'unet_m2nist.pth'。\n"
            "请将权重放在：\n  - 与 im2segment.py 同目录；或\n  - 当前工作目录。\n"
            "当前搜索路径：\n  " + "\n  ".join(WEIGHT_PATHS)
        )

    # —— 更健壮的加载 + 校验 ——
    state = torch.load(weight_path, map_location=_device)
    # 兼容常见保存格式（state_dict / model / 纯字典）
    if isinstance(state, dict) and any(k in state for k in ["state_dict", "model"]):
        state = state.get("state_dict", state.get("model", state))
    # 去掉可能的 "module." 前缀（DataParallel 保存时常见）
    if isinstance(state, dict):
        state = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }

    missing, unexpected = _model.load_state_dict(state, strict=False)
    print(f"[weights] loaded: {weight_path}")
    print(f"[weights] missing={len(missing)} unexpected={len(unexpected)}")

    # 零图像 sanity check：自动探测“背景类索引”
    with torch.no_grad():
        z = torch.zeros(1, 1, 64, 84, device=_device)  # 尺寸随意
        mu = _model(z).mean(dim=(0, 2, 3)).detach().cpu().numpy()  # (C,)
        bg_guess = int(mu.argmax())
        print("[bg] detected background class index =", bg_guess)
        BG_CLASS = bg_guess  # 覆盖全局 BG_CLASS

    _model.eval()
    return _model, _device

# -----------------------------
# 后处理：语义分割 -> 实例掩膜（类无关 + 小岛剔除）
# -----------------------------
def _semantic_to_instances(pred_cls: np.ndarray) -> list:
    """
    类无关实例化：
      1) 合并前景 fg = (pred_cls != BG_CLASS)
      2) （可选）开运算去毛刺
      3) 8邻连通域 -> 候选实例（过滤小块）
      4) 剔除“被更大实例的填洞区域完全包含”的小岛（避免 8/9 的孔成为独立实例）
      5) 按 xmin 排序，返回 0/1 uint8 掩膜列表
    """
    H, W = pred_cls.shape
    fg = (pred_cls != BG_CLASS)

    # 可选：轻微开运算祛毛刺（不要闭运算，闭运算易粘连相邻数字）
    if OPENING_KERNEL and OPENING_KERNEL >= 2:
        se = np.ones((OPENING_KERNEL, OPENING_KERNEL), dtype=np.uint8)
        fg = ndimage.binary_opening(fg, structure=se)

    # 连通域（8邻）
    structure = ndimage.generate_binary_structure(2, CONNECTIVITY)
    labeled, n = ndimage.label(fg, structure=structure)

    comps, areas = [], []
    for k in range(1, n + 1):
        c = (labeled == k)
        a = int(c.sum())
        if a >= MIN_INSTANCE_PIX:
            comps.append(c)
            areas.append(a)

    if not comps:
        return []

    # 剔除“被包含的小岛”：若 c_i 完全包含于某个更大实例的“填洞区域”中，则丢弃
    filled = [ndimage.binary_fill_holes(c) for c in comps]  # 仅用于包含判定
    keep = [True] * len(comps)
    order = np.argsort(areas)  # 从小到大
    for idx_small in order:
        if not keep[idx_small]:
            continue
        c_small = comps[idx_small]
        a_small = areas[idx_small]
        # 与更大的比较
        for idx_big in order[::-1]:
            if areas[idx_big] <= a_small:
                break
            if (c_small & filled[idx_big]).sum() == a_small:
                keep[idx_small] = False
                break

    masks = [comps[i].astype(np.uint8) for i, k in enumerate(keep) if k]

    # 从左到右排序
    def _xmin(m: np.ndarray):
        cols = np.where(m.any(axis=0))[0]
        return cols[0] if cols.size else 1e9

    masks.sort(key=_xmin)
    return [(m > 0).astype(np.uint8) for m in masks]

# -----------------------------
# 核心接口：im2segment
# -----------------------------
def im2segment(im: np.ndarray):
    """
    输入：im，灰度图 (H,W) 或彩色图 (H,W,3)，值域任意。
    输出：List[np.ndarray]，每个为 (H,W) uint8 掩膜（{0,1}），同尺寸。
    """
    # 预处理：转灰度 + 归一化到 [0,1]
    im = _to_gray01(im)

    model, device = _load_model()

    import torch
    with torch.no_grad():
        # 一次推理
        x = torch.from_numpy(im[None, None, ...]).float().to(device)  # 1x1xH xW
        logits = model(x)                                             # 1xC xH xW
        pred_cls = logits.argmax(1)[0].cpu().numpy().astype(np.int32)  # HxW

        # —— 可选：低置信度像素强制设为背景 ——
        if CONF_THRESHOLD is not None:
            prob = torch.softmax(logits, dim=1)[0].cpu().numpy()      # (C,H,W)
            maxp = prob.max(axis=0)                                   # (H,W)
            pred_cls[maxp < float(CONF_THRESHOLD)] = BG_CLASS

        # —— 统计前景占比与洞内类别（用于调试/诊断） ——
        fg = (pred_cls != BG_CLASS)
        print("fg_ratio =", float(fg.mean()))
        holes = ndimage.binary_fill_holes(fg) & (~fg)
        if holes.any():
            vals, cnts = np.unique(pred_cls[holes], return_counts=True)
            print("classes_in_holes =", dict(zip(vals.tolist(), cnts.tolist())))
        else:
            print("classes_in_holes = {}  (no holes)")

        # —— 极性自检/自动翻转（若几乎全前景或全背景） ——
        if float(fg.mean()) > 0.95 or float(fg.mean()) < 0.01:
            print("[warn] abnormal fg_ratio; trying auto invert...")
            im_inv = 1.0 - im
            x2 = torch.from_numpy(im_inv[None, None, ...]).float().to(device)
            logits2 = model(x2)
            pred2 = logits2.argmax(1)[0].cpu().numpy().astype(np.int32)
            if CONF_THRESHOLD is not None:
                prob2 = torch.softmax(logits2, dim=1)[0].cpu().numpy()
                maxp2 = prob2.max(axis=0)
                pred2[maxp2 < float(CONF_THRESHOLD)] = BG_CLASS
            fg2 = (pred2 != BG_CLASS)
            print("fg_ratio_after_invert =", float(fg2.mean()))
            # 选择更合理的一边（更接近中等占比）
            if abs(float(fg2.mean()) - 0.15) < abs(float(fg.mean()) - 0.15):
                pred_cls = pred2
                fg = fg2

        # 实例化
        masks = _semantic_to_instances(pred_cls)

        # 返回 0/1 uint8 掩膜（已在实例化中确保）
        return masks


# -----------------------------
# 自测（可选）
# -----------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 测一张数据集里的图（若有）
    if os.path.isfile("combined.npy"):
        img = np.load("combined.npy")[0]  # 改索引看不同样本
    else:
        # 随机图占位（仅验证流程，不代表有效分割）
        H, W = 64, 84
        img = (np.random.rand(H, W) * 255).astype(np.uint8)

    masks = im2segment(img)

    # 画输入+每个实例掩膜
    cols = 1 + max(1, len(masks))
    fig, axs = plt.subplots(1, cols, figsize=(3*cols, 3))
    axs = np.atleast_1d(axs)
    axs[0].imshow(img if img.ndim == 2 else img[...,0], cmap="gray")
    axs[0].set_title("Input")
    axs[0].axis("off")

    for i, m in enumerate(masks, start=1):
        axs[i].imshow(m, cmap="gray", vmin=0, vmax=1)
        axs[i].set_title(f"Mask {i} (sum={int(m.sum())})")
        axs[i].axis("off")

    plt.tight_layout()
    plt.show()
