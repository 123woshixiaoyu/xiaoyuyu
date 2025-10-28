#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M2NIST 多数字语义分割（图像 -> 11类掩膜：10个数字 + 背景）

依赖（在你的 4.OCR 环境）：
    pip install segmentation-models-pytorch==0.3.3 albumentations==1.4.8 scikit-learn
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


# --------- 配置：如果数据在其他文件夹，改这两行即可 ---------
COMBINED_PATH = os.path.join("combined.npy")
SEGMENTED_PATH = os.path.join("segmented.npy")
# -----------------------------------------------------------


def set_seed(s: int = 42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------- Dataset ----------
class M2NIST(Dataset):
    """
    combined.npy: (N, 64, 84) 灰度图，0~255 或 0~1
    segmented.npy: (N, 64, 84, 11) one-hot
    """
    def __init__(self, X, Y, train=True):
        # 归一化到 [0,1]
        self.X = X.astype(np.float32)
        if self.X.max() > 1.5:
            self.X /= 255.0

        # one-hot -> 类别索引(0..10)，0常为背景
        self.Y = np.argmax(Y, axis=-1).astype(np.int64)

        if train:
            self.tf = A.Compose([
                A.OneOf([
                    A.MotionBlur(blur_limit=3, p=0.3),
                    A.GaussianBlur(blur_limit=3, p=0.3),
                    A.GaussNoise(var_limit=(5.0, 15.0), p=0.4),
                ], p=0.5),
                A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
                A.CoarseDropout(max_holes=2, max_height=8, max_width=8, fill_value=0, p=0.2),
                ToTensorV2(),
            ])
        else:
            self.tf = A.Compose([ToTensorV2()])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        img = self.X[i]          # (H,W) 例：64×84
        mask = self.Y[i]         # (H,W) int64 类别图

        # Albumentations 需要 HWC，这里给一个虚拟的通道维
        aug = self.tf(image=img[..., None], mask=mask)   # image: H×W×1
        img_t = aug["image"].float()                     # -> 1×H×W（ToTensorV2已是CHW）
        mask_t = aug["mask"].long()                      # -> H×W（long）

        return img_t, mask_t


# ---------- Metric ----------
@torch.no_grad()
def iou_score(pred, target, num_classes=11, eps=1e-6):
    """
    pred: (N,C,H,W) logits
    target: (N,H,W) int64
    计算宏平均 IoU（忽略未出现的类）
    """
    pred_cls = pred.argmax(1)   # (N,H,W)
    ious = []

    # 按类计算 IoU（可根据需要忽略背景类0）
    for c in range(num_classes):
        p = (pred_cls == c)
        t = (target   == c)
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        if union == 0:
            continue
        ious.append((inter + eps) / (union + eps))
    return sum(ious)/len(ious) if ious else 0.0


def main():
    set_seed(42)

    # ---------- 路径检查 ----------
    here = os.path.dirname(os.path.abspath(__file__))
    combined_path = COMBINED_PATH if os.path.isabs(COMBINED_PATH) else os.path.join(here, COMBINED_PATH)
    segmented_path = SEGMENTED_PATH if os.path.isabs(SEGMENTED_PATH) else os.path.join(here, SEGMENTED_PATH)

    if not os.path.isfile(combined_path) or not os.path.isfile(segmented_path):
        raise FileNotFoundError(
            f"找不到数据文件：\n  {combined_path}\n  {segmented_path}\n"
            "请确认文件路径是否正确，或修改脚本顶部的 COMBINED_PATH / SEGMENTED_PATH。"
        )

    # ---------- 载入数据 ----------
    X = np.load(combined_path)     # (N, 64, 84)
    Y = np.load(segmented_path)    # (N, 64, 84, 11)

    if X.ndim != 3 or Y.ndim != 4 or X.shape[:2] != (64, 84)[:2] or Y.shape[-1] != 11:
        print("[WARN] 数据形状看起来与预期不一致：")
        print("  X.shape =", X.shape, "  Y.shape =", Y.shape)

    # ---------- 划分训练/验证 ----------
    N = len(X)
    if N < 2:
        raise ValueError("样本数太少，至少需要 2 个样本以划分训练/验证。")

    # 验证集规模：max(250, 0.1N)，但不得 >= N
    val_size = max(250, int(0.1 * N))
    if val_size >= N:
        val_size = max(1, N // 5)

    idx_all = np.arange(N)
    idx_train, idx_val = train_test_split(idx_all, test_size=val_size, random_state=42, shuffle=True)

    train_set = M2NIST(X[idx_train], Y[idx_train], train=True)
    val_set   = M2NIST(X[idx_val],   Y[idx_val],   train=False)

    # Windows 下先用 num_workers=0 跑通；稳定后可改为 2/4
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    # ---------- Model ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "| cuda:", torch.cuda.is_available())
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,   # 灰度输入 -> 不加载 ImageNet 权重
        in_channels=1,
        classes=11
    ).to(device)

    # ---------- Loss / Optim ----------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ---------- Train loop ----------
    best_iou = 0.0
    epochs = 30
    for epoch in range(epochs):
        # ---- train ----
        model.train()
        tr_loss = 0.0
        tr_iou  = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)                 # (N,11,H,W)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            tr_loss += loss.item() * bs
            tr_iou  += iou_score(logits.detach(), y) * bs

        tr_loss /= len(train_loader.dataset)
        tr_iou  /= len(train_loader.dataset)

        # ---- val ----
        model.eval()
        va_loss = 0.0
        va_iou  = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = model(x)
                loss = criterion(logits, y)

                bs = x.size(0)
                va_loss += loss.item() * bs
                va_iou  += iou_score(logits, y) * bs

        va_loss /= len(val_loader.dataset)
        va_iou  /= len(val_loader.dataset)

        print(f"Epoch {epoch+1:02d} | train loss {tr_loss:.4f} IoU {tr_iou:.3f} | "
              f"val loss {va_loss:.4f} IoU {va_iou:.3f}")

        if va_iou > best_iou:
            best_iou = va_iou
            torch.save(model.state_dict(), "unet_m2nist.pth")
            print(f"  -> 保存最佳权重：unet_m2nist.pth  (val IoU={best_iou:.3f})")

    print("Best val IoU:", best_iou)


if __name__ == "__main__":
    main()
