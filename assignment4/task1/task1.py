#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:51:35 2024

@author: magnuso
"""

# -*- coding: utf-8 -*-
# ===== Task 1: Color correction of images =====

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from flip_1 import computeFLIP
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# ===== Path to the WB_sRGB Python implementation =====
sys.path.append('D:/Lund_university/ongoing/ia/asssignment/assignment4/task1/WB_sRGB/WB_sRGB_Python')
from classes import WBsRGB as wb_srgb


# ---------- 1. Basic Gray-World method ----------
def gray_world(im):
    """
    Implements the basic Gray-World white balance.
    Each color channel is scaled so that the average of the three channels
    becomes equal (assuming the average scene color should be gray).
    """
    img = np.clip(im.astype(np.float32), 0.0, 1.0)
    mean_rgb = img.reshape(-1, 3).mean(axis=0) + 1e-8
    target = float(mean_rgb.mean())
    gain = target / mean_rgb
    img = img * gain[None, None, :]
    return np.clip(img, 0.0, 1.0)


# ---------- 2. Improved version ----------
def gray_world_mod(im, p=4, alpha=0.35, wp=95, eps=1e-8):
    """
    A refined version combining Shades-of-Gray and White-Patch concepts.
      1) Uses a p-norm (Shades-of-Gray) to compute more robust channel gains.
      2) Incorporates a percentile-based White-Patch correction (default 95%).
      3) Blends the two correction factors with weight alpha.
      4) Preserves global brightness by matching overall luminance.
    Parameters:
      p      – Minkowski norm (p=4 gives moderate correction)
      alpha  – blend ratio between SoG and WP gains (0.35 = moderate mix)
      wp     – percentile for white patch computation
    """
    img = np.clip(im.astype(np.float32), 0.0, 1.0)
    H, W, C = img.shape
    assert C == 3, "Expected RGB input"

    # Compute average luminance (for later exposure adjustment)
    def mean_luma(x):
        return float((0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]).mean())
    y_before = mean_luma(img)

    # (1) Shades-of-Gray gain
    Ip = (np.mean(np.power(img, p, dtype=np.float32), axis=(0, 1)) + eps) ** (1.0 / p)
    sog_target = float(np.mean(Ip))
    sog_gain = sog_target / Ip

    # (2) White-Patch gain (using percentile instead of max)
    wp_vals = np.array([np.percentile(img[..., c], wp) + eps for c in range(3)], dtype=np.float32)
    wp_target = float(np.mean(wp_vals))
    wp_gain = wp_target / wp_vals

    # (3) Blended gain
    gain = (1.0 - alpha) * sog_gain + alpha * wp_gain
    img_corr = img * gain[None, None, :]

    # (4) Match global brightness (keep similar overall exposure)
    y_after = mean_luma(np.clip(img_corr, 0.0, 1.0))
    if y_after > eps:
        exposure = y_before / y_after
        img_corr = img_corr * exposure

    return np.clip(img_corr, 0.0, 1.0)


# ---------- 3. WB_sRGB interface handling ----------
def _afifi_infer_any(wb_model, im_bgr_u8):
    """
    Automatically detects the correct inference method name for WB_sRGB.
    Different versions use different entry points such as:
    run / infer / render / correctImage / correct_image / process / __call__
    """
    candidates = ["run", "infer", "render", "correctImage", "correct_image", "process", "__call__"]
    last_err = None
    for name in candidates:
        try:
            if name == "__call__":
                return wb_model(im_bgr_u8)
            if hasattr(wb_model, name):
                fn = getattr(wb_model, name)
                return fn(im_bgr_u8)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"WBsRGB: No valid inference interface found. Last error: {last_err}")


# ---------- 4. Normalize Afifi output to RGB float32 [0,1] ----------
def _to_rgb01_any(x, assume_bgr=True):
    """
    Converts any WB_sRGB output format (uint8/float, BGR/RGB, list/dict)
    to a normalized RGB float32 array in [0,1].
    """
    if isinstance(x, (list, tuple)):
        x = x[0]
    elif isinstance(x, dict):
        for k in ["wb", "balanced", "corrected", "srgb", "out", "img"]:
            if k in x:
                x = x[k]
                break

    arr = np.asarray(x)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Unexpected WB_sRGB output shape: {arr.shape}")

    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
        m, M = float(arr.min()), float(arr.max())
        if M > 1.5:
            arr = np.clip(arr, 0.0, 255.0) / 255.0
        else:
            arr = np.clip(arr, 0.0, 1.0)

    if assume_bgr:
        arr = arr[:, :, ::-1]  # Convert BGR → RGB
    return np.clip(arr, 0.0, 1.0)


# ---------- 5. Metric computation ----------
def compute_metrics(ref01, test01):
    """
    Computes three evaluation metrics between the reference and test image:
      - FLIP
      - SSIM
      - PSNR
    """
    ref = np.clip(ref01.astype(np.float32), 0.0, 1.0)
    tst = np.clip(test01.astype(np.float32), 0.0, 1.0)
    try:
        flip_val, _ = computeFLIP(ref, tst)
    except:
        flip_val = computeFLIP(ref, tst)
    ssim_val = ssim(ref, tst, data_range=1.0, channel_axis=2)
    psnr_val = psnr(ref, tst, data_range=1.0)
    return flip_val, ssim_val, psnr_val


# ---------- 6. Main script ----------
if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    bad_path = os.path.join(here, "abbey_badcolor.jpg")
    ok_path = os.path.join(here, "abbey_correct.jpg")

    if not os.path.isfile(bad_path) or not os.path.isfile(ok_path):
        raise FileNotFoundError("Please place abbey_badcolor.jpg and abbey_correct.jpg in the same directory as this script.")

    # Load and normalize images to [0,1]
    im_bad = plt.imread(bad_path).astype(np.float32)
    if im_bad.max() > 1.5:
        im_bad = im_bad / 255.0
    im_ok = plt.imread(ok_path).astype(np.float32)
    if im_ok.max() > 1.5:
        im_ok = im_ok / 255.0

    # Visualize input vs reference
    plt.imshow(np.hstack((im_bad, im_ok)))
    plt.title("Input (bad) vs Reference (correct)")
    plt.axis('off')
    plt.show()

    # ---- Method 1 and 2 ----
    im_gray = gray_world(im_bad)
    im_gray_mod = gray_world_mod(im_bad)

    # ---- Method 3: Afifi WB_sRGB ----
    upgraded_model = 1
    gamut_mapping = 2
    wb = wb_srgb.WBsRGB(gamut_mapping=gamut_mapping, upgraded=upgraded_model)

    im_bad_bgr_u8 = (np.clip(im_bad, 0, 1)[:, :, ::-1] * 255.0).astype(np.uint8)
    raw_out = _afifi_infer_any(wb, im_bad_bgr_u8)
    im_afifi = _to_rgb01_any(raw_out, assume_bgr=True)

    # Show all outputs side by side
    plt.imshow(np.hstack((im_gray, im_gray_mod, im_afifi)))
    plt.title("Gray-World | Gray-World (Improved) | WB_sRGB")
    plt.axis('off')
    plt.show()

    # ---- Save output images ----
    plt.imsave("output_grayworld.png", im_gray)
    plt.imsave("output_grayworld_mod.png", im_gray_mod)
    plt.imsave("output_wbsrgb.png", im_afifi)
    print("Saved output images: output_grayworld.png, output_grayworld_mod.png, output_wbsrgb.png")

    # ---- Compute metrics ----
    flip_gray, ssim_gray, psnr_gray = compute_metrics(im_ok, im_gray)
    flip_mod, ssim_mod, psnr_mod = compute_metrics(im_ok, im_gray_mod)
    flip_afifi, ssim_afifi, psnr_afifi = compute_metrics(im_ok, im_afifi)

    # ---- Create result table ----
    df = pd.DataFrame({
        "Method": ["Gray-World", "Gray-World (Improved)", "WB_sRGB"],
        "FLIP (↓)": [flip_gray, flip_mod, flip_afifi],
        "SSIM (↑)": [ssim_gray, ssim_mod, ssim_afifi],
        "PSNR (↑)": [psnr_gray, psnr_mod, psnr_afifi],
    })

    print("\n===== Result Table =====")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}" if x < 1 else f"{x:.3f}"))

