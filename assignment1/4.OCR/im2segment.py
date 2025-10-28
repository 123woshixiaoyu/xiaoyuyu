import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

def im2segment(I, show_steps=False, connectivity=8):
    """
    Segment a light-on-dark text image into a list of instance masks (one digit per mask).
    Returns a list where each element is a binary (0/1) image with the same size as the input.

    Parameters
    ----------
    I : np.ndarray
        Input image. Either HxW grayscale or HxWx3 color.
    show_steps : bool
        If True, visualize key intermediate results (Otsu binary, closing) and each output mask.
    connectivity : int
        Connectivity option. The current implementation uses 8-connectivity (explicit 8-neighborhood).
    """
    # ---------- Preprocess: to grayscale + normalize to [0,1] ----------
    A = I.astype(np.float32)
    if A.ndim == 3:                      # convert color to grayscale
        A = A.mean(axis=2)
    if A.max() > 1.5:
        A /= 255.0

    # ---------- Otsu thresholding ----------
    hist, _ = np.histogram(A.ravel(), bins=256, range=(0, 1))
    total = A.size
    sumB = wB = maximum = 0.0
    sum1 = np.dot(np.arange(256), hist)
    level = 128
    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum1 - sumB) / wF
        between = wB * wF * (mB - mF) ** 2
        if between >= maximum:
            level, maximum = t, between
    fg_otsu = (A >= level / 255.0)

    # ---------- Morphological closing ----------
    fg_close = ndi.binary_closing(fg_otsu, structure=np.ones((1, 3)))
    fg = fg_close

    # ---------- Column projection to find active column runs ----------
    H, W = fg.shape
    colsum = fg.sum(axis=0)
    active = colsum >= max(2, int(0.02 * H))  # adaptive threshold

    runs, start = [], None
    for x, a in enumerate(active):
        if a and start is None:
            start = x
        if (not a) and start is not None:
            runs.append((start, x - 1))
            start = None
    if start is not None:
        runs.append((start, W - 1))

    # Merge small gaps between neighboring runs
    GAP_MAX = int(0.33 * H)   # tunable (H = image height, GAP_MAX measured in columns)
    merged = []
    for r in runs:
        if not merged or r[0] - merged[-1][1] > GAP_MAX:
            merged.append(list(r))
        else:
            merged[-1][1] = r[1]

    # ---------- Within each run, keep the largest connected components ----------
    S = []
    structure = np.ones((3, 3), dtype=bool) if connectivity == 8 else None
    for x0, x1 in merged:
        xl, xr = max(0, x0 - 1), min(W - 1, x1 + 1)
        block = fg[:, xl:xr + 1]
        labels, nlab = ndi.label(block, structure=structure)
        if nlab == 0:
            continue

        # Area filtering: keep the union of “largest component + components with area ≥ alpha * max_area”
        areas = np.array([(labels == k).sum() for k in range(1, nlab + 1)])
        max_area = areas.max()
        alpha = 0.20  # area relaxation ratio
        keep_ids = [i + 1 for i, a in enumerate(areas) if a >= alpha * max_area]

        merged_mask = np.isin(labels, keep_ids)

        mask = np.zeros_like(fg, dtype=np.uint8)
        mask[:, xl:xr + 1] = merged_mask.astype(np.uint8)
        S.append(mask)

    # ---------- Visualization ----------
    if show_steps:
        n_masks = len(S)
        plt.figure(figsize=(3 * (3 + n_masks), 3))

        plt.subplot(1, 3 + n_masks, 1); plt.imshow(A, cmap='gray');       plt.title("Input");         plt.axis('off')
        plt.subplot(1, 3 + n_masks, 2); plt.imshow(fg_otsu, cmap='gray'); plt.title("After Otsu");    plt.axis('off')
        plt.subplot(1, 3 + n_masks, 3); plt.imshow(fg_close, cmap='gray');plt.title("After Closing"); plt.axis('off')

        for i, m in enumerate(S):
            plt.subplot(1, 3 + n_masks, 4 + i)
            plt.imshow(m, cmap='gray'); plt.title(f"Mask {i}"); plt.axis('off')

        plt.tight_layout(); plt.show()

    return S
