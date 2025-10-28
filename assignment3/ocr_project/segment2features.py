# segment2features.py
import numpy as np
from scipy import ndimage

EPS = 1e-8

def _bbox_of_mask(B):
    ys, xs = np.where(B)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())

def _perimeter_4n(B):
    # 4 邻域周长近似：轮廓 = B ^ erode(B)
    se = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=bool)
    er = ndimage.binary_erosion(B, structure=se)
    contour = B & (~er)
    return float(contour.sum())

def _holes_and_euler(B):
    filled = ndimage.binary_fill_holes(B)
    holes = filled & (~B)
    _, n_comp  = ndimage.label(B)
    _, n_holes = ndimage.label(holes)
    euler = int(n_comp) - int(n_holes)
    return int(n_holes), int(euler)

def _projection_peaks(B):
    s = B.sum()
    if s == 0:
        return 0.0, 0.0
    row_sums = B.sum(axis=1).astype(np.float32)
    col_sums = B.sum(axis=0).astype(np.float32)
    row_peak = float(row_sums.max() / (s + EPS))
    col_peak = float(col_sums.max() / (s + EPS))
    return row_peak, col_peak

def _symmetry_scores(B_crop):
    if B_crop.size == 0:
        return 0.0, 0.0
    Blr = np.fliplr(B_crop)
    Bud = np.flipud(B_crop)
    inter_lr = np.logical_and(B_crop, Blr).sum()
    union_lr = np.logical_or(B_crop, Blr).sum() + EPS
    inter_ud = np.logical_and(B_crop, Bud).sum()
    union_ud = np.logical_or(B_crop, Bud).sum() + EPS
    s_lr = float(inter_lr / union_lr)
    s_ud = float(inter_ud / union_ud)
    return s_lr, s_ud

def _central_moments(B):
    ys, xs = np.nonzero(B)
    if ys.size == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    y_mean = ys.mean()
    x_mean = xs.mean()
    dy = ys - y_mean
    dx = xs - x_mean
    mu20 = float((dx*dx).sum())
    mu02 = float((dy*dy).sum())
    mu11 = float((dx*dy).sum())
    return y_mean, x_mean, mu20, mu02, mu11

def _eccentricity(mu20, mu02, mu11):
    a = (mu20 + mu02) / 2.0
    b = np.sqrt(((mu20 - mu02) / 2.0)**2 + (mu11**2))
    l1 = a + b + EPS
    l2 = max(a - b, EPS)
    return float(np.sqrt(1.0 - (l2 / l1)))  # [0,1)

def _hu_moments_7(B):
    ys, xs = np.nonzero(B)
    if ys.size == 0:
        return (0.0,)*7
    y_mean = ys.mean()
    x_mean = xs.mean()
    dy = ys - y_mean
    dx = xs - x_mean
    m00 = float(len(xs))
    def mu(p,q):  return float((dx**p * dy**q).sum())
    def eta(p,q): return mu(p,q) / (m00 ** (1.0 + 0.5*(p+q)) + EPS)

    n20, n02, n11 = eta(2,0), eta(0,2), eta(1,1)
    n30, n03 = eta(3,0), eta(0,3)
    n12, n21 = eta(1,2), eta(2,1)

    phi1 = n20 + n02
    phi2 = (n20 - n02)**2 + 4*(n11**2)
    phi3 = (n30 - 3*n12)**2 + (3*n21 - n03)**2
    phi4 = ((n30 + n12)**2 + (n21 + n03)**2)
    phi5 = (n30 - 3*n12)*(n30 + n12)*(((n30 + n12)**2) - 3*((n21 + n03)**2)) + \
           (3*n21 - n03)*(n21 + n03)*(3*((n30 + n12)**2) - ((n21 + n03)**2))
    phi6 = (n20 - n02)*(((n30 + n12)**2) - ((n21 + n03)**2)) + 4*n11*(n30 + n12)*(n21 + n03)
    phi7 = (3*n21 - n03)*(n30 + n12)*(((n30 + n12)**2) - 3*((n21 + n03)**2)) - \
           (n30 - 3*n12)*(n21 + n03)*(3*((n30 + n12)**2) - ((n21 + n03)**2))

    def hulog(x): return float(np.sign(x) * np.log10(abs(x) + 1e-12))
    return tuple(hulog(p) for p in (phi1,phi2,phi3,phi4,phi5,phi6,phi7))

def _cell_densities(B, nrows, ncols):
    """把 bbox 裁剪区划分为 nrows x ncols 网格，返回每格的前景占比（按行优先展开）。"""
    ys, xs = np.where(B)
    if ys.size == 0:
        return [0.0] * (nrows*ncols)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    Bh = y1 - y0 + 1
    Bw = x1 - x0 + 1
    C = B[y0:y1+1, x0:x1+1]
    # 为避免极端尺寸对齐偏差，做等宽切片（最后一格吃掉余数）
    rs = np.linspace(0, Bh, nrows+1, dtype=int)
    cs = np.linspace(0, Bw, ncols+1, dtype=int)
    out = []
    tot = float(C.size) + EPS
    for i in range(nrows):
        for j in range(ncols):
            r0, r1 = rs[i], rs[i+1]
            c0, c1 = cs[j], cs[j+1]
            block = C[r0:r1, c0:c1]
            if block.size == 0:
                out.append(0.0)
            else:
                out.append(float(block.sum()) / (block.size + EPS))
    return out

def segment2features(Si: np.ndarray) -> np.ndarray:
    """
    输入：Si  (H, W) 二值掩膜 (0/1 或 False/True)
    输出：28 维列向量 (28,1)

    维度定义（与之前 14 维兼容并扩展到 28 维）：
      f1  = 面积比 (area / HW)
      f2  = 填充度 (area / bbox_area)
      f3  = bbox 宽高比 (w/h)
      f4  = 孔洞数
      f5  = Euler 数
      f6  = 紧致度 (perimeter^2 / area)
      f7  = 左右对称分数（IoU）
      f8  = 上下对称分数（IoU）
      f9  = 归一化笔画粗细 ~ 2*mean(EDT) / sqrt(area)
      f10 = 方向性峰值比 (row_peak / col_peak)
      f11..f17 = Hu 不变矩 #1..#7（对数带符号）
      f18 = 离心率（eccentricity，基于二阶中心矩）
      f19..f22 = 2x2 网格密度（4 维）
      f23..f28 = 2x3 网格密度（6 维）
    """
    B = (Si > 0).astype(bool)
    H, W = B.shape
    total = float(H*W)
    area = float(B.sum())
    if area == 0 or total == 0:
        return np.zeros((28, 1), dtype=np.float32)

    # bbox 与裁剪
    y0, y1, x0, x1 = _bbox_of_mask(B)
    Bh = y1 - y0 + 1
    Bw = x1 - x0 + 1
    Bcrop = B[y0:y1+1, x0:x1+1]

    # 1) 基础几何
    f1 = area / (total + EPS)
    f2 = area / (float(Bh*Bw) + EPS)
    f3 = float(Bw / (Bh + EPS))
    holes, euler = _holes_and_euler(B)
    f4 = float(holes)
    f5 = float(euler)
    per = _perimeter_4n(B)
    f6 = float((per*per) / (area + EPS))

    # 2) 对称性
    sym_lr, sym_ud = _symmetry_scores(Bcrop)
    f7, f8 = float(sym_lr), float(sym_ud)

    # 3) 粗细 & 投影峰值
    edt = ndimage.distance_transform_edt(B)
    mean_radius = float(edt[B].mean()) if area > 0 else 0.0
    f9 = float((2.0 * mean_radius) / (np.sqrt(area) + EPS))
    row_peak, col_peak = _projection_peaks(B)
    f10 = float(row_peak / (col_peak + EPS))

    # 4) Hu 1..7
    hu1, hu2, hu3, hu4, hu5, hu6, hu7 = _hu_moments_7(B)

    # 5) 离心率
    _, _, mu20, mu02, mu11 = _central_moments(B)
    ecc = _eccentricity(mu20, mu02, mu11)

    # 6) 网格密度（局部形状分布）
    dens_2x2 = _cell_densities(B, 2, 2)  # 4 维
    dens_2x3 = _cell_densities(B, 2, 3)  # 6 维

    f = np.array([
        f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,
        hu1, hu2, hu3, hu4, hu5, hu6, hu7,
        ecc,
        *dens_2x2,
        *dens_2x3
    ], dtype=np.float32)

    assert f.size == 28
    return f.reshape(-1, 1)
