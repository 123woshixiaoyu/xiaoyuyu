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
    # 4-邻域近似周长：轮廓 = B ^ erode(B)
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
    # 与左右翻转 / 上下翻转的重合程度（IoU风格的相似度）
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
    # 0/1 图 -> 质心 & 二阶中心矩（用以主轴、离心率）
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

def _eccentricity_orientation(mu20, mu02, mu11):
    # 协方差矩阵的特征值求离心率 & 主轴方向
    cov20 = mu20
    cov02 = mu02
    cov11 = mu11
    a = (cov20 + cov02) / 2.0
    b = np.sqrt(((cov20 - cov02) / 2.0)**2 + (cov11**2))
    l1 = a + b + EPS
    l2 = max(a - b, EPS)
    ecc = float(np.sqrt(1.0 - (l2 / l1)))  # [0,1)
    theta = 0.5 * np.arctan2(2.0*cov11, (cov20 - cov02 + EPS))  # 主轴角
    return ecc, theta

def _hu_moments_first4(B):
    # 基于中心矩计算 Hu 前4项，不依赖skimage
    ys, xs = np.nonzero(B)
    if ys.size == 0:
        return (0.0, 0.0, 0.0, 0.0)
    y_mean = ys.mean()
    x_mean = xs.mean()
    dy = ys - y_mean
    dx = xs - x_mean
    m00 = float(len(xs))
    # 归一化中心矩 η_pq = μ_pq / m00^{1 + (p+q)/2}
    def mu(p, q):
        return float((dx**p * dy**q).sum())
    def eta(p, q):
        return mu(p, q) / (m00 ** (1.0 + 0.5*(p+q)) + EPS)

    n20, n02, n11 = eta(2,0), eta(0,2), eta(1,1)
    n30, n03 = eta(3,0), eta(0,3)
    n12, n21 = eta(1,2), eta(2,1)

    # Hu 不变矩（前4个）
    phi1 = n20 + n02
    phi2 = (n20 - n02)**2 + 4*(n11**2)
    phi3 = (n30 - 3*n12)**2 + (3*n21 - n03)**2
    phi4 = ((n30 + n12)**2 + (n21 + n03)**2)
    # 对数压缩（保留符号）：sign(x)*log10(|x|+eps)
    def hulog(x):
        return float(np.sign(x) * np.log10(abs(x) + 1e-12))
    return (hulog(phi1), hulog(phi2), hulog(phi3), hulog(phi4))

def segment2features(Si: np.ndarray) -> np.ndarray:
    """
    输入：Si  (H, W) 二值掩膜 (0/1 或 False/True)
    输出：14维列向量 (14,1)
    维度说明：
      f1: 面积比 (area/图像面积)
      f2: 填充度/矩形度 (area/外接框面积)
      f3: 宽高比 (bbox w/h)
      f4: 孔洞数
      f5: Euler 数
      f6: 紧致度 (perimeter^2 / area)
      f7: 左右对称分数
      f8: 上下对称分数
      f9: 笔画粗细(尺度无关) ~ 2*mean(EDT)/sqrt(area)
      f10: 方向性峰值比 (row_peak/col_peak)
      f11-f14: Hu 不变矩前4项的对数形式
    """
    B = (Si > 0).astype(bool)
    H, W = B.shape
    total = float(H * W)
    area  = float(B.sum())
    if area == 0 or total == 0:
        return np.zeros((14, 1), dtype=np.float32)

    # bbox 与裁剪
    y0, y1, x0, x1 = _bbox_of_mask(B)
    Bh = y1 - y0 + 1
    Bw = x1 - x0 + 1
    Bcrop = B[y0:y1+1, x0:x1+1]

    # 1) 面积比
    f1 = area / total

    # 2) 填充度(矩形度) = area / bbox_area
    bbox_area = float(Bh * Bw)
    f2 = area / (bbox_area + EPS)

    # 3) 宽高比
    f3 = float(Bw / (Bh + EPS))

    # 4) 孔洞数 & 5) Euler
    holes, euler = _holes_and_euler(B)
    f4 = float(holes)
    f5 = float(euler)

    # 6) 紧致度
    per = _perimeter_4n(B)
    f6 = float((per * per) / (area + EPS))

    # 7-8) 对称性（在bbox内评估，减少背景干扰）
    sym_lr, sym_ud = _symmetry_scores(Bcrop)
    f7, f8 = float(sym_lr), float(sym_ud)

    # 9) 笔画粗细（EDT 距离变换的平均半径，做尺度归一）
    #   mean_radius = mean(EDT[B])；thickness ~ 2*mean_radius
    #   为去尺度影响，用 thickness / sqrt(area)
    edt = ndimage.distance_transform_edt(B)
    mean_radius = float(edt[B].mean()) if area > 0 else 0.0
    f9 = float((2.0 * mean_radius) / (np.sqrt(area) + EPS))

    # 10) 投影峰值比
    row_peak, col_peak = _projection_peaks(B)
    f10 = float(row_peak / (col_peak + EPS))

    # 11-14) Hu 不变矩（前4个, 对数形式）
    phi1, phi2, phi3, phi4 = _hu_moments_first4(B)
    f11, f12, f13, f14 = float(phi1), float(phi2), float(phi3), float(phi4)

    f = np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14],
                 dtype=np.float32)
    return f.reshape(-1, 1)
