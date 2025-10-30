# report_metrics.py
# Classical, training-free segmentation benchmark for multi-digit MNIST rows
# Strategy: Dense cut proposals (PSC + vp minima + grid + jitter) ->
#           boundary-aware DP (PSC reward + ink penalty + width prior + TTA-calibrated NLL) ->
#           optional recursive local resegment (PSC+seam) for wide merges.
# Outputs: confusion_matrix.png, per_class_accuracy.png, end_to_end_summary.png, dbg_XXX.png, EVAL_SUMMARY.md

import os
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from PIL import Image

# Optional deps
_HAS_SK = False
_HAS_CV2 = False
_HAS_SCIPY = False
try:
    from sklearn.metrics import confusion_matrix, classification_report
    _HAS_SK = True
except Exception:
    pass
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    pass
try:
    from scipy.ndimage import gaussian_filter1d, center_of_mass, shift as ndi_shift
    _HAS_SCIPY = True
except Exception:
    pass

# ---------------- CNN (same as training) ----------------
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)

# ---------------- Reproducibility ----------------
def set_seed(seed=1337):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ---------------- Data loaders ----------------
def get_test_loader(batch_size=512):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD)
    ])
    test_set = torchvision.datasets.MNIST(root="../data", train=False, transform=tf, download=True)
    return DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

def get_raw_testset():
    return torchvision.datasets.MNIST(root="../data", train=False, transform=transforms.ToTensor(), download=True)

# ---------------- MNIST-style preprocessing for segments ----------------
def mnist_style_28x28(arr_u8, use_otsu=True, bin_thresh=10, cm_center=True):
    """(H,W) uint8 -> (28,28) float in [0,1] (no normalization)"""
    assert arr_u8.ndim == 2
    img = arr_u8
    if _HAS_CV2 and use_otsu:
        _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        bw = (img > bin_thresh).astype(np.uint8) * 255

    ys, xs = np.where(bw > 0)
    if xs.size == 0 or ys.size == 0:
        return np.zeros((28, 28), dtype=np.float32)

    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    crop = img[y0:y1, x0:x1]  # uint8

    h, w = crop.shape
    if h >= w:
        new_h, new_w = 20, max(1, int(round(20 * w / h)))
    else:
        new_w, new_h = 20, max(1, int(round(20 * h / w)))

    small = np.asarray(Image.fromarray(crop, mode='L').resize((new_w, new_h), Image.LANCZOS), dtype=np.uint8)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    y_off = (28 - new_h) // 2
    x_off = (28 - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = small

    if cm_center and _HAS_SCIPY:
        m = (canvas > 0).astype(np.float32)
        if m.sum() > 0:
            cy, cx = center_of_mass(m)
            dy = (28/2 - 0.5) - cy
            dx = (28/2 - 0.5) - cx
            canvas = ndi_shift(canvas.astype(np.float32), shift=(dy, dx), order=1, mode='constant', cval=0.0)
            canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    return canvas.astype(np.float32) / 255.0

def normalize_28x28(arr28_float):
    t = torch.from_numpy(arr28_float).unsqueeze(0).unsqueeze(0).float()
    return (t - MNIST_MEAN[0]) / MNIST_STD[0]

# ---------------- Synthetic overlapped canvas ----------------
def create_segments(mnist_dataset, num_digits=3,
                    overlap_range=(1, 3), spacing=2,
                    left_pad=10, right_pad=10, blend='max',
                    crop_threshold=0.1, max_overlap_frac=0.12, p_no_overlap=0.25):
    digits, labels = [], []
    idxs = np.random.choice(len(mnist_dataset), size=num_digits, replace=False)
    for idx in idxs:
        img, label = mnist_dataset[idx]  # [1,28,28]
        img_np = img.squeeze().numpy()
        cols = np.where(img_np.max(axis=0) > crop_threshold)[0]
        if len(cols) == 0: 
            continue
        start, end = cols[0], cols[-1]
        content = img_np[:, start:end+1]
        digits.append(content.astype(np.float32))
        labels.append(int(label))
    if not digits:
        raise RuntimeError("No digits selected contained content.")
    overlaps = []
    if len(digits) > 1:
        lo, hi = overlap_range
        for i in range(len(digits)-1):
            cap = int(max_overlap_frac * min(digits[i].shape[1], digits[i+1].shape[1])); cap = max(1, cap)
            upper, lower = min(hi, cap), lo
            if upper < lower: upper = lower
            ov = np.random.randint(lower, upper+1)
            if np.random.rand() < p_no_overlap: ov = 0
            overlaps.append(ov)
    widths = [d.shape[1] for d in digits]
    total_width = left_pad + sum(widths) - sum(overlaps) + spacing * (len(digits)-1) + right_pad
    canvas = np.zeros((28, total_width), dtype=np.float32)
    true_segments = []
    x = left_pad
    for i, content in enumerate(digits):
        w = content.shape[1]
        region = canvas[:, x:x+w]
        if blend == 'max':
            canvas[:, x:x+w] = np.maximum(region, content)
        else:
            canvas[:, x:x+w] = np.clip(region + content, 0.0, 1.0)
        true_segments.append((x, x+w))
        if i < len(digits)-1:
            x = x + w - overlaps[i] + spacing
    return canvas, labels, true_segments

# ---------------- Classic helpers ----------------
def _smooth_1d(x, sigma=2.0):
    if _HAS_SCIPY:
        return gaussian_filter1d(x.astype(np.float32), sigma=sigma)
    win = max(3, int(2*sigma) | 1)
    k = np.ones(win, dtype=np.float32)/win
    return np.convolve(x.astype(np.float32), k, mode='same')

def _otsu_or_thresh_u8(arr_u8, thresh=10, open_iters=0, close_iters=1):
    """Preserve narrow vertical valleys: avoid strong opening; prefer vertical closing."""
    if _HAS_CV2:
        if thresh == 0:
            _, mask = cv2.threshold(arr_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, mask = cv2.threshold(arr_u8, thresh, 255, cv2.THRESH_BINARY)
        if close_iters > 0:
            k_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_vert, iterations=close_iters)
        if open_iters > 0:
            k_light = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_light, iterations=open_iters)
        return mask
    return ((arr_u8 > (thresh or 10)) * 255).astype(np.uint8)

def _reservoir_scores(mask255):
    H, W = mask255.shape
    col = (mask255 > 0).astype(np.uint8)
    top_gap = np.full(W, H, dtype=np.int32)
    bot_gap = np.full(W, H, dtype=np.int32)
    for c in range(W):
        ys = np.flatnonzero(col[:, c])
        if ys.size:
            top_gap[c] = int(ys[0])
            bot_gap[c] = int(H-1-ys[-1])
    top_s = _smooth_1d(np.clip(top_gap / max(1, H), 0, 1), 1.2)
    bot_s = _smooth_1d(np.clip(bot_gap / max(1, H), 0, 1), 1.2)
    return np.maximum(top_s, bot_s)

def _shear_min_cov(arr_u8, max_abs=0.45):
    """Shear x' = x + s*y to minimize cov(x', y)."""
    mask = (arr_u8 > 10).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if xs.size < 40:
        return arr_u8, 0.0
    x = xs.astype(np.float32); y = ys.astype(np.float32)
    x -= x.mean(); y -= y.mean()
    var_y  = float(np.var(y) + 1e-6)
    cov_xy = float(np.mean(x * y))
    s = float(-cov_xy / var_y)
    s = max(-max_abs, min(max_abs, s))
    H, W = arr_u8.shape
    if _HAS_CV2:
        cy = (H - 1) / 2.0
        M = np.array([[1, s, -s*cy], [0, 1, 0]], dtype=np.float32)
        out = cv2.warpAffine(arr_u8, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    else:
        cy = (H - 1) / 2.0
        out = np.asarray(
            Image.fromarray(arr_u8).transform(
                (W, H), Image.AFFINE, (1, s, -s*cy, 0, 1, 0),
                resample=Image.BILINEAR
            ),
            dtype=np.uint8
        )
    return out, s

# ---------------- PSC profile + dense cut augmentation ----------------
def _compute_psc_profile(arr_u8):
    """Per-column boundary likelihood: multi-scale valley + reservoirs + (1-|∂x|) gradient gap."""
    mask = _otsu_or_thresh_u8(arr_u8, thresh=10, open_iters=0, close_iters=1)
    vp = (mask > 0).sum(axis=0).astype(np.float32)
    def valley(v, sig):
        if v.max() <= v.min(): return np.zeros_like(v)
        raw = (v.max() - v) / max(1e-6, (v.max() - v.min()))
        return _smooth_1d(raw, sig)
    s = np.zeros_like(vp)
    for sig in (0.8, 1.4, 2.2):
        s = np.maximum(s, 0.55 * valley(vp, sig))
    res = _reservoir_scores(mask)  # [0..1]
    s += 0.30 * res
    if _HAS_CV2:
        gx = np.abs(cv2.Sobel(arr_u8, cv2.CV_32F, 1, 0, ksize=3)).sum(axis=0)
        if gx.max() > 0:
            grad_gap = 1.0 - (gx / gx.max())
            s += 0.25 * _smooth_1d(grad_gap, 1.0)
    s = np.clip(s, 0, 1)
    return s, mask, vp

def _augment_cuts_dense(psc, vp, H, base_peaks, min_gap_px=6, jitter=1, topk_extra=24, grid_every=None):
    """Unify PSC peaks + vp minima + uniform grid + jitter, then NMS by min_gap_px."""
    W = len(psc)
    cand = set(base_peaks)
    # local minima of column ink (vp)
    mins = [i for i in range(1, W-1) if vp[i] <= vp[i-1] and vp[i] <= vp[i+1]]
    th_vp = np.percentile(vp, 40.0)
    for i in mins:
        if vp[i] <= th_vp:
            cand.add(i)
    # top-K PSC peaks anywhere
    tops = sorted(range(1, W-1), key=lambda i: psc[i], reverse=True)[:topk_extra]
    for i in tops:
        cand.add(i)
    # uniform grid ~ one per digit-height
    if grid_every is None:
        grid_every = max(6, int(round(0.85 * H)))
    for x in range(grid_every, W, grid_every):
        cand.add(x)
    # ±1 jitter around each candidate
    expanded = []
    for c in cand:
        for d in range(-jitter, jitter+1):
            j = c + d
            if 1 <= j < W-1:
                expanded.append(j)
    # NMS with PSC score
    expanded = sorted(set(expanded), key=lambda j: psc[j], reverse=True)
    keep, used = [], np.zeros(W, dtype=bool)
    for c in expanded:
        L = max(0, c - min_gap_px); R = min(W, c + min_gap_px + 1)
        if used[L:R].any():
            continue
        keep.append(c); used[L:R] = True
    cuts = [0] + sorted(keep) + [W]
    return cuts, keep

def valley_shave(arr_u8, cols, band=2, factor=0.55):
    out = arr_u8.copy()
    for c in cols:
        a = max(0, c-band); b = min(arr_u8.shape[1], c+band+1)
        out[:, a:b] = (out[:, a:b].astype(np.float32) * factor).astype(np.uint8)
    return out

# ---------------- Segment scoring (TTA + temperature) ----------------
@torch.no_grad()
def _predict_proba_tta(model, t_norm, device, angles=(-6, 0, 6), temp=1.5):
    """Small rotations + temperature scaling for better-calibrated probs."""
    x01 = t_norm.cpu() * MNIST_STD[0] + MNIST_MEAN[0]
    arr = (x01.squeeze().numpy().clip(0,1) * 255).astype(np.uint8)
    imgs = []
    for a in angles:
        pil = Image.fromarray(arr, mode='L').rotate(a, resample=Image.BILINEAR, expand=False, fillcolor=0)
        ar = np.asarray(pil, dtype=np.float32) / 255.0
        tt = torch.from_numpy(ar).unsqueeze(0).unsqueeze(0).float()
        tt = (tt - MNIST_MEAN[0]) / MNIST_STD[0]
        imgs.append(tt)
    batch = torch.cat(imgs, dim=0).to(device)
    logits = model(batch) / float(temp)
    probs = torch.softmax(logits, dim=1).mean(dim=0).cpu().numpy()
    return probs

@torch.no_grad()
def _segment_score(arr_for_scoring_u8, model, device, x0, x1, use_otsu=True, tta=True, temp=1.5):
    H = arr_for_scoring_u8.shape[0]; w = x1 - x0
    if w <= 0:
        return 9e9, None, 0.0
    arr28 = mnist_style_28x28(arr_for_scoring_u8[:, x0:x1], use_otsu=use_otsu)
    t = normalize_28x28(arr28).to(device)
    if tta:
        probs = _predict_proba_tta(model, t, device, angles=(-6, 0, 6), temp=temp)
    else:
        logits = model(t)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pmax = float(probs.max())
    nll = -math.log(max(1e-6, pmax))
    ratio = w / float(H)
    # geometry prior pushes against very-wide merges and too-thin slivers
    geom_pen  = 0.55 * max(0.0, ratio - 0.95)
    geom_pen += 0.45 * max(0.0, ratio - 1.30)
    geom_pen += 0.45 * max(0.0, ratio - 1.70)
    geom_pen += 0.30 * max(0.0, 0.22 - ratio)
    return (nll + geom_pen), int(np.argmax(probs)), pmax

def _column_ink(arr_u8, col, band=1):
    a = max(0, col - band); b = min(arr_u8.shape[1], col + band + 1)
    return float(arr_u8[:, a:b].mean() / 255.0)

# ---------------- Boundary-aware DP decoder ----------------
def decode_best_partition_boundary(arr_for_decode, arr_u8, psc, model, device, cuts,
                                   max_span_factor=1.6, per_cut_bias=0.00,
                                   ink_penalty=0.12, boundary_reward=0.30,
                                   use_otsu=True, tta=True, temp=1.5, valley_band=1):
    """
    DP over candidate cuts with boundary reward (PSC) and ink penalty near cuts.
    """
    H, W = arr_for_decode.shape
    max_span = max(6, int(max_span_factor * H))
    K = len(cuts)
    dp = [9e9] * K
    bt = [-1] * K
    dp[0] = 0.0

    seg_cost = {}
    for j in range(1, K):
        xj = cuts[j]
        for i in range(0, j):
            xi = cuts[i]
            if (xj - xi) > max_span:
                continue
            key = (xi, xj)
            if key not in seg_cost:
                sc, _, _ = _segment_score(arr_u8, model, device, xi, xj,
                                          use_otsu=use_otsu, tta=tta, temp=temp)
                seg_cost[key] = sc
            # boundary terms at xi and xj
            cut_reward = boundary_reward * 0.5 * (psc[min(max(xi,0),W-1)] + psc[min(max(xj,0),W-1)])
            cut_ink = ink_penalty * 0.5 * (_column_ink(arr_u8, xi, valley_band) + _column_ink(arr_u8, xj, valley_band))
            cand = dp[i] + seg_cost[key] + per_cut_bias + cut_ink - cut_reward
            if cand < dp[j]:
                dp[j] = cand
                bt[j] = i

    spans = []
    j = K - 1
    if bt[j] == -1:
        return [(0, W)]
    while j > 0:
        i = bt[j]
        spans.append((cuts[i], cuts[j]))
        j = i
    spans.reverse()
    return spans

# ---------------- Seam fallback ----------------
def _seam_vertical_cut(arr_u8, x0, x1):
    """Minimal-cost vertical seam via DP; returns median seam column or None."""
    H, W = arr_u8.shape
    x0 = max(0, int(x0)); x1 = min(W, int(x1))
    if x1 - x0 < 6:  # too narrow
        return None
    band = arr_u8[:, x0:x1].astype(np.float32) / 255.0
    if _HAS_CV2:
        band = cv2.GaussianBlur(band, (3,3), 0)
    C = band.copy()
    back = np.zeros_like(C, dtype=np.int16)
    for r in range(1, H):
        prev = C[r-1]; cur = C[r]
        acc = np.full_like(cur, np.inf)
        from_idx = np.zeros_like(cur, dtype=np.int16)
        for dc in (-1, 0, 1):
            shift = np.roll(prev, dc)
            better = shift < acc
            acc[better] = shift[better]
            from_idx[better] = dc
        C[r] += acc
        back[r] = from_idx
    end_c = int(np.argmin(C[-1]))
    cs = [end_c]
    c = end_c
    for r in range(H-1, 0, -1):
        c = int(c + back[r, c])
        c = max(0, min(c, C.shape[1]-1))
        cs.append(c)
    seam_col = int(np.median(np.array(cs))) + x0
    return seam_col

# ---------------- High-level segmenter ----------------
@torch.no_grad()
def segment_by_psc_decode(arr_u8, model, device,
                          use_otsu=True,
                          seg_score_floor=0.18,
                          seg_nms_px=6,
                          seg_min_gap_px=6,
                          seg_max_span_factor=1.6,
                          per_cut_bias=0.00,
                          recur_max_depth=2,
                          recur_gain_thr=0.00,
                          boundary_reward=0.30,
                          ink_penalty=0.12,
                          tta_temp=1.5):
    """
    Returns: boxes [(x0,x1,y0,y1)] and normalized tensors [N,1,28,28]
    """
    H, W = arr_u8.shape

    # PSC profile + base peaks
    psc, mask_for_y, vp = _compute_psc_profile(arr_u8)
    base_peaks = [i for i in range(1, W-1)
                  if psc[i] > psc[i-1] and psc[i] > psc[i+1]
                  and psc[i] >= max(seg_score_floor, float(np.percentile(psc,60))*0.9)]

    # Dense candidate set
    cuts, peaks = _augment_cuts_dense(psc, vp, H, base_peaks,
                                      min_gap_px=seg_min_gap_px, jitter=1, topk_extra=24)

    # Shave valleys to help decoding choose boundaries
    arr_for_decode = valley_shave(arr_u8, peaks, band=2, factor=0.55)

    # DP with boundary-aware terms
    spans = decode_best_partition_boundary(
        arr_for_decode, arr_u8, psc, model, device, cuts,
        max_span_factor=seg_max_span_factor,
        per_cut_bias=per_cut_bias,
        ink_penalty=ink_penalty,
        boundary_reward=boundary_reward,
        use_otsu=use_otsu, tta=True, temp=tta_temp, valley_band=1
    )

    # optional refinement: recursively split very wide spans if beneficial
    def try_resegment(span, depth):
        x0, x1 = span
        if depth >= recur_max_depth:
            return [span]
        base_cost, _, _ = _segment_score(arr_u8, model, device, x0, x1, use_otsu=use_otsu, tta=True, temp=tta_temp)

        sub = arr_u8[:, x0:x1]
        sub_psc, sub_mask, sub_vp = _compute_psc_profile(sub)
        sub_base = [i for i in range(1, sub.shape[1]-1)
                    if sub_psc[i] > sub_psc[i-1] and sub_psc[i] > sub_psc[i+1]
                    and sub_psc[i] >= max(seg_score_floor*0.9, float(np.percentile(sub_psc,60))*0.85)]
        sub_cuts, sub_peaks = _augment_cuts_dense(sub_psc, sub_vp, H, sub_base,
                                                  min_gap_px=max(6, seg_min_gap_px-2),
                                                  jitter=1, topk_extra=16)
        sub_decode_img = valley_shave(sub, sub_peaks, band=2, factor=0.55)
        sub_spans = decode_best_partition_boundary(
            sub_decode_img, sub, sub_psc, model, device, sub_cuts,
            max_span_factor=max(1.4, seg_max_span_factor*0.9),
            per_cut_bias=per_cut_bias,
            ink_penalty=ink_penalty,
            boundary_reward=boundary_reward,
            use_otsu=use_otsu, tta=True, temp=tta_temp, valley_band=1
        )
        if len(sub_spans) <= 1:
            # seam fallback
            cut = _seam_vertical_cut(sub, 0, sub.shape[1])
            if cut is None:
                return [span]
            left_c  = (x0, x0 + cut); right_c = (x0 + cut, x1)
            l_cost, _, _ = _segment_score(arr_u8, model, device, *left_c, use_otsu=use_otsu, tta=True, temp=tta_temp)
            r_cost, _, _ = _segment_score(arr_u8, model, device, *right_c, use_otsu=use_otsu, tta=True, temp=tta_temp)
            gain = base_cost - (l_cost + r_cost)
            if gain >= recur_gain_thr:
                out = []
                out.extend(try_resegment(left_c,  depth+1))
                out.extend(try_resegment(right_c, depth+1))
                return out
            return [span]

        # gain check for multi-split
        sub_cost = 0.0
        for a,b in sub_spans:
            c,_,_ = _segment_score(arr_u8, model, device, x0+a, x0+b, use_otsu=use_otsu, tta=True, temp=tta_temp)
            sub_cost += c
        gain = base_cost - sub_cost
        if gain >= recur_gain_thr:
            out = []
            for a,b in sub_spans:
                out.extend(try_resegment((x0+a, x0+b), depth+1))
            return out
        return [span]

    refined_spans = []
    for x0,x1 in spans:
        if (x1 - x0) > int(0.95 * H):  # suspiciously wide
            refined_spans.extend(try_resegment((x0,x1), 0))
        else:
            refined_spans.append((x0,x1))

    # boxes + tensors (vertical refine with mask)
    mask_y = _otsu_or_thresh_u8(arr_u8, thresh=10, open_iters=0, close_iters=1)
    boxes, tensors = [], []
    for x0,x1 in refined_spans:
        rs = (mask_y[:, x0:x1] > 0).sum(axis=1)
        ny = np.where(rs > 0)[0]
        if ny.size == 0:
            continue
        y0 = int(ny[0]); y1 = int(ny[-1]) + 1
        boxes.append((x0, x1, y0, y1))
        arr28 = mnist_style_28x28(arr_u8[y0:y1, x0:x1], use_otsu=use_otsu)
        tensors.append(normalize_28x28(arr28))

    # sort L->R
    order = np.argsort([b[0] for b in boxes])
    boxes  = [boxes[i]  for i in order]
    tensors= [tensors[i] for i in order]
    return boxes, tensors

# ---------------- Utilities ----------------
def load_model(weights, device):
    model = DigitCNN().to(device)
    state = torch.load(weights, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

@torch.no_grad()
def eval_classifier_on_mnist(model, device, outdir):
    loader = get_test_loader()
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(1).cpu().numpy().tolist()
        y_pred += pred
        y_true += y.numpy().tolist()
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = (y_true == y_pred).mean()

    if _HAS_SK:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
        fig, ax = plt.subplots(1,2, figsize=(12,4))
        im = ax[0].imshow(cm, interpolation='nearest', cmap='Blues')
        ax[0].set_title('Confusion Matrix (counts)')
        plt.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
        ax[0].set_xlabel('Pred'); ax[0].set_ylabel('True')
        ax[0].set_xticks(range(10)); ax[0].set_yticks(range(10))

        cmn = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im2 = ax[1].imshow(cmn, interpolation='nearest', cmap='Greens', vmin=0, vmax=1)
        ax[1].set_title('Confusion Matrix (row-normalized)')
        plt.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)
        ax[1].set_xlabel('Pred'); ax[1].set_ylabel('True')
        ax[1].set_xticks(range(10)); ax[1].set_yticks(range(10))
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'confusion_matrix.png'), dpi=160)
        plt.close(fig)

        per_class_acc = np.diag(cmn)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(range(10), per_class_acc)
        ax.set_ylim(0,1)
        ax.set_xticks(range(10))
        ax.set_title('Per-class Accuracy (MNIST test)')
        ax.set_xlabel('Digit'); ax.set_ylabel('Accuracy')
        for i,v in enumerate(per_class_acc):
            ax.text(i, v+0.02, f'{v*100:.1f}%', ha='center', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'per_class_accuracy.png'), dpi=160)
        plt.close(fig)

        rep = classification_report(y_true, y_pred, digits=3)
        with open(os.path.join(outdir, 'mnist_test_report.txt'), 'w') as f:
            f.write(rep)

    return float(acc)

# ---- Segmentation quality helpers (IoU on x-spans) ----
def iou1d(span_a, span_b):
    a0, a1 = span_a; b0, b1 = span_b
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - inter
    return inter / union if union > 0 else 0.0

def x_spans_from_boxes(boxes):
    return [(x0, x1) for (x0, x1, _, _) in boxes]

def match_by_center(true_spans, pred_spans):
    """Greedy nearest-center matching along x; returns [(ti, pj, iou), ...]."""
    if not true_spans or not pred_spans:
        return []
    t_cent = np.array([(a+b)/2.0 for (a,b) in true_spans])
    p_cent = np.array([(a+b)/2.0 for (a,b) in pred_spans])
    used_p = set()
    matches = []
    for ti, tc in enumerate(t_cent):
        diffs = np.abs(p_cent - tc)
        order = np.argsort(diffs)
        for pj in order:
            if pj not in used_p:
                used_p.add(pj)
                matches.append((ti, pj, iou1d(true_spans[ti], pred_spans[pj])))
                break
    return matches

# ---------------- End-to-end evaluation ----------------
@torch.no_grad()
def eval_end_to_end(model, device, n_samples=300, num_digits_range=(3,4), overlap=(1,3),
                    spacing=4, outdir='./report_assets',
                    seg_params=None, iou_thresh=0.30, debug_first=12, use_shear=True):
    raw = get_raw_testset()

    seq_correct = 0
    total = 0
    digit_correct = 0
    digit_total = 0
    matched_samples = 0
    oversplits = 0
    undersplits = 0

    seg_prec_sum = 0.0
    seg_rec_sum  = 0.0
    seg_iou_sum  = 0.0
    seg_samples  = 0

    os.makedirs(outdir, exist_ok=True)

    for i in range(n_samples):
        k = np.random.randint(num_digits_range[0], num_digits_range[1]+1)
        canvas, labels, true_spans = create_segments(
            raw, num_digits=k, overlap_range=overlap, spacing=spacing, blend='max'
        )
        arr_u8 = (canvas*255).astype(np.uint8)

        # optional shear (helps deepen valleys)
        arr_proc = arr_u8
        if use_shear:
            arr_proc, _ = _shear_min_cov(arr_u8, max_abs=0.45)

        # Segment
        boxes, tensors = segment_by_psc_decode(
            arr_proc, model, device,
            use_otsu=seg_params['use_otsu'],
            seg_score_floor=seg_params['seg_score_floor'],
            seg_nms_px=seg_params['seg_nms_px'],
            seg_min_gap_px=seg_params['seg_min_gap_px'],
            seg_max_span_factor=seg_params['seg_max_span_factor'],
            per_cut_bias=seg_params['per_cut_bias'],
            recur_max_depth=seg_params['recur_max_depth'],
            recur_gain_thr=seg_params['recur_gain_thr'],
            boundary_reward=seg_params['boundary_reward'],
            ink_penalty=seg_params['ink_penalty'],
            tta_temp=seg_params['tta_temp']
        )

        # --- Segmentation metrics
        pred_spans = x_spans_from_boxes(boxes)
        matches = match_by_center(true_spans, pred_spans)
        TP = sum(1 for *_x, iou in matches if iou >= iou_thresh)
        prec = TP / max(1, len(pred_spans))
        rec  = TP / max(1, len(true_spans))
        mean_iou = 0.0 if len(matches)==0 else (sum(i for *_, i in matches) / len(matches))
        seg_prec_sum += prec
        seg_rec_sum  += rec
        seg_iou_sum  += mean_iou
        seg_samples  += 1

        # Debug figs (first N)
        if i < debug_first:
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.imshow(arr_proc, cmap='gray')
            for (x0,x1,y0,y1) in boxes:
                ax.add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor='red', linewidth=2))
            ax.set_title(f"true={''.join(map(str,labels))} #boxes={len(boxes)} prec={prec:.2f} rec={rec:.2f}")
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f'dbg_{i:03d}.png'), dpi=160)
            plt.close(fig)

        # Recognition metrics
        if len(tensors) == 0:
            undersplits += 1
            total += 1
            continue

        batch = torch.cat(tensors, dim=0).to(device)  # [N,1,28,28]
        logits = model(batch)
        preds = logits.argmax(1).cpu().numpy().tolist()

        pred_seq = ''.join(str(p) for p in preds)
        true_seq = ''.join(str(x) for x in labels)
        if pred_seq == true_seq:
            seq_correct += 1

        if len(preds) > len(labels):
            oversplits += 1
        elif len(preds) < len(labels):
            undersplits += 1

        if len(preds) == len(labels):
            matched_samples += 1
            digit_total += len(labels)
            digit_correct += sum(int(p == t) for p, t in zip(preds, labels))

        total += 1

    seq_acc = seq_correct / max(1,total)
    dig_acc = digit_correct / max(1,digit_total)
    over_rate = oversplits/max(1,total)
    under_rate = undersplits/max(1,total)

    seg_prec = seg_prec_sum/max(1,seg_samples)
    seg_rec  = seg_rec_sum /max(1,seg_samples)
    seg_iou  = seg_iou_sum /max(1,seg_samples)

    # Plot summary
    fig, ax = plt.subplots(figsize=(8,4))
    bars = ['Seq Acc', 'Digit Acc (matched)', 'Over-split', 'Under-split',
            'Segm Prec', 'Segm Rec', 'Segm IoU(match)']
    vals = [seq_acc, dig_acc, over_rate, under_rate, seg_prec, seg_rec, seg_iou]
    ax.bar(bars, vals)
    ax.set_ylim(0,1)
    for i,v in enumerate(vals):
        ax.text(i, v+0.02, f'{v*100:.1f}%', ha='center', fontsize=9)
    ax.set_title(f'End-to-End (N={total}, digits {num_digits_range[0]}–{num_digits_range[1]}, '
                 f'overlap {overlap[0]}–{overlap[1]}, spacing={spacing})')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'end_to_end_summary.png'), dpi=160)
    plt.close(fig)

    return {
        'sequence_accuracy': float(seq_acc),
        'digit_accuracy_matched': float(dig_acc),
        'over_split_rate': float(over_rate),
        'under_split_rate': float(under_rate),
        'samples': int(total),
        'matched_samples': int(matched_samples),
        'segmentation_precision_avg': float(seg_prec),
        'segmentation_recall_avg': float(seg_rec),
        'segmentation_iou_avg': float(seg_iou)
    }

# ---------------- Main ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=str, default='mnist_cnn.pth')
    p.add_argument('--samples', type=int, default=300)
    p.add_argument('--digits_min', type=int, default=3)
    p.add_argument('--digits_max', type=int, default=4)
    p.add_argument('--overlap_min', type=int, default=1)
    p.add_argument('--overlap_max', type=int, default=3)
    p.add_argument('--spacing', type=int, default=4)
    p.add_argument('--outdir', type=str, default='./report_assets')
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--use_shear', action='store_true', help='apply shear deskew (recommended)')

    # Segmentation knobs (defaults biased to reduce undersplit)
    p.add_argument('--seg_score_floor', type=float, default=0.18)
    p.add_argument('--seg_nms_px', type=int, default=6)
    p.add_argument('--seg_min_gap_px', type=int, default=6)
    p.add_argument('--seg_max_span_factor', type=float, default=1.6)
    p.add_argument('--per_cut_bias', type=float, default=0.00)

    # Refinement
    p.add_argument('--recur_max_depth', type=int, default=2)
    p.add_argument('--recur_gain_thr', type=float, default=0.00)

    # Boundary-aware decode weights
    p.add_argument('--boundary_reward', type=float, default=0.30)
    p.add_argument('--ink_penalty', type=float, default=0.12)
    p.add_argument('--tta_temp', type=float, default=1.5)

    # Misc
    p.add_argument('--iou_thresh', type=float, default=0.30, help='IoU threshold to count a segment TP')
    p.add_argument('--debug_first', type=int, default=12, help='save first N canvases with boxes')

    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.weights, device)

    print('== Evaluating classifier on MNIST test set ==')
    test_acc = eval_classifier_on_mnist(model, device, args.outdir)
    print(f'MNIST test accuracy: {test_acc*100:.2f}%')

    print('== Evaluating end-to-end pipeline on synthetic overlapped canvases ==')
    seg_params = dict(
        use_otsu=True,
        seg_score_floor=args.seg_score_floor,
        seg_nms_px=args.seg_nms_px,
        seg_min_gap_px=args.seg_min_gap_px,
        seg_max_span_factor=args.seg_max_span_factor,
        per_cut_bias=args.per_cut_bias,
        recur_max_depth=args.recur_max_depth,
        recur_gain_thr=args.recur_gain_thr,
        boundary_reward=args.boundary_reward,
        ink_penalty=args.ink_penalty,
        tta_temp=args.tta_temp
    )
    metrics = eval_end_to_end(
        model, device,
        n_samples=args.samples,
        num_digits_range=(args.digits_min, args.digits_max),
        overlap=(args.overlap_min, args.overlap_max),
        spacing=args.spacing,
        outdir=args.outdir,
        seg_params=seg_params,
        iou_thresh=args.iou_thresh,
        debug_first=args.debug_first,
        use_shear=args.use_shear
    )

    print('End-to-End metrics:')
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f'  {k}: {v*100:.2f}%')
        else:
            print(f'  {k}: {v}')

    # Markdown summary
    md = []
    md.append('# Evaluation Summary')
    md.append(f'- MNIST test accuracy: **{test_acc*100:.2f}%**')
    md.append(f'- Sequence accuracy: **{metrics["sequence_accuracy"]*100:.2f}%**')
    md.append(f'- Digit accuracy (matched lengths): **{metrics["digit_accuracy_matched"]*100:.2f}%**')
    md.append(f'- Matched-length samples: **{metrics["matched_samples"]}** / {metrics["samples"]}')
    md.append(f'- Over-split rate: **{metrics["over_split_rate"]*100:.2f}%**')
    md.append(f'- Under-split rate: **{metrics["under_split_rate"]*100:.2f}%**')
    md.append(f'- Segmentation precision (avg): **{metrics["segmentation_precision_avg"]*100:.2f}%**')
    md.append(f'- Segmentation recall (avg): **{metrics["segmentation_recall_avg"]*100:.2f}%**')
    md.append(f'- Segmentation IoU@match (avg): **{metrics["segmentation_iou_avg"]*100:.2f}%**')
    md.append('')
    if os.path.exists(os.path.join(args.outdir, 'confusion_matrix.png')):
        md.append('![Confusion Matrix](confusion_matrix.png)')
        md.append('')
        md.append('![Per-class Accuracy](per_class_accuracy.png)')
        md.append('')
    md.append('![End-to-End Summary](end_to_end_summary.png)')
    with open(os.path.join(args.outdir, 'EVAL_SUMMARY.md'), 'w') as f:
        f.write('\n'.join(md))

    print(f'\nSaved figures + EVAL_SUMMARY.md to: {os.path.abspath(args.outdir)}')

if __name__ == '__main__':
    main()
