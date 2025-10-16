# draw_gui_mnist_multidigit.py
# ----------------------------
# Draw multiple digits, segment them with vertical projection valleys,
# and classify each with your MNIST CNN.

import os
import numpy as np
from PIL import Image, ImageDraw, ImageTk

import tkinter as tk
from tkinter import filedialog, messagebox

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---------- Optional deps ----------
_HAS_CV2 = False
_HAS_SCIPY = False
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

# ---------------- CNN (same as your training script) ----------------
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

    def forward(self, x):  # x: [B,1,28,28]
        return self.classifier(self.features(x))

MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)

def load_model(device, weights="mnist_cnn.pth"):
    model = DigitCNN().to(device)
    if os.path.exists(weights):
        try:
            state = torch.load(weights, map_location=device)
            model.load_state_dict(state)
            print(f"[OK] Loaded weights: {weights}")
        except Exception as e:
            print(f"[WARN] Failed to load weights: {e}\nUsing untrained model.")
    else:
        print("[WARN] No weights file found. Using untrained model.")
    model.eval()
    return model

# ==== Robust MNIST-style preprocessing and segmentation ====
def mnist_style_28x28(arr_u8, use_otsu=True, bin_thresh=10, cm_center=True):
    """
    arr_u8: uint8 image (H,W) white-on-black.
    Returns (28,28) float in [0,1] BEFORE normalization.
    """
    assert arr_u8.ndim == 2
    img = arr_u8

    # 1) Binarize to find tight bbox
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

    # 2) Keep aspect, resize longest side to 20 px
    h, w = crop.shape
    if h >= w:
        new_h, new_w = 20, max(1, int(round(20 * w / h)))
    else:
        new_w, new_h = 20, max(1, int(round(20 * h / w)))
    crop_small = Image.fromarray(crop, mode='L').resize((new_w, new_h), Image.LANCZOS)
    small = np.asarray(crop_small, dtype=np.uint8)

    # 3) Paste into 28×28 canvas, centered roughly
    canvas = np.zeros((28, 28), dtype=np.uint8)
    y_off = (28 - new_h) // 2
    x_off = (28 - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = small

    # 4) Refine center: shift by center-of-mass so the ink sits in the middle
    if cm_center and _HAS_SCIPY:
        m = (canvas > 0).astype(np.float32)
        if m.sum() > 0:
            cy, cx = center_of_mass(m)
            dy = (28/2 - 0.5) - cy
            dx = (28/2 - 0.5) - cx
            canvas = ndi_shift(canvas.astype(np.float32), shift=(dy, dx), order=1,
                               mode='constant', cval=0.0).clip(0, 255).astype(np.uint8)

    # 5) Return float [0,1]; normalization to MNIST happens separately
    return (canvas.astype(np.float32) / 255.0)

def normalize_28x28(arr28_float):
    """(28,28) float [0..1] -> torch [1,1,28,28] normalized like MNIST."""
    t = torch.from_numpy(arr28_float).unsqueeze(0).unsqueeze(0).float()
    return (t - MNIST_MEAN[0]) / MNIST_STD[0]


# ---------- Deskew helpers ----------
def _estimate_skew_angle_deg(arr_u8, bin_thresh=10, max_abs_deg=25):
    """
    Estimate dominant line angle via PCA on foreground pixels.
    Returns angle in degrees; rotate by -angle to deskew.
    """
    mask = (arr_u8 > bin_thresh).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if xs.size < 50:
        return 0.0  # nothing drawn, or too little ink
    x = xs.astype(np.float32)
    y = ys.astype(np.float32)
    x -= x.mean()
    y -= y.mean()
    # 2x2 covariance and eigenvectors
    cov = np.cov(np.vstack([x, y]))
    vals, vecs = np.linalg.eig(cov)
    v = vecs[:, np.argmax(vals)]  # principal direction
    angle = math.degrees(math.atan2(v[1], v[0]))  # in [-180,180]
    # clamp to reasonable deskew range
    if angle > 90: angle -= 180
    if angle < -90: angle += 180
    angle = max(-max_abs_deg, min(max_abs_deg, angle))
    return float(angle)

def _rotate_u8(arr_u8, angle_deg):
    """Rotate around center without expanding canvas; black fill."""
    H, W = arr_u8.shape
    try:
        import cv2
        M = cv2.getRotationMatrix2D(((W-1)/2.0, (H-1)/2.0), angle_deg, 1.0)
        return cv2.warpAffine(arr_u8, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    except Exception:
        from PIL import Image
        pil = Image.fromarray(arr_u8, mode='L')
        # PIL rotates counter-clockwise; expand=False keeps size
        return np.asarray(pil.rotate(angle_deg, resample=Image.BILINEAR, expand=False, fillcolor=0), dtype=np.uint8)

def _map_box_back_to_original(box, angle_deg, W, H):
    """
    Map an axis-aligned box from the rotated image back to the original coords.
    We rotated the canvas by -angle for processing, so map back by +angle.
    """
    x0, x1, y0, y1 = box
    cx, cy = (W-1)/2.0, (H-1)/2.0
    theta = math.radians(+angle_deg)  # inverse rotation
    ct, st = math.cos(theta), math.sin(theta)

    def inv_rot(x, y):
        # rotate point (x,y) by +angle around center
        dx, dy = x - cx, y - cy
        xr =  ct*dx - st*dy + cx
        yr =  st*dx + ct*dy + cy
        return xr, yr

    corners = [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]
    pts = [inv_rot(x,y) for (x,y) in corners]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xa, xb = int(max(0, min(xs))), int(min(W, max(xs)))
    ya, yb = int(max(0, min(ys))), int(min(H, max(ys)))
    if xb <= xa or yb <= ya:
        return None
    return (xa, xb, ya, yb)

# ---------- (Optional) Test-Time Augmentation for a single 28x28 tensor ----------
@torch.no_grad()
def predict_with_tta_single(model, t_norm, device, angles_deg=(-8,-4,0,4,8)):
    """
    t_norm: [1,1,28,28] normalized tensor
    Returns: probs averaged over rotations, shape [10]
    """
    x = t_norm.detach().cpu()
    # de-normalize to 0..1 float
    x01 = x * MNIST_STD[0] + MNIST_MEAN[0]  # [1,1,28,28]
    img = (x01.squeeze(0).squeeze(0).numpy().clip(0,1) * 255).astype(np.uint8)  # (28,28)
    imgs = []
    for a in angles_deg:
        pil = Image.fromarray(img, mode='L').rotate(a, resample=Image.BILINEAR, expand=False, fillcolor=0)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
        t = (t - MNIST_MEAN[0]) / MNIST_STD[0]
        imgs.append(t)
    batch = torch.cat(imgs, dim=0).to(device)  # [K,1,28,28]
    logits = model(batch)
    probs = torch.softmax(logits, dim=1).mean(dim=0).cpu().numpy()  # [10]
    return probs

def segment_by_projection(arr_u8,
                          use_otsu=True,
                          bin_thresh=10,
                          smooth_sigma=2.5,   # more smoothing reduces false splits
                          prom_frac=0.18,     # valley strength relative to signal range
                          min_gap_px=10,      # min distance between cuts
                          min_width_px=12,    # drop very thin slices
                          min_area_px=200,    # drop tiny blobs
                          post_merge=True,
                          post_split=True):
    """
    Returns: boxes [(x0,x1,y0,y1)] and tensors [N,1,28,28] normalized.
    """
    H, W = arr_u8.shape

    # 0) Light morphology to clean strokes
    if _HAS_CV2:
        if use_otsu:
            _, mask = cv2.threshold(arr_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            mask = ((arr_u8 > bin_thresh) * 255).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
    else:
        mask = ((arr_u8 > bin_thresh) * 255).astype(np.uint8)

    col_sum = (mask > 0).sum(axis=0).astype(np.float32)
    if col_sum.sum() == 0:
        return [], []

    # 1) Smooth projection
    if _HAS_SCIPY:
        vp = gaussian_filter1d(col_sum, sigma=smooth_sigma)
    else:
        k = max(3, int(2 * smooth_sigma) | 1)
        vp = np.convolve(col_sum, np.ones(k, dtype=np.float32)/k, mode='same')

    # 2) Find valleys as peaks on inverted VP
    inv = vp.max() - vp
    rng = float(vp.max() - vp.min())
    min_prom = prom_frac * max(rng, 1.0)

    # simple peak picking with min-gap & prominence
    peaks = []
    for i in range(1, W-1):
        if inv[i] > inv[i-1] and inv[i] > inv[i+1]:
            peaks.append(i)
    peaks = sorted(peaks, key=lambda i: inv[i], reverse=True)

    good = []
    used = np.zeros(W, dtype=bool)
    for p in peaks:
        if used[max(0, p-min_gap_px):min(W, p+min_gap_px+1)].any():
            continue
        L = max(0, p-12); R = min(W, p+13)
        left_peak  = vp[L:p+1].max() if p > L else vp[p]
        right_peak = vp[p:R].max() if R > p else vp[p]
        drop = 0.5*(left_peak + right_peak) - vp[p]
        if drop >= min_prom:
            good.append(p)
            used[max(0, p-min_gap_px):min(W, p+min_gap_px+1)] = True
    good = sorted(good)

    cuts = [0] + good + [W]

    # 3) Build boxes with vertical refinement
    boxes = []
    for a,b in zip(cuts[:-1], cuts[1:]):
        if b - a < min_width_px:
            continue
        cs = (mask[:, a:b] > 0).sum(axis=0)
        nz = np.where(cs > 0)[0]
        if nz.size == 0:
            continue
        aa = a + int(nz[0]); bb = a + int(nz[-1]) + 1
        if bb - aa < min_width_px:
            continue
        rs = (mask[:, aa:bb] > 0).sum(axis=1)
        ny = np.where(rs > 0)[0]
        if ny.size == 0:
            continue
        y0 = int(ny[0]); y1 = int(ny[-1]) + 1
        area = int((mask[y0:y1, aa:bb] > 0).sum())
        if area < min_area_px:
            continue
        boxes.append((aa, bb, y0, y1))

    # 4) Post-merge overly weak boundaries or skinny neighbors
    def boundary_drop(idx):
        x0a,x1a,_,_ = boxes[idx]
        x0b,x1b,_,_ = boxes[idx+1]
        p = x1a
        L = max(0, p-12); R = min(W, p+13)
        valley = vp[p] if 0 <= p < W else vp[min(max(p,0),W-1)]
        left_peak  = vp[L:p+1].max() if p > L else vp[p]
        right_peak = vp[p:R].max() if R > p else vp[p]
        return 0.5*(left_peak+right_peak) - valley  # larger = stronger boundary

    if post_merge and len(boxes) >= 2:
        changed = True
        while changed:
            changed = False
            weakest = None
            weakest_score = 1e9
            for i in range(len(boxes)-1):
                wL = boxes[i][1] - boxes[i][0]
                wR = boxes[i+1][1] - boxes[i+1][0]
                score = boundary_drop(i)
                if wL < 10 or wR < 10 or score < (0.15 * rng):
                    if score < weakest_score:
                        weakest_score = score
                        weakest = i
            if weakest is not None:
                a = boxes[weakest]
                b = boxes[weakest+1]
                merged = (min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3]))
                boxes = boxes[:weakest] + [merged] + boxes[weakest+2:]
                changed = True

    # 5) Post-split very wide boxes
    if post_split:
        refined = []
        for (x0,x1,y0,y1) in boxes:
            w = x1 - x0
            if w > int(0.95 * (y1 - y0)):
                sub = arr_u8[y0:y1, x0:x1]
                sub_boxes, _ = segment_by_projection(
                    sub, use_otsu=use_otsu, bin_thresh=bin_thresh,
                    smooth_sigma=smooth_sigma, prom_frac=prom_frac,
                    min_gap_px=max(6, min_gap_px-2),
                    min_width_px=max(10, min_width_px-2),
                    min_area_px=max(150, min_area_px-50),
                    post_merge=True, post_split=False  # avoid recursion
                )
                if len(sub_boxes) >= 2:
                    for (a0,a1,b0,b1) in sub_boxes:
                        refined.append((x0+a0, x0+a1, y0+b0, y0+b1))
                    continue
            refined.append((x0,x1,y0,y1))
        boxes = refined

    # 6) Produce normalized tensors for the model (left->right)
    tensors = []
    for (x0,x1,y0,y1) in boxes:
        arr28 = mnist_style_28x28(arr_u8[y0:y1, x0:x1], use_otsu=use_otsu)
        tensors.append(normalize_28x28(arr28))
    order = np.argsort([b[0] for b in boxes])
    boxes  = [boxes[i]  for i in order]
    tensors= [tensors[i]for i in order]
    return boxes, tensors

# ---------------- Tkinter GUI ----------------
class MultiDigitGUI:
    def __init__(self, root, device, weights_path="mnist_cnn.pth"):
        self.root = root
        self.device = device
        self.model = load_model(device, weights_path)

        self.CANVAS = 280
        self.BRUSH  = 18

        root.title("Multi-digit MNIST Predictor")
        self.canvas = tk.Canvas(root, width=self.CANVAS, height=self.CANVAS,
                                bg="black", cursor="tcross",
                                highlightthickness=1, highlightbackground="#444")
        self.canvas.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

        self.img = Image.new("L", (self.CANVAS, self.CANVAS), color=0)
        self.draw = ImageDraw.Draw(self.img)

        tk.Button(root, text="Predict", width=12, command=self.on_predict).grid(row=1, column=0, padx=6, pady=6, sticky="ew")
        tk.Button(root, text="Clear",   width=12, command=self.on_clear).grid(row=1, column=1, padx=6, pady=6, sticky="ew")
        tk.Button(root, text="Save PNG",width=12, command=self.on_save).grid(row=1, column=2, padx=6, pady=6, sticky="ew")
        tk.Button(root, text="Undo Box",width=12, command=self.on_clear_boxes).grid(row=1, column=3, padx=6, pady=6, sticky="ew")

        self.msg = tk.StringVar(value="Draw multiple digits (e.g., 123), then click Predict")
        tk.Label(root, textvariable=self.msg, font=("Arial", 12)).grid(row=2, column=0, columnspan=4, padx=10, pady=(0,8))

        self.last_x = None
        self.last_y = None
        self.canvas.bind("<ButtonPress-1>", self.on_down)
        self.canvas.bind("<B1-Motion>",     self.on_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_up)

    # Drawing
    def on_down(self, e):
        self.last_x, self.last_y = e.x, e.y
        self._dot(e.x, e.y)

    def on_move(self, e):
        if self.last_x is not None:
            self.canvas.create_line(self.last_x, self.last_y, e.x, e.y,
                                    fill="white", width=self.BRUSH, capstyle=tk.ROUND, smooth=True)
            self.draw.line([self.last_x, self.last_y, e.x, e.y],
                           fill=255, width=self.BRUSH, joint="curve")
        self.last_x, self.last_y = e.x, e.y

    def on_up(self, _):
        self.last_x, self.last_y = None, None

    def _dot(self, x, y):
        r = self.BRUSH // 2
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
        self.draw.ellipse((x-r, y-r, x+r, y+r), fill=255)

    # Buttons
    def on_clear(self):
        self.canvas.delete("all")
        self.draw.rectangle((0,0,self.CANVAS,self.CANVAS), fill=0)
        self.msg.set("Draw multiple digits, then click Predict")

    def on_clear_boxes(self):
        self.canvas.delete("bbox")

    def on_save(self):
        f = filedialog.asksaveasfilename(defaultextension=".png",
                                         filetypes=[("PNG","*.png")],
                                         initialfile="digits.png")
        if f:
            self.img.save(f)
            messagebox.showinfo("Saved", f"Saved drawing to:\n{f}")

    @torch.no_grad()
    def on_predict(self):
        arr = np.asarray(self.img, dtype=np.uint8)  # (H,W)

        # 0) Global deskew (rotate by -angle to make the line horizontal)
        angle = _estimate_skew_angle_deg(arr, bin_thresh=10, max_abs_deg=25)
        arr_rot = _rotate_u8(arr, -angle)  # rotate canvas by -angle

        # 1) Segment on the deskewed canvas
        boxes_rot, tensors = segment_by_projection(
            arr_rot,
            use_otsu=True,
            bin_thresh=10,
            smooth_sigma=2.8,     # a bit higher smoothing helps when strokes were angled
            prom_frac=0.20,       # slightly stricter to avoid false splits on slants
            min_gap_px=10,
            min_width_px=12,
            min_area_px=200,
            post_merge=True,
            post_split=True
        )

        if not boxes_rot:
            self.msg.set("No digits detected — try thicker/straighter strokes.")
            self.on_clear_boxes()
            return

        # 2) Predict each box (with optional TTA)
        preds = []
        probs_list = []
        for t in tensors:
            # EITHER plain prediction:
            # logits = self.model(t.to(self.device))
            # p = torch.softmax(logits, dim=1)[0].cpu().numpy()
            # OR TTA-averaged prediction (more robust to small angle errors):
            p = predict_with_tta_single(self.model, t.to(self.device), self.device, angles_deg=(-10,-5,0,5,10))
            preds.append(int(p.argmax()))
            probs_list.append(float(p.max()))

        # 3) Map boxes back to original canvas coords and draw them
        self.on_clear_boxes()
        H, W = arr.shape
        for (x0,x1,y0,y1) in boxes_rot:
            bb = _map_box_back_to_original((x0,x1,y0,y1), angle_deg=-angle, W=W, H=H)
            # Note: We rotated by -angle for processing; inverse mapping uses +(-angle) = -angle.
            # If you prefer, pass angle_deg=+angle and change arr_rot rotation direction consistently.
            if bb is None:
                continue
            xa, xb, ya, yb = bb
            self.canvas.create_rectangle(xa, ya, xb, yb, outline="red", width=2, tags="bbox")

        seq = "".join(str(d) for d in preds)
        self.msg.set(f"Sequence: {seq}  (N={len(preds)})  angle≈{angle:+.1f}°")
        print("Per-digit predictions:", preds)
        print("Per-digit confidences:", [f"{c:.2f}" for c in probs_list])
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = tk.Tk()
    gui  = MultiDigitGUI(root, device, weights_path="mnist_cnn.pth")
    root.resizable(False, False)
    root.mainloop()

if __name__ == "__main__":
    main()
