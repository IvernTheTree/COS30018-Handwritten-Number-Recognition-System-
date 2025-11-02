# gui_acquisition_expr16.py
# Acquisition / Segmentation / Evaluation GUI for a 16-class (digits + operators) CNN.
# This version:
#   • NO photo pre-processing
#   • Scale-aware seg thresholds (optional)
#   • Drawing Pad (freehand canvas)
#   • Touching-digits synthesizer
#   • NEW: Safe expression evaluation after prediction (Value: ...)

import os, re, glob, argparse, random, math, ast
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageOps, ImageDraw

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.nn as nn

# -------------------- Segmentation pipeline (your report module) --------------------
from report_metric_hybrid import segment_by_psc_decode

try:
    from report_metric_hybrid import _shear_min_cov as shear_fn
    HAS_SHEAR = True
except Exception:
    HAS_SHEAR = False

# -------------------- Utils --------------------
def to_u8_gray(pil_img: Image.Image) -> np.ndarray:
    return np.asarray(pil_img.convert("L"), dtype=np.uint8)

def auto_invert_if_needed(arr_u8: np.ndarray) -> np.ndarray:
    return (255 - arr_u8) if float(arr_u8.mean()) > 128 else arr_u8

def _safe_resize_28(img_u8: np.ndarray) -> np.ndarray:
    if img_u8.shape != (28,28):
        return np.asarray(Image.fromarray(img_u8, mode="L").resize((28,28), Image.BILINEAR), dtype=np.uint8)
    return img_u8

def _make_param_subdir(root_outdir, len_min, len_max, ov_min, ov_max, spacing, N, tag=""):
    from datetime import datetime
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{tag}len{len_min}-{len_max}_ov{ov_min}-{ov_max}_sp{spacing}_N{N}_{stamp}"
    outdir = os.path.join(root_outdir, name)
    if not os.path.exists(outdir):
        return outdir
    k = 1
    while True:
        cand = f"{outdir}-{k}"
        if not os.path.exists(cand):
            return cand
        k += 1

# -------------------- 16-class classifier --------------------
class FlexibleDigitCNN(nn.Module):
    def __init__(self, num_classes: int = 16):
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
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# -------------------- Non-silent checkpoint loader --------------------
def _extract_state_dict(obj):
    if hasattr(obj, "state_dict") and callable(getattr(obj, "state_dict")):
        try: return obj.state_dict()
        except Exception: pass
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "weights"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        for key in ("model", "net"):
            if key in obj:
                inner = obj[key]
                if isinstance(inner, dict): return inner
                if hasattr(inner, "state_dict") and callable(getattr(inner, "state_dict")):
                    return inner.state_dict()
    return obj

def _strip_known_prefixes(sd: dict) -> dict:
    prefixes = ("module.", "model.", "net.", "cnn.", "backbone.")
    out = {}
    for k, v in sd.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p): nk = nk[len(p):]
        out[nk] = v
    return out

def _find_head_outdim(sd: dict):
    candidates = []
    for k, v in sd.items():
        if hasattr(v, "ndim") and int(getattr(v, "ndim", 0)) == 2:
            candidates.append((k, int(v.shape[0]), int(v.shape[1])))
    for pref in (16, 10):
        for k, out_f, _ in candidates:
            if out_f == pref: return k, out_f
    for k, out_f, in_f in reversed(candidates):
        if in_f == 128: return k, out_f
    if candidates:
        k, out_f, _ = candidates[-1]; return k, out_f
    return None, None

def _compare_key_shapes(model: nn.Module, sd: dict):
    msd = model.state_dict(); missing, mism = [], []
    for k in msd.keys():
        if k not in sd: missing.append(k)
        elif tuple(msd[k].shape) != tuple(sd[k].shape):
            mism.append(f"{k}: checkpoint {tuple(sd[k].shape)} vs model {tuple(msd[k].shape)}")
    return missing, mism

def load_model_expr16_nonsilent(weights_path: str, device: torch.device) -> nn.Module:
    chk = torch.load(weights_path, map_location=device)
    sd_raw = _extract_state_dict(chk)
    if not isinstance(sd_raw, dict):
        raise RuntimeError("Checkpoint does not contain a state_dict dictionary.")
    sd = _strip_known_prefixes(sd_raw)
    head_key, out_dim = _find_head_outdim(sd)
    if head_key is None: raise RuntimeError("Cannot locate final classifier weights in checkpoint.")
    if int(out_dim) != 16:
        raise RuntimeError(f"Requires 16-class checkpoint; got head size {out_dim} (key: {head_key}).")
    model = FlexibleDigitCNN(num_classes=16).to(device)
    missing, mism = _compare_key_shapes(model, sd)
    if mism or missing:
        msgs = []
        if mism:
            msgs.append("Shape mismatches:"); msgs += mism[:20]
            if len(mism) > 20: msgs.append(f"... (+{len(mism)-20} more)")
        if missing:
            msgs.append("Missing keys:"); msgs += missing[:20]
            if len(missing) > 20: msgs.append(f"... (+{len(missing)-20} more)")
        raise RuntimeError("Incompatible checkpoint:\n" + "\n".join(msgs))
    model.load_state_dict(sd, strict=False); model.eval(); return model

# -------------------- Synthesis (digits only / touching) --------------------
def compose_from_digit_folders(root_digits, outdir,
                               N=100, len_min=3, len_max=4,
                               overlap_min=1, overlap_max=3,
                               spacing=4, left_pad=10, right_pad=10,
                               blend="max"):
    os.makedirs(outdir, exist_ok=True)
    cls_files = {}
    for d in range(10):
        files = glob.glob(os.path.join(root_digits, str(d), "*"))
        if not files: raise RuntimeError(f"No files in {root_digits}/{d}")
        cls_files[d] = files

    def load_digit(d):
        p = random.choice(cls_files[d])
        arr = _safe_resize_28(to_u8_gray(Image.open(p)))
        return (arr.astype(np.float32)/255.0)

    def make_sample():
        k = random.randint(int(len_min), int(len_max))
        labels = [int(random.randint(0,9)) for _ in range(k)]
        parts  = [load_digit(d) for d in labels]
        overlaps = [max(0, random.randint(int(overlap_min), int(overlap_max))) for _ in range(k-1)]
        H = parts[0].shape[0]
        total_w = left_pad + sum(p.shape[1] for p in parts) - sum(overlaps) + spacing*(k-1) + right_pad
        canvas = np.zeros((H, total_w), dtype=np.float32)
        x = left_pad
        for i,p in enumerate(parts):
            h,w = p.shape
            if blend == "max": canvas[:, x:x+w] = np.maximum(canvas[:, x:x+w], p)
            else:              canvas[:, x:x+w] = np.clip(canvas[:, x:x+w] + p, 0, 1)
            if i < k-1: x = x + w - overlaps[i] + spacing
        gt = "".join(map(str, labels))
        return (canvas*255).astype(np.uint8), gt

    for i in range(int(N)):
        arr, seq = make_sample()
        Image.fromarray(arr, mode="L").save(os.path.join(outdir, f"img_{i:03d}_{seq}.png"))
    return outdir

def compose_touching_digits(root_digits, outdir,
                            N=100, len_min=3, len_max=5,
                            overlap_min=14, overlap_max=24,
                            left_pad=8, right_pad=8, spacing=0):
    return compose_from_digit_folders(
        root_digits=root_digits,
        outdir=outdir,
        N=N, len_min=len_min, len_max=len_max,
        overlap_min=max(0, int(overlap_min)),
        overlap_max=min(27, int(overlap_max)),
        spacing=int(spacing),
        left_pad=int(left_pad), right_pad=int(right_pad),
        blend="max"
    )

# -------------------- Expression composer (digits + operators) --------------------
OP_DIR_ALIASES: Dict[str, List[str]] = {
    "plus":   ["plus", "+", "add", "plus_sign"],
    "minus":  ["minus", "-", "sub", "minus_sign"],
    "times":  ["times", "mul", "multiply", "asterisk", "x", "*"],
    "divide": ["divide", "div", "slash", "/", "over"],
    "lparen": ["lparen", "lpar", "left_paren", "(", "Ipar"],
    "rparen": ["rparen", "rpar", "right_paren", ")"],
}
OP_SAFE = {"+":"A", "-":"B", "*":"C", "/":"D", "(":"L", ")":"R"}

def _gather_digit_files(root_digits: str) -> Dict[int, List[str]]:
    dct = {}
    for d in range(10):
        files = glob.glob(os.path.join(str(root_digits), str(d), "*"))
        if not files: raise RuntimeError(f"No digit glyphs found in: {root_digits}/{d}")
        dct[d] = files
    return dct

def _gather_op_files(root_ops: str) -> Dict[str, List[str]]:
    buckets = {}
    for key, aliases in OP_DIR_ALIASES.items():
        files: List[str] = []
        for name in aliases: files.extend(glob.glob(os.path.join(str(root_ops), str(name), "*")))
        if not files: raise RuntimeError(f"No operator glyphs for '{key}' under {root_ops}/({','.join(aliases)})")
        buckets[key] = files
    return buckets

def _clamp(v, a, b): return max(a, min(b, v))

def compose_one_expression(
    digits_root: str, ops_root: str,
    terms_min=2, terms_max=4,
    overlap_min=1, overlap_max=3,
    spacing=4, p_parentheses=0.35, two_digit_prob=0.40
) -> Tuple[np.ndarray, str, str]:
    dig_files = _gather_digit_files(digits_root)
    op_files  = _gather_op_files(ops_root)

    terms_min = int(_clamp(terms_min, 2, 8)); terms_max = int(_clamp(terms_max, terms_min, 8))
    overlap_min = int(_clamp(overlap_min, 0, 27)); overlap_max = int(_clamp(overlap_max, overlap_min, 27))
    spacing = int(max(0, spacing))
    p_parentheses = float(_clamp(p_parentheses, 0.0, 1.0))
    two_digit_prob = float(_clamp(two_digit_prob, 0.0, 1.0))

    def load_digit(d: int):
        p = random.choice(dig_files[d])
        arr = _safe_resize_28(to_u8_gray(Image.open(p)))
        return (arr.astype(np.float32)/255.0, str(d))

    def load_op(sym: str):
        key = {"+":"plus","-":"minus","*":"times","/":"divide","(":"lparen",")":"rparen"}[sym]
        p = random.choice(op_files[key])
        arr = _safe_resize_28(to_u8_gray(Image.open(p)))
        return (arr.astype(np.float32)/255.0, sym)

    def sample_number():
        n_digits = 2 if random.random() < two_digit_prob else 1
        glyphs, labels = [], []
        for _ in range(n_digits):
            d = random.randint(0,9)
            g, lab = load_digit(d)
            glyphs.append(g); labels.append(lab)
        return glyphs, labels

    tokens: List[Tuple[np.ndarray,str]] = []
    terms = random.randint(terms_min, terms_max)

    g, labs = sample_number()
    tokens += list(zip(g, labs))
    for _ in range(terms-1):
        op = random.choice(["+","-","*","/"])
        go, lo = load_op(op)
        tokens.append((go, lo))
        g, labs = sample_number()
        tokens += list(zip(g, labs))

    if random.random() <= p_parentheses and len(tokens) >= 5:
        for _ in range(16):
            i = random.randint(0, len(tokens)-2)
            j_min = i + 2
            if j_min >= len(tokens): continue
            j = random.randint(j_min, len(tokens)-1)
            L,_ = load_op("("); R,_ = load_op(")")
            tokens = tokens[:i] + [(L,"(")] + tokens[i:j] + [(R,")")] + tokens[j:]
            break

    glyphs = [g for g,_ in tokens]
    labels = [l for _,l in tokens]
    overlaps = [random.randint(overlap_min, overlap_max) for _ in range(len(glyphs)-1)]
    H = 28; left_pad = right_pad = 10
    total_w = left_pad + sum(g.shape[1] for g in glyphs) - sum(overlaps) + spacing*(len(glyphs)-1) + right_pad
    total_w = max(total_w, 28)
    canvas = np.zeros((H, total_w), dtype=np.float32)
    x = left_pad
    for i,g in enumerate(glyphs):
        h,w = g.shape
        canvas[:, x:x+w] = np.maximum(canvas[:, x:x+w], g)
        if i < len(glyphs)-1:
            x = x + w - overlaps[i] + spacing

    label = "".join(labels)
    safe  = "".join(OP_SAFE.get(ch, ch) for ch in label)
    return (canvas*255).astype(np.uint8), label, safe

# -------------------- Safe expression evaluation --------------------
ALLOWED_CHARS = set("0123456789+-*/()")

def normalize_expr(seq: str) -> str:
    s = re.sub(r"[^0-9+\-*/()]", "", seq)
    # small helpers for implicit multiply (rare): 2(3)->2*(3), )(->)*(
    s = re.sub(r"(\d)\(", r"\1*(", s)
    s = re.sub(r"\)(\d)", r")*\1", s)
    s = re.sub(r"\)\(", r")*(", s)
    return s

def safe_eval_expr(expr: str) -> float:
    """
    Evaluate + - * / and parentheses safely via AST (supports unary +/-).
    """
    node = ast.parse(expr, mode="eval")
    def _eval(n):
        if isinstance(n, ast.Expression): return _eval(n.body)
        if isinstance(n, ast.Num):        return n.n  # py<3.8
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)): return n.value
            raise ValueError("non-numeric constant")
        if isinstance(n, ast.UnaryOp):
            v = _eval(n.operand)
            if isinstance(n.op, ast.UAdd): return +v
            if isinstance(n.op, ast.USub): return -v
            raise ValueError("bad unary op")
        if isinstance(n, ast.BinOp):
            a, b = _eval(n.left), _eval(n.right)
            if isinstance(n.op, ast.Add):  return a + b
            if isinstance(n.op, ast.Sub):  return a - b
            if isinstance(n.op, ast.Mult): return a * b
            if isinstance(n.op, ast.Div):  return a / b
            raise ValueError("bad binary op")
        if isinstance(n, ast.Paren):      # not used in Python AST, kept for clarity
            return _eval(n.value)
        raise ValueError(f"disallowed node: {type(n).__name__}")
    return float(_eval(node))

# -------------------- GUI --------------------
class AcquisitionGUI:
    LABEL_MAP_16 = ['0','1','2','3','4','5','6','7','8','9','+','-','*','/','(',')']

    def __init__(self, root, initial_weights: Optional[str]):
        self.root = root
        self.root.title("Acquisition / Segmentation / Evaluation — Expr (16-class)")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None

        # Segmentation defaults
        self.seg_params = {
            "use_otsu": True,
            "seg_score_floor": 0.18,
            "seg_nms_px": 8,
            "seg_min_gap_px": 8,
            "seg_max_span_factor": 1.8,
            "per_cut_bias": 0.02,
            "recur_max_depth": 3,
            "recur_gain_thr": 0.00,
        }
        self.scale_adapt = tk.BooleanVar(value=True)

        paned = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL); paned.pack(fill=tk.BOTH, expand=True)

        # Left viewer
        left = ttk.Frame(paned); paned.add(left, weight=3)
        self.fig, self.ax = plt.subplots(figsize=(7.2,4.2))
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax.axis("off"); self.fig.tight_layout(); self.canvas_plot.draw()

        # Right controls
        right = ttk.Frame(paned); paned.add(right, weight=2)

        # Model chooser
        model_frame = ttk.LabelFrame(right, text="Expression Model (16 classes)")
        model_frame.pack(fill=tk.X, padx=6, pady=(8,6))
        ttk.Button(model_frame, text="Choose 16-class Weights…", command=self.on_choose_model)\
            .grid(row=0, column=0, padx=4, pady=4, sticky="w")
        self.var_model_info = tk.StringVar(value="No model loaded.")
        ttk.Label(model_frame, textvariable=self.var_model_info, foreground="#444")\
            .grid(row=0, column=1, columnspan=4, sticky="w", padx=8)

        if initial_weights and os.path.exists(initial_weights):
            try:
                mdl = load_model_expr16_nonsilent(initial_weights, self.device)
                self.model = mdl
                self.var_model_info.set(f"Loaded: {os.path.basename(initial_weights)} (16 classes)")
            except Exception as e:
                messagebox.showerror("Model Load Error", str(e))

        # File actions
        btns = ttk.Frame(right); btns.pack(fill=tk.X, pady=(6,4))
        ttk.Button(btns, text="Load Image & Predict", command=self.on_load_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Evaluate Folder (digits-only)", command=self.on_eval_folder).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(btns, text="Scale-aware params", variable=self.scale_adapt).pack(side=tk.LEFT, padx=8)

        # Digits batch composer
        comp = ttk.LabelFrame(right, text="Compose Synthetic Data (digits batch)")
        comp.pack(fill=tk.X, padx=6, pady=6)
        self.var_root_digits = tk.StringVar(value="acq_digits")
        self.var_outdir_root = tk.StringVar(value="acq_test")
        self.var_n           = tk.IntVar(value=200)
        self.var_len_min     = tk.IntVar(value=3)
        self.var_len_max     = tk.IntVar(value=4)
        self.var_ov_min      = tk.IntVar(value=1)
        self.var_ov_max      = tk.IntVar(value=3)
        self.var_spacing     = tk.IntVar(value=4)

        r = 0
        ttk.Label(comp, text="digits_root").grid(row=r, column=0, sticky="w")
        ttk.Entry(comp, textvariable=self.var_root_digits, width=18).grid(row=r, column=1, padx=4)
        ttk.Label(comp, text="out_root").grid(row=r, column=2, sticky="w")
        ttk.Entry(comp, textvariable=self.var_outdir_root, width=18).grid(row=r, column=3, padx=4)
        ttk.Label(comp, text="N").grid(row=r, column=4, sticky="w")
        ttk.Entry(comp, textvariable=self.var_n, width=6).grid(row=r, column=5, padx=4); r += 1

        ttk.Label(comp, text="len[min,max]").grid(row=r, column=0, sticky="w")
        ttk.Entry(comp, textvariable=self.var_len_min, width=6).grid(row=r, column=1, sticky="w")
        ttk.Entry(comp, textvariable=self.var_len_max, width=6).grid(row=r, column=1, sticky="e", padx=(58,0))
        ttk.Label(comp, text="overlap[min,max]").grid(row=r, column=2, sticky="w")
        ttk.Entry(comp, textvariable=self.var_ov_min, width=6).grid(row=r, column=3, sticky="w")
        ttk.Entry(comp, textvariable=self.var_ov_max, width=6).grid(row=r, column=3, sticky="e", padx=(58,0))
        ttk.Label(comp, text="spacing").grid(row=r, column=4, sticky="w")
        ttk.Entry(comp, textvariable=self.var_spacing, width=6).grid(row=r, column=5, sticky="w"); r += 1

        ttk.Button(comp, text="Compose (digits-only, batch)", command=self.on_compose_digits)\
            .grid(row=r, column=0, columnspan=6, sticky="we", pady=(2,2))

        # Touching digits composer
        touch = ttk.LabelFrame(right, text="Compose TOUCHING Digits (stress test)")
        touch.pack(fill=tk.X, padx=6, pady=6)
        self.var_touch_outroot = tk.StringVar(value="acq_touch")
        self.var_touch_n       = tk.IntVar(value=120)
        self.var_touch_len_min = tk.IntVar(value=3)
        self.var_touch_len_max = tk.IntVar(value=5)
        self.var_touch_ov_min  = tk.IntVar(value=14)
        self.var_touch_ov_max  = tk.IntVar(value=24)
        self.var_touch_spacing = tk.IntVar(value=0)

        rt = 0
        ttk.Label(touch, text="out_root").grid(row=rt, column=0, sticky="w")
        ttk.Entry(touch, textvariable=self.var_touch_outroot, width=18).grid(row=rt, column=1, padx=4)
        ttk.Label(touch, text="N").grid(row=rt, column=2, sticky="w")
        ttk.Entry(touch, textvariable=self.var_touch_n, width=8).grid(row=rt, column=3, padx=4)
        ttk.Label(touch, text="len[min,max]").grid(row=rt, column=4, sticky="w")
        ttk.Entry(touch, textvariable=self.var_touch_len_min, width=6).grid(row=rt, column=5, sticky="w")
        ttk.Entry(touch, textvariable=self.var_touch_len_max, width=6).grid(row=rt, column=5, sticky="e", padx=(58,0)); rt += 1

        ttk.Label(touch, text="overlap[min,max]").grid(row=rt, column=0, sticky="w")
        ttk.Entry(touch, textvariable=self.var_touch_ov_min, width=6).grid(row=rt, column=1, sticky="w")
        ttk.Entry(touch, textvariable=self.var_touch_ov_max, width=6).grid(row=rt, column=1, sticky="e", padx=(58,0))
        ttk.Label(touch, text="spacing").grid(row=rt, column=2, sticky="w")
        ttk.Entry(touch, textvariable=self.var_touch_spacing, width=6).grid(row=rt, column=3, sticky="w")
        ttk.Button(touch, text="Compose TOUCHING (batch)", command=self.on_compose_touching)\
            .grid(row=rt, column=4, columnspan=2, sticky="we", padx=6); rt += 1

        # ONE expression composer
        expr = ttk.LabelFrame(right, text="Compose ONE Expression (digits + operators)")
        expr.pack(fill=tk.X, padx=6, pady=6)
        self.var_root_ops     = tk.StringVar(value="acq_ops")
        self.var_terms_min    = tk.IntVar(value=2)
        self.var_terms_max    = tk.IntVar(value=4)
        self.var_p_paren      = tk.DoubleVar(value=0.35)
        self.var_two_digit_p  = tk.DoubleVar(value=0.40)
        self.var_ov_min      = tk.IntVar(value=1)
        self.var_ov_max      = tk.IntVar(value=3)
        self.var_spacing     = tk.IntVar(value=4)

        r2 = 0
        ttk.Label(expr, text="ops_root").grid(row=r2, column=0, sticky="w")
        ttk.Entry(expr, textvariable=self.var_root_ops, width=18).grid(row=r2, column=1, padx=4)
        ttk.Label(expr, text="terms[min,max]").grid(row=r2, column=2, sticky="w")
        ttk.Entry(expr, textvariable=self.var_terms_min, width=6).grid(row=r2, column=3, sticky="w")
        ttk.Entry(expr, textvariable=self.var_terms_max, width=6).grid(row=r2, column=3, sticky="e", padx=(58,0))
        ttk.Label(expr, text="p(parentheses)").grid(row=r2, column=4, sticky="w")
        ttk.Spinbox(expr, from_=0.0, to=1.0, increment=0.05, textvariable=self.var_p_paren, width=6)\
            .grid(row=r2, column=5, sticky="w"); r2 += 1

        ttk.Label(expr, text="p(two-digit numbers)").grid(row=r2, column=0, sticky="w")
        ttk.Spinbox(expr, from_=0.0, to=1.0, increment=0.05, textvariable=self.var_two_digit_p, width=6)\
            .grid(row=r2, column=1, sticky="w")
        ttk.Label(expr, text="overlap[min,max]").grid(row=r2, column=2, sticky="w")
        ttk.Entry(expr, textvariable=self.var_ov_min, width=6).grid(row=r2, column=3, sticky="w")
        ttk.Entry(expr, textvariable=self.var_ov_max, width=6).grid(row=r2, column=3, sticky="e", padx=(58,0))
        ttk.Label(expr, text="spacing").grid(row=r2, column=4, sticky="w")
        ttk.Entry(expr, textvariable=self.var_spacing, width=6).grid(row=r2, column=5, sticky="w"); r2 += 1

        btn_row2 = ttk.Frame(expr); btn_row2.grid(row=r2, column=0, columnspan=6, sticky="we")
        ttk.Button(btn_row2, text="Compose ONE Expr (preview)", command=self.on_compose_one_expr)\
            .pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row2, text="Save Preview As…", command=self.on_save_preview)\
            .pack(side=tk.LEFT, padx=4)

        # Segmentation hyperparameters
        hp = ttk.LabelFrame(right, text="Segmentation Hyperparameters")
        hp.pack(fill=tk.X, padx=6, pady=6)
        self.use_shear = tk.BooleanVar(value=False)
        if HAS_SHEAR:
            ttk.Checkbutton(hp, text="Use shear deskew (helps slanted/touching)", variable=self.use_shear)\
                .grid(row=0, column=0, columnspan=3, sticky="w", pady=(0,4))

        self.var_score_floor = tk.DoubleVar(value=self.seg_params["seg_score_floor"])
        self.var_nms = tk.IntVar(value=self.seg_params["seg_nms_px"])
        self.var_min_gap = tk.IntVar(value=self.seg_params["seg_min_gap_px"])
        self.var_max_span = tk.DoubleVar(value=self.seg_params["seg_max_span_factor"])
        self.var_per_cut = tk.DoubleVar(value=self.seg_params["per_cut_bias"])
        self.var_recur_depth = tk.IntVar(value=self.seg_params["recur_max_depth"])
        self.var_recur_gain = tk.DoubleVar(value=self.seg_params["recur_gain_thr"])

        rr = 1
        ttk.Label(hp, text="score_floor").grid(row=rr, column=0, sticky="w")
        ttk.Spinbox(hp, from_=0.10, to=0.40, increment=0.01, textvariable=self.var_score_floor, width=6)\
            .grid(row=rr, column=1, sticky="w"); rr+=1
        ttk.Label(hp, text="nms_px").grid(row=rr, column=0, sticky="w")
        ttk.Spinbox(hp, from_=2, to=20, increment=1, textvariable=self.var_nms, width=6)\
            .grid(row=rr, column=1, sticky="w"); rr+=1
        ttk.Label(hp, text="min_gap_px").grid(row=rr, column=0, sticky="w")
        ttk.Spinbox(hp, from_=4, to=22, increment=1, textvariable=self.var_min_gap, width=6)\
            .grid(row=rr, column=1, sticky="w"); rr+=1
        ttk.Label(hp, text="max_span_factor").grid(row=rr, column=0, sticky="w")
        ttk.Spinbox(hp, from_=1.0, to=3.0, increment=0.05, textvariable=self.var_max_span, width=6)\
            .grid(row=rr, column=1, sticky="w"); rr+=1
        ttk.Label(hp, text="per_cut_bias").grid(row=rr, column=0, sticky="w")
        ttk.Spinbox(hp, from_=0.00, to=0.10, increment=0.005, textvariable=self.var_per_cut, width=6)\
            .grid(row=rr, column=1, sticky="w"); rr+=1
        ttk.Label(hp, text="recur_max_depth").grid(row=rr, column=0, sticky="w")
        ttk.Spinbox(hp, from_=0, to=5, increment=1, textvariable=self.var_recur_depth, width=6)\
            .grid(row=rr, column=1, sticky="w"); rr+=1
        ttk.Label(hp, text="recur_gain_thr").grid(row=rr, column=0, sticky="w")
        ttk.Spinbox(hp, from_=-0.05, to=0.05, increment=0.005, textvariable=self.var_recur_gain, width=6)\
            .grid(row=rr, column=1, sticky="w"); rr+=1

        preset_row = ttk.Frame(hp); preset_row.grid(row=rr, column=0, columnspan=3, sticky="we", pady=(6,2)); rr+=1
        ttk.Label(preset_row, text="Preset:").pack(side=tk.LEFT)
        self.preset_var = tk.StringVar(value="Default (report)")
        presets = ["Default (report)", "Tight / Overlapped", "Well-spaced / Conservative", "Aggressive Split"]
        self.preset_menu = ttk.Combobox(preset_row, textvariable=self.preset_var, values=presets, state="readonly", width=26)
        self.preset_menu.pack(side=tk.LEFT, padx=6)
        ttk.Button(preset_row, text="Apply Preset", command=self.apply_preset).pack(side=tk.LEFT, padx=4)

        run_row = ttk.Frame(hp); run_row.grid(row=rr, column=0, columnspan=3, sticky="we", pady=(4,2)); rr+=1
        ttk.Button(run_row, text="Re-run on Current Image", command=self.on_rerun_current).pack(side=tk.LEFT, padx=4)
        ttk.Button(run_row, text="Reset to Report Defaults", command=self.reset_defaults).pack(side=tk.LEFT, padx=4)

        # Drawing Pad
        pad = ttk.LabelFrame(right, text="Drawing Pad")
        pad.pack(fill=tk.X, padx=6, pady=6)
        self.pad_w, self.pad_h = 560, 160
        self.draw_canvas = tk.Canvas(pad, width=self.pad_w, height=self.pad_h, bg="white", cursor="pencil")
        self.draw_canvas.grid(row=0, column=0, columnspan=6, padx=4, pady=4, sticky="we")

        self.brush = tk.IntVar(value=10)
        ttk.Label(pad, text="Brush").grid(row=1, column=0, sticky="e")
        ttk.Spinbox(pad, from_=3, to=32, increment=1, textvariable=self.brush, width=5).grid(row=1, column=1, sticky="w")
        ttk.Button(pad, text="Predict from Pad", command=self.on_canvas_predict).grid(row=1, column=2, padx=6)
        ttk.Button(pad, text="Clear", command=self.on_canvas_clear).grid(row=1, column=3, padx=6)
        ttk.Button(pad, text="Invert", command=self.on_canvas_invert).grid(row=1, column=4, padx=6)

        # backing PIL image for strokes (white bg)
        self.pad_img = Image.new("L", (self.pad_w, self.pad_h), color=255)
        self.pad_draw = ImageDraw.Draw(self.pad_img)
        self.last_xy = None
        self.draw_canvas.bind("<ButtonPress-1>", self._pad_start)
        self.draw_canvas.bind("<B1-Motion>", self._pad_draw)
        self.draw_canvas.bind("<ButtonRelease-1>", self._pad_end)

        # Outputs
        self.var_seq = tk.StringVar(value="Sequence: —")
        ttk.Label(right, textvariable=self.var_seq, font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=6, pady=(6,0))

        self.var_value = tk.StringVar(value="Value: —")
        ttk.Label(right, textvariable=self.var_value, font=("Segoe UI", 11)).pack(anchor="w", padx=6, pady=(2,8))

        self.prob_tree = ttk.Treeview(right, columns=("digit","conf"), show="headings", height=8)
        self.prob_tree.heading("digit", text="Pred Class")
        self.prob_tree.heading("conf", text="Confidence")
        self.prob_tree.column("digit", width=100, anchor="center")
        self.prob_tree.column("conf", width=100, anchor="center")
        self.prob_tree.pack(fill=tk.X, padx=6, pady=(0,6))

        ttk.Label(right, text="Folder Evaluation Summary").pack(anchor="w", padx=6)
        self.txt = tk.Text(right, height=10); self.txt.pack(fill=tk.BOTH, expand=True, padx=6, pady=(2,8))

        # State
        self.current_image = None
        self.current_boxes = None
        self.preview_small = None
        self.preview_label = "-"
        self.preview_safe  = "-"

    # -------- Drawing pad handlers --------
    def _pad_start(self, e): self.last_xy = (e.x, e.y)
    def _pad_draw(self, e):
        if self.last_xy is None: return
        x0, y0 = self.last_xy; x1, y1 = e.x, e.y
        w = int(self.brush.get())
        self.draw_canvas.create_line(x0, y0, x1, y1, width=w, fill="black", capstyle=tk.ROUND, smooth=True)
        self.pad_draw.line((x0, y0, x1, y1), fill=0, width=w)
        self.last_xy = (x1, y1)
    def _pad_end(self, e): self.last_xy = None
    def on_canvas_clear(self):
        self.draw_canvas.delete("all")
        self.pad_img = Image.new("L", (self.pad_w, self.pad_h), color=255); self.pad_draw = ImageDraw.Draw(self.pad_img)
    def on_canvas_invert(self):
        arr = 255 - np.asarray(self.pad_img, dtype=np.uint8)
        self.pad_img = Image.fromarray(arr, mode="L")

    def on_canvas_predict(self):
        arr = to_u8_gray(self.pad_img)
        self.predict_and_render(arr)

    # -------- Model / presets --------
    def on_choose_model(self):
        path = filedialog.askopenfilename(
            title="Select 16-class expression weights (.pth / .pt)",
            filetypes=[("PyTorch weights","*.pth;*.pt"), ("All files","*.*")]
        )
        if not path: return
        try:
            mdl = load_model_expr16_nonsilent(str(path), self.device)
        except Exception as e:
            messagebox.showerror("Model Load Error", str(e)); return
        self.model = mdl; self.var_model_info.set(f"Loaded: {os.path.basename(path)} (16 classes)")

    def apply_preset(self):
        name = self.preset_var.get()
        if name == "Default (report)":
            vals = dict(sf=0.18, nms=8,  gap=8,  msf=1.8,  pcb=0.02, rd=3, rg=0.00, shear=False)
        elif name == "Tight / Overlapped":
            vals = dict(sf=0.16, nms=5,  gap=6,  msf=1.35, pcb=0.005, rd=4, rg=-0.005, shear=True)
        elif name == "Well-spaced / Conservative":
            vals = dict(sf=0.20, nms=10, gap=10, msf=2.2,  pcb=0.03, rd=2, rg=0.01, shear=False)
        else:
            vals = dict(sf=0.14, nms=4,  gap=5,  msf=1.20, pcb=0.000, rd=5, rg=-0.010, shear=True)
        self.var_score_floor.set(vals["sf"]); self.var_nms.set(vals["nms"]); self.var_min_gap.set(vals["gap"])
        self.var_max_span.set(vals["msf"]); self.var_per_cut.set(vals["pcb"])
        self.var_recur_depth.set(vals["rd"]); self.var_recur_gain.set(vals["rg"])
        if HAS_SHEAR: self.use_shear.set(vals["shear"])
        self.on_rerun_current()

    def reset_defaults(self):
        self.var_score_floor.set(0.18); self.var_nms.set(8); self.var_min_gap.set(8)
        self.var_max_span.set(1.8); self.var_per_cut.set(0.02)
        self.var_recur_depth.set(3); self.var_recur_gain.set(0.00)
        if HAS_SHEAR: self.use_shear.set(False)
        self.on_rerun_current()

    # -------- Pipeline --------
    def _decode_logits(self, logits: torch.Tensor) -> Tuple[List[str], List[float]]:
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        ids   = probs.argmax(1).tolist()
        confs = probs.max(axis=1).tolist()
        C = logits.shape[1]
        active_charset = self.LABEL_MAP_16[:C]
        out = []
        for idx in ids:
            out.append(active_charset[idx] if 0 <= idx < len(active_charset) else "?")
        return out, confs

    def current_seg_params(self, height_hint: Optional[int] = None):
        use_scale = self.scale_adapt.get()
        H = int(height_hint) if height_hint else None
        params = {
            "use_otsu": True,
            "seg_score_floor": float(self.var_score_floor.get()),
            "seg_nms_px": int(self.var_nms.get()),
            "seg_min_gap_px": int(self.var_min_gap.get()),
            "seg_max_span_factor": float(self.var_max_span.get()),
            "per_cut_bias": float(self.var_per_cut.get()),
            "recur_max_depth": int(self.var_recur_depth.get()),
            "recur_gain_thr": float(self.var_recur_gain.get()),
        }
        if use_scale and H is not None and H > 40:
            params["seg_nms_px"]       = max(params["seg_nms_px"], max(6, H//32))
            params["seg_min_gap_px"]   = max(params["seg_min_gap_px"], max(8, H//20))
            params["seg_score_floor"]  = max(params["seg_score_floor"], 0.22)
            params["seg_max_span_factor"] = max(params["seg_max_span_factor"], 2.2)
            params["per_cut_bias"]     = min(params["per_cut_bias"], 0.01)
            params["recur_max_depth"]  = min(params["recur_max_depth"], 1)
            params["recur_gain_thr"]   = max(params["recur_gain_thr"], 0.01)
        return params

    def run_pipeline_on_array(self, arr_u8):
        if self.model is None:
            raise RuntimeError("Load a 16-class model first (Choose 16-class Weights…).")
        arr_u8 = auto_invert_if_needed(arr_u8)
        if HAS_SHEAR and self.use_shear.get():
            try: arr_u8, _ = shear_fn(arr_u8, max_abs=0.45)
            except Exception: pass
        sp = self.current_seg_params(height_hint=arr_u8.shape[0])
        boxes, tensors = segment_by_psc_decode(
            arr_u8, self.model, self.device,
            use_otsu=sp["use_otsu"],
            seg_score_floor=sp["seg_score_floor"],
            seg_nms_px=sp["seg_nms_px"],
            seg_min_gap_px=sp["seg_min_gap_px"],
            seg_max_span_factor=sp["seg_max_span_factor"],
            per_cut_bias=sp["per_cut_bias"],
            recur_max_depth=sp["recur_max_depth"],
            recur_gain_thr=sp["recur_gain_thr"],
        )
        return boxes, tensors, arr_u8

    def show_image_with_boxes(self, arr_u8, boxes):
        self.ax.clear()
        self.ax.imshow(arr_u8, cmap="gray")
        for (x0,x1,y0,y1) in boxes or []:
            rect = patches.Rectangle((x0,y0), x1-x0, y1-y0, linewidth=2, edgecolor='red', facecolor='none')
            self.ax.add_patch(rect)
        self.ax.axis("off"); self.fig.tight_layout(); self.canvas_plot.draw()

    def predict_and_render(self, arr_u8):
        boxes, tensors, arr_proc = self.run_pipeline_on_array(arr_u8)
        self.current_image = arr_proc; self.current_boxes = boxes
        self.show_image_with_boxes(arr_proc, boxes)

        self.prob_tree.delete(*self.prob_tree.get_children())
        if len(tensors) == 0:
            self.var_seq.set("Sequence: — (no segments)")
            self.var_value.set("Value: —")
            return
        with torch.no_grad():
            batch  = torch.cat(tensors, dim=0).to(self.device)
            logits = self.model(batch)
        chars, confs = self._decode_logits(logits)
        seq_raw = "".join(chars)
        self.var_seq.set(f"Sequence: {seq_raw}   N={len(chars)}")
        for ch,c in zip(chars, confs):
            self.prob_tree.insert("", "end", values=(ch, f"{c*100:.1f}%"))

        # ---- Evaluate safely
        try:
            expr = normalize_expr(seq_raw)
            val = safe_eval_expr(expr)
            # show as int if very close
            if abs(val - round(val)) < 1e-9:
                self.var_value.set(f"Value: {int(round(val))}")
            else:
                self.var_value.set(f"Value: {val:.6g}")
        except ZeroDivisionError:
            self.var_value.set("Value: ⌀  (division by zero)")
        except Exception as e:
            self.var_value.set(f"Value: ⌀  (parse error)")

    # -------- UI callbacks --------
    def on_load_image(self):
        if self.model is None:
            messagebox.showwarning("Model", "Load a 16-class weights file first.")
        f = filedialog.askopenfilename(filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp;*.webp")])
        if not f: return
        arr = to_u8_gray(Image.open(f))
        self.predict_and_render(arr)

    def on_rerun_current(self):
        if self.current_image is None: return
        try:
            self.predict_and_render(self.current_image)
        except Exception as e:
            messagebox.showerror("Run Error", str(e))

    def _evaluate_folder_digits_only(self, folder):
        if self.model is None:
            messagebox.showwarning("Model", "Load a 16-class weights file first."); return
        names = [n for n in os.listdir(folder) if n.lower().endswith((".png",".jpg",".jpeg",".bmp",".webp"))]
        names.sort()
        total = seq_correct = 0
        digit_correct = digit_total = 0
        over = under = matched = 0
        self.txt.delete("1.0", tk.END)

        for name in names:
            path = os.path.join(folder, name)
            arr = to_u8_gray(Image.open(path))
            boxes, tensors, _ = self.run_pipeline_on_array(arr)
            if len(tensors) == 0:
                under += 1; total += 1; continue
            with torch.no_grad():
                batch = torch.cat(tensors, dim=0).to(self.device)
                logits = self.model(batch)
            chars, _confs = self._decode_logits(logits)

            pred_seq = "".join(ch for ch in chars if ch.isdigit())  # digits only for eval
            m = re.search(r"_([0-9ABCDLR]+)\.[A-Za-z]+$", name)
            gt_raw = m.group(1) if m else ""
            gt = "".join(ch for ch in gt_raw if ch.isdigit())

            if pred_seq == gt: seq_correct += 1
            if len(pred_seq) > len(gt): over += 1
            elif len(pred_seq) < len(gt): under += 1
            else:
                matched += 1
                digit_total += len(gt)
                digit_correct += sum(int(p==t) for p,t in zip(pred_seq, gt))
            total += 1

        seq_acc = seq_correct/max(1,total)
        dig_acc = digit_correct/max(1,digit_total) if digit_total>0 else 0.0
        over_r  = over/max(1,total)
        under_r = under/max(1,total)

        out = []
        out.append(f"Folder: {folder}")
        out.append(f"samples: {total}")
        out.append(f"sequence_accuracy (digits-only): {seq_acc*100:.2f}%")
        out.append(f"digit_accuracy_matched: {dig_acc*100:.2f}%")
        out.append(f"over_split_rate: {over_r*100:.2f}%")
        out.append(f"under_split_rate: {under_r*100:.2f}%")
        out.append(f"matched_samples: {matched}")
        self.txt.insert(tk.END, "\n".join(out))

        messagebox.showinfo("Folder Evaluation",
                            f"Seq Acc (digits-only): {seq_acc*100:.2f}%\n"
                            f"Digit Acc (matched): {dig_acc*100:.2f}%\n"
                            f"Over-split: {over_r*100:.2f}%\nUnder-split: {under_r*100:.2f}%")

    def on_eval_folder(self):
        d = filedialog.askdirectory()
        if not d: return
        self._evaluate_folder_digits_only(d)

    def on_compose_digits(self):
        root_digits = self.var_root_digits.get().strip()
        outroot     = self.var_outdir_root.get().strip()
        N           = max(1, int(self.var_n.get()))
        len_min     = int(self.var_len_min.get()); len_max = int(self.var_len_max.get())
        ov_min      = int(self.var_ov_min.get());  ov_max  = int(self.var_ov_max.get())
        spacing     = int(self.var_spacing.get())
        try:
            outdir = _make_param_subdir(outroot, len_min, len_max, ov_min, ov_max, spacing, N, tag="")
            saved = compose_from_digit_folders(
                root_digits, outdir, N=N,
                len_min=len_min, len_max=len_max,
                overlap_min=ov_min, overlap_max=ov_max,
                spacing=spacing
            )
            messagebox.showinfo("Compose (digits)", f"Saved {N} images to:\n{os.path.abspath(saved)}")
        except Exception as e:
            messagebox.showerror("Compose Error", str(e))

    def on_compose_touching(self):
        root_digits = self.var_root_digits.get().strip()
        outroot     = self.var_touch_outroot.get().strip()
        N           = max(1, int(self.var_touch_n.get()))
        len_min     = int(self.var_touch_len_min.get()); len_max = int(self.var_touch_len_max.get())
        ov_min      = int(self.var_touch_ov_min.get()); ov_max = int(self.var_touch_ov_max.get())
        spacing     = int(self.var_touch_spacing.get())
        try:
            outdir = _make_param_subdir(outroot, len_min, len_max, ov_min, ov_max, spacing, N, tag="touch_")
            saved = compose_touching_digits(
                root_digits=root_digits, outdir=outdir, N=N,
                len_min=len_min, len_max=len_max,
                overlap_min=ov_min, overlap_max=ov_max,
                spacing=spacing
            )
            messagebox.showinfo("Compose TOUCHING", f"Saved {N} images to:\n{os.path.abspath(saved)}")
        except Exception as e:
            messagebox.showerror("Compose TOUCHING Error", str(e))

    def on_compose_one_expr(self):
        digits_root = self.var_root_digits.get().strip()
        ops_root    = self.var_root_ops.get().strip()
        terms_min   = int(self.var_terms_min.get()); terms_max = int(self.var_terms_max.get())
        ov_min      = int(self.var_ov_min.get());    ov_max    = int(self.var_ov_max.get())
        spacing     = int(self.var_spacing.get())
        p_paren     = float(self.var_p_paren.get())
        p_two       = float(self.var_two_digit_p.get())

        try:
            img_u8, label, safe = compose_one_expression(
                digits_root=digits_root, ops_root=ops_root,
                terms_min=terms_min, terms_max=terms_max,
                overlap_min=ov_min, overlap_max=ov_max,
                spacing=spacing, p_parentheses=p_paren, two_digit_prob=p_two,
            )
        except Exception as e:
            messagebox.showerror("Compose ONE Expr", str(e)); return

        self.preview_small = img_u8; self.preview_label = label; self.preview_safe = safe
        self._show_plain_image(img_u8, title=f"Composed ONE Expr — label: {label} (safe: {safe})")
        self.current_image = img_u8
        self.on_rerun_current()

    def on_save_preview(self):
        if self.preview_small is None:
            messagebox.showwarning("Save", "Nothing to save. Click 'Compose ONE Expr (preview)' first."); return
        initial = f"expr_{self.preview_safe or 'row'}.png"
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG","*.png")], initialfile=str(initial))
        if not path: return
        try:
            Image.fromarray(self.preview_small, mode="L").save(str(path))
            messagebox.showinfo("Save", f"Saved:\n{path}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="", help="Initial 16-class weights to load (optional)")
    args = ap.parse_args()

    root = tk.Tk()
    app = AcquisitionGUI(root, args.weights if args.weights else None)
    root.geometry("1220x920")
    root.mainloop()

if __name__ == "__main__":
    main()
