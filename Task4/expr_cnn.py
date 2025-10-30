# expr16_train.py
# Train / fine-tune a 16-class CNN for digits (0-9) + math ops (+ - * / ( )).
# Adds: 1-epoch digit-first fine-tune stage (freeze conv trunk; digits-focused sampling).
# Also: filters image extensions to avoid PIL errors on non-images.

import os, glob, math, argparse, random, time
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, UnidentifiedImageError

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# --------------------- Model (matches your GUI) ---------------------
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

# --------------------- Class map (fixed order) ---------------------
CLASS_NAMES = ['0','1','2','3','4','5','6','7','8','9','+','-','*','/','(',')']
CLASS_TO_ID = {c:i for i,c in enumerate(CLASS_NAMES)}

OP_DIR_ALIASES: Dict[str, List[str]] = {
    "plus":   ["plus", "+", "add", "plus_sign"],
    "minus":  ["minus", "-", "sub", "minus_sign"],
    "times":  ["times", "mul", "multiply", "asterisk", "x", "*"],
    "divide": ["divide", "div", "slash", "/", "over"],
    "lparen": ["lparen", "lpar", "left_paren", "(", "Ipar"],
    "rparen": ["rparen", "rpar", "right_paren", ")"],
}
OP_TO_CLASS = {"plus":'+', "minus":'-', "times":'*', "divide":'/', "lparen":'(', "rparen":')'}

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.gif', '.tif', '.tiff')

# --------------------- IO / preprocessing ---------------------
def load_u8(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    a = np.asarray(img, dtype=np.uint8)
    return a

def auto_invert_if_needed(u8: np.ndarray) -> np.ndarray:
    return (255 - u8) if float(u8.mean()) > 128 else u8

def safe_resize_28(u8: np.ndarray) -> np.ndarray:
    if u8.shape != (28, 28):
        img = Image.fromarray(u8, mode="L").resize((28, 28), Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)
    return u8

def to_tensor01(u8: np.ndarray) -> torch.Tensor:
    f = (u8.astype(np.float32) / 255.0)[None, ...]
    return torch.from_numpy(f)

# --------------------- Augmentation utils ---------------------
try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

def thickness_jitter(u8: np.ndarray, p: float=0.6) -> np.ndarray:
    if random.random() > p:
        return u8
    k = random.choice([1,2])
    if HAS_CV2:
        kernel = np.ones((3,3), np.uint8)
        if random.random() < 0.5:
            v = cv2.erode(u8, kernel, iterations=k)
        else:
            v = cv2.dilate(u8, kernel, iterations=k)
        return v
    else:
        img = Image.fromarray(u8, mode="L")
        if random.random() < 0.5:
            v = img.filter(ImageFilter.MinFilter(size=3))
        else:
            v = img.filter(ImageFilter.MaxFilter(size=3))
        return np.asarray(v, dtype=np.uint8)

def affine_jitter(u8: np.ndarray) -> np.ndarray:
    img = Image.fromarray(u8, mode="L")
    angle = random.uniform(-15, 15)
    shear = random.uniform(-12, 12)
    scale = random.uniform(0.85, 1.25)
    tx = random.uniform(-2.0, 2.0)
    ty = random.uniform(-2.0, 2.0)
    rad = math.radians(angle)
    cos, sin = math.cos(rad)*scale, math.sin(rad)*scale
    a = cos
    b = -sin + math.tan(math.radians(shear))*cos/28.0
    d = sin
    e =  cos + math.tan(math.radians(shear))*sin/28.0
    img = img.transform(img.size, Image.AFFINE, (a, b, tx, d, e, ty), resample=Image.BILINEAR, fillcolor=0)
    return np.asarray(img, dtype=np.uint8)

def photometric_jitter(u8: np.ndarray) -> np.ndarray:
    img = Image.fromarray(u8, mode="L")
    if random.random() < 0.7:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.4))
    if random.random() < 0.7:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))
    a = np.asarray(img, dtype=np.uint8)
    if random.random() < 0.3:
        noise = np.random.normal(0.0, 8.0, a.shape).astype(np.float32)
        a = np.clip(a.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return a

def pad_margin(u8: np.ndarray, px: int=2) -> np.ndarray:
    img = ImageOps.expand(Image.fromarray(u8, "L"), border=px, fill=0).resize((28,28), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)

def augment(u8: np.ndarray) -> np.ndarray:
    a = thickness_jitter(u8, p=0.7)
    a = affine_jitter(a)
    a = photometric_jitter(a)
    a = pad_margin(a, px=random.choice([1,2,3]))
    return a

# --------------------- Dataset ---------------------
class GlyphFolderDataset(Dataset):
    """Loads single-glyph images and assigns class ids using directory structure."""
    def __init__(self, digits_root: str, ops_root: str, augment_prob: float=0.7):
        self.samples: List[Tuple[str,int]] = []
        self.augment_prob = augment_prob

        # digits 0..9
        for d in range(10):
            files = sorted(glob.glob(os.path.join(digits_root, str(d), "*")))
            files = [p for p in files if p.lower().endswith(IMG_EXTS)]
            for p in files:
                self.samples.append((p, d))

        # operators
        buckets: Dict[str, List[str]] = {}
        for key, aliases in OP_DIR_ALIASES.items():
            files: List[str] = []
            for name in aliases:
                cand = glob.glob(os.path.join(ops_root, name, "*"))
                files.extend([p for p in cand if p.lower().endswith(IMG_EXTS)])
            buckets[key] = files

        for key, files in buckets.items():
            if not files:
                continue
            cls_char = OP_TO_CLASS[key]
            cls_id = CLASS_TO_ID[cls_char]
            for p in files:
                self.samples.append((p, cls_id))

        if len(self.samples) == 0:
            raise RuntimeError("No glyphs found. Check digits_root and ops_root.")

        random.shuffle(self.samples)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        try:
            u8 = load_u8(path)
        except UnidentifiedImageError as e:
            raise RuntimeError(f"Unrecognized image file: {path}") from e
        u8 = auto_invert_if_needed(u8)
        u8 = safe_resize_28(u8)

        if random.random() < self.augment_prob:
            u8 = augment(u8)

        x = to_tensor01(u8)  # 1x28x28, [0,1]
        return x, y

def split_dataset(ds: Dataset, val_ratio: float=0.1, seed: int=42):
    n = len(ds)
    n_val = int(n * val_ratio)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    val_idx = set(indices[:n_val])
    tr_idx  = [i for i in indices if i not in val_idx]
    from torch.utils.data import Subset
    return Subset(ds, tr_idx), Subset(ds, list(val_idx))

def make_samplers(ds: Dataset):
    counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    for _, y in ds:
        counts[y] += 1
    weights = [1.0 / max(1, counts[y]) for _, y in ds]
    sampler = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True)
    return sampler, counts

def make_samplers_digits_focus(ds: Dataset, digits_focus: float = 0.85):
    """Target mix: digits portion = digits_focus; ops portion = 1 - digits_focus."""
    counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    ys = []
    for _, y in ds:
        ys.append(y); counts[y] += 1
    total = len(ds)
    target = np.zeros(16, dtype=np.float32)
    target[:10] = digits_focus / 10.0
    target[10:] = (1.0 - digits_focus) / 6.0
    empirical = counts / max(1, total)
    weights_per_class = target / np.clip(empirical, 1e-8, None)
    weights = [float(weights_per_class[y]) for y in ys]
    sampler = WeightedRandomSampler(weights, num_samples=total, replacement=True)
    return sampler, counts

# --------------------- Training / Eval ---------------------
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(1)
    return (pred == y).float().mean().item()

def confusion_matrix(pred: np.ndarray, true: np.ndarray, ncls: int=16) -> np.ndarray:
    cm = np.zeros((ncls, ncls), dtype=np.int64)
    for p,t in zip(pred, true):
        cm[t, p] += 1
    return cm

def evaluate(model, loader, device):
    model.eval()
    tot, acc_sum = 0, 0.0
    preds_all, true_all = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            acc_sum += accuracy(logits, y) * x.size(0)
            tot += x.size(0)
            preds_all += logits.argmax(1).cpu().tolist()
            true_all  += y.cpu().tolist()
    cm = confusion_matrix(np.array(preds_all), np.array(true_all), ncls=16)
    per_class = (cm.diagonal() / np.clip(cm.sum(1), 1, None)).tolist()
    return acc_sum / max(1, tot), cm, per_class

# Digit/Op-aware label smoothing (lower on digits)
class DigitOpAwareCE(nn.Module):
    def __init__(self, ls_digits=0.02, ls_ops=0.10, num_classes=16):
        super().__init__()
        self.ls_digits = float(ls_digits)
        self.ls_ops    = float(ls_ops)
        self.C = int(num_classes)
    def forward(self, logits, y):
        y = y.view(-1)
        ls = torch.where(y < 10,
                         torch.full_like(y, self.ls_digits, dtype=torch.float32),
                         torch.full_like(y, self.ls_ops,    dtype=torch.float32))
        with torch.no_grad():
            target = torch.zeros_like(logits).scatter_(1, y.unsqueeze(1), 1.0)
            smooth = ls.unsqueeze(1) / (self.C - 1)
            q = target * (1.0 - ls.unsqueeze(1)) + (1.0 - target) * smooth
        logp = torch.log_softmax(logits, dim=1)
        return -(q * logp).sum(dim=1).mean()

def maybe_load_ckpt(model, weights_path, device):
    ckpt = torch.load(weights_path, map_location=device)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(sd, strict=True)
    return model

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    ds = GlyphFolderDataset(args.digits_root, args.ops_root, augment_prob=args.augment)
    tr_ds, va_ds = split_dataset(ds, val_ratio=args.val_ratio, seed=42)

    # Base loaders (uniform-ish via inverse-frequency)
    tr_sampler, class_counts = make_samplers(tr_ds)
    print("[Class counts]", dict(zip(CLASS_NAMES, class_counts.tolist())))
    tr = DataLoader(tr_ds, batch_size=args.batch, sampler=tr_sampler,
                    num_workers=0, pin_memory=(device.type=="cuda"))
    va = DataLoader(va_ds, batch_size=args.batch, shuffle=False,
                    num_workers=0, pin_memory=(device.type=="cuda"))

    model = FlexibleDigitCNN(16).to(device)

    # Optional: start from checkpoint before base training
    if args.finetune_from and args.epochs == 0:
        print(f"[Init] Loading checkpoint for fine-tune: {args.finetune_from}")
        maybe_load_ckpt(model, args.finetune_from, device)

    # -------- Base training (if epochs > 0) --------
    best_va = 0.0
    best_state = None
    if args.epochs > 0:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        # Keep original smoothing for base stage
        base_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)

        patience = args.patience
        stall = 0

        for epoch in range(1, args.epochs+1):
            model.train()
            t0 = time.time()
            running = 0.0; nseen = 0

            for x, y in tr:
                x = x.to(device); y = y.to(device)
                logits = model(x)
                loss = base_loss(logits, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                running += loss.item() * x.size(0)
                nseen  += x.size(0)

            sched.step()
            tr_loss = running / max(1, nseen)
            va_acc, cm, per_class = evaluate(model, va, device)

            print(f"Epoch {epoch:02d} | tr_loss {tr_loss:.4f} | va_acc {va_acc*100:.2f}% | time {time.time()-t0:.1f}s")
            print(" per-class acc:", " ".join(f"{CLASS_NAMES[i]}:{a*100:.1f}%" for i,a in enumerate(per_class)))

            if va_acc > best_va + 1e-4:
                best_va = va_acc
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                stall = 0
            else:
                stall += 1
                if stall >= patience:
                    print(f"Early stop (patience {patience})")
                    break

        if best_state is None:
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state, strict=True)

    # -------- Digit-first fine-tune (optional) --------
    if args.ft_epochs > 0:
        if args.finetune_from and args.epochs > 0:
            print("[Fine-tune] Base training already ran; ignoring --finetune_from.")
        elif args.finetune_from and args.epochs == 0:
            # already loaded above
            pass
        elif best_state is not None:
            print("[Fine-tune] Starting from best base state.")
            model.load_state_dict(best_state, strict=True)

        # Freeze conv trunk unless unfreeze requested
        if not args.ft_unfreeze:
            for p in model.features.parameters():
                p.requires_grad = False
            print("[Fine-tune] Frozen feature extractor; training classifier head only.")
            ft_params = [p for p in model.classifier.parameters() if p.requires_grad]
        else:
            print("[Fine-tune] Unfrozen all layers for fine-tune.")
            ft_params = [p for p in model.parameters() if p.requires_grad]

        # Digits-focused sampler
        ft_sampler, _ = make_samplers_digits_focus(tr_ds, digits_focus=args.ft_digits_focus)
        ft_tr = DataLoader(tr_ds, batch_size=args.batch, sampler=ft_sampler,
                           num_workers=0, pin_memory=(device.type=="cuda"))

        ft_opt = torch.optim.AdamW(ft_params, lr=args.ft_lr, weight_decay=1e-4)
        ft_loss = DigitOpAwareCE(ls_digits=0.02, ls_ops=0.10, num_classes=16)

        for epoch in range(1, args.ft_epochs+1):
            model.train()
            t0 = time.time()
            run = 0.0; nseen = 0
            for x, y in ft_tr:
                x = x.to(device); y = y.to(device)
                logits = model(x)
                loss = ft_loss(logits, y)
                ft_opt.zero_grad(set_to_none=True)
                loss.backward()
                ft_opt.step()
                run += loss.item() * x.size(0)
                nseen += x.size(0)
            tr_loss = run / max(1, nseen)
            va_acc, cm, per_class = evaluate(model, va, device)
            print(f"[FT] Epoch {epoch:02d} | tr_loss {tr_loss:.4f} | va_acc {va_acc*100:.2f}% | time {time.time()-t0:.1f}s")
            print("      per-class acc:", " ".join(f"{CLASS_NAMES[i]}:{a*100:.1f}%" for i,a in enumerate(per_class)))

        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state, strict=True)

    # -------- Save + final eval on full (non-aug) set --------
    out = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "meta": {
            "class_names": CLASS_NAMES,
            "preprocess": "auto_invert_then_div255_resize28",
            "arch": "FlexibleDigitCNN(16)",
        }
    }
    torch.save(out, args.out)
    print(f"[Saved] {args.out}")

    ds_eval = GlyphFolderDataset(args.digits_root, args.ops_root, augment_prob=0.0)
    ev_loader = DataLoader(ds_eval, batch_size=args.batch, shuffle=False, num_workers=0)
    acc, cm, per_class = evaluate(model, ev_loader, device)
    print(f"[Full-set accuracy] {acc*100:.2f}%")
    print(" per-class:", " ".join(f"{CLASS_NAMES[i]}:{a*100:.1f}%" for i,a in enumerate(per_class)))

def eval_only(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.weights, map_location=device)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model = FlexibleDigitCNN(16).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()

    ds_eval = GlyphFolderDataset(args.digits_root, args.ops_root, augment_prob=0.0)
    ev_loader = DataLoader(ds_eval, batch_size=args.batch, shuffle=False, num_workers=0)
    acc, cm, per_class = evaluate(model, ev_loader, device)
    print(f"[Eval-only] accuracy {acc*100:.2f}%")
    print(" per-class:", " ".join(f"{CLASS_NAMES[i]}:{a*100:.1f}%" for i,a in enumerate(per_class)))

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--digits_root", default="acq_digits")
    ap.add_argument("--ops_root",    default="acq_ops")
    ap.add_argument("--out",         default="expr16_clean.pth")
    ap.add_argument("--epochs",      type=int, default=12)
    ap.add_argument("--batch",       type=int, default=256)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--val_ratio",   type=float, default=0.1)
    ap.add_argument("--augment",     type=float, default=0.7, help="probability to apply augments")
    ap.add_argument("--patience",    type=int, default=4)

    # Fine-tune controls
    ap.add_argument("--finetune_from", default="", help="Path to checkpoint to fine-tune from (optional)")
    ap.add_argument("--ft_epochs",     type=int, default=0, help="Fine-tune epochs (e.g., 1)")
    ap.add_argument("--ft_lr",         type=float, default=1e-4, help="Fine-tune learning rate")
    ap.add_argument("--ft_digits_focus", type=float, default=0.85, help="Target digit portion during FT (0..1)")
    ap.add_argument("--ft_unfreeze",   action="store_true", help="If set, unfreeze all layers during FT")

    ap.add_argument("--eval_only",   action="store_true")
    ap.add_argument("--weights",     default="")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.eval_only:
        if not args.weights:
            raise SystemExit("--eval_only requires --weights")
        eval_only(args)
    else:
        train(args)
