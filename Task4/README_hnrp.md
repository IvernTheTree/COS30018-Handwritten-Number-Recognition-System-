# Expression (16‑class) CNN — Training & End‑to‑End Pipeline

This repo trains and uses a **16‑class CNN** that recognizes digits `0–9` and operators `+ − × ÷ ( )` on 28×28 grayscale glyphs. It includes:
- **Training / eval script:** `expr16_train.py` (now supports **1‑epoch digit‑first fine‑tune**)
- **Acquisition + Segmentation + Inference GUI:** `gui_acquisition_expr16.py`
  - Segmentation matches your report’s PSC → DP → recursive resegment (+ seam fix).
  - Folder evaluation (digits‑only), and two composers: **digits batch** and **one expression**.

> Works on CPU or GPU (CUDA if available). Python **3.12.7** recommended (3.9+ OK).

---

## 1) Environment Setup

### 1.1 Create & activate a virtual environment
**Windows (PowerShell):**
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

> If PowerShell blocks activation:  
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` (then re‑activate).

### 1.2 Install dependencies
Minimal set (CPU):
```bash
pip install torch numpy pillow matplotlib opencv-python
```
GUI & segmentation use the standard library (`tkinter`) and `matplotlib`. On Linux you might need:
```bash
sudo apt-get update && sudo apt-get install -y python3-tk
```

> For CUDA builds of PyTorch, install from https://pytorch.org/get-started/locally/

---

## 2) Prepare Data

Organize glyph PNGs into these folders (names are **case‑insensitive**; other files are **ignored**):
```powershell
python export_mnist.py
python export_mnist_ops.py
```
```
acq_digits/
  0/ *.png
  1/ *.png
  ...
  9/ *.png

acq_ops/
  plus/    or "+", "add", "plus_sign"
  minus/   or "-", "sub", "minus_sign"
  times/   or "mul", "multiply", "asterisk", "x", "*"
  divide/  or "div", "slash", "/", "over"
  lparen/  or "lpar", "left_paren", "(" , "Ipar"
  rparen/  or "rpar", "right_paren", ")"
```

- The loader auto‑inverts bright backgrounds and resizes to **28×28**.
- Non‑image files are skipped; corrupt images raise a readable error.
- PNG is recommended; JPG/BMP/WEBP/TIF are accepted.

---

## 3) Train the 16‑class Model

### 3.1 Train from scratch
**PowerShell:**
```powershell
python .\expr16_train.py `
  --digits_root .\acq_digits `
  --ops_root .\acq_ops `
  --epochs 12 `
  --batch 256 `
  --out .\expr16_clean.pth
```

**bash:**
```bash
python expr16_train.py \
  --digits_root ./acq_digits \
  --ops_root ./acq_ops \
  --epochs 12 \
  --batch 256 \
  --out ./expr16_clean.pth
```

The script prints validation accuracy and **per‑class** accuracy each epoch. After training it saves the best model and also reports a “full‑set” accuracy (no augments).

### 3.2 One‑epoch digit‑first fine‑tune (recommended)
This sharpens digit decision boundaries while retaining strong operator robustness.

**Fine‑tune only (start from an existing checkpoint):**
```bash
python expr16_train.py \
  --digits_root ./acq_digits \
  --ops_root ./acq_ops \
  --epochs 0 \
  --finetune_from ./expr16_clean.pth \
  --ft_epochs 1 \
  --ft_lr 1e-4 \
  --ft_digits_focus 0.85 \
  --out ./expr16_ft.pth
```

**Train then fine‑tune in one run:**
```bash
python expr16_train.py \
  --digits_root ./acq_digits \
  --ops_root ./acq_ops \
  --epochs 12 \
  --batch 256 \
  --out ./expr16_clean.pth \
  --ft_epochs 1 \
  --ft_lr 1e-4 \
  --ft_digits_focus 0.85
```

Fine‑tune defaults:
- **Freezes** the convolutional trunk (classifier head trains). Add `--ft_unfreeze` to train all layers.
- Uses **Digit/Op‑aware label smoothing** (0.02 digits / 0.10 ops).
- Uses a **digits‑focused sampler** targeting ~85% digits in mini‑batches.

### 3.3 Quick eval (no training)
```bash
python expr16_train.py --eval_only --weights ./expr16_ft.pth \
  --digits_root ./acq_digits --ops_root ./acq_ops
```

---

## 4) End‑to‑End GUI (Segmentation + Inference + Evaluation)

Run the GUI:
```bash
python gui_acquisition_expr16.py --weights ./expr16_ft.pth
```
Then:

1. **Load model:** “Choose 16‑class Weights…” (required if you didn’t pass `--weights`).
2. **Load an image & predict:** Shows the image with **segmentation boxes** and a table of predicted classes + confidence.
3. **Evaluate a folder (digits‑only):**  
   - The GUI expects file names like `img_000_12345.png`. It extracts the ground truth from the trailing `_...` part.
   - For **expressions**, operators are mapped to safe letters in file names:  
     `+ -> A`, `- -> B`, `* -> C`, `/ -> D`, `(` -> L, `)` -> R`.  
     Example: `expr_12+3*(4-5)` becomes `expr_12A3C(4B5R).png` → safe label `12A3CL4B5R`.
   - The evaluator **filters to digits** to compute digits‑only sequence accuracy.
4. **Compose synthetic data (digits batch):**  
   - Choose `acq_digits` root, length range (e.g., 3–4), overlaps, spacing, and N.  
   - Saves `img_{idx}_{GT}.png` into a parameter‑stamped folder.
5. **Compose ONE expression (digits + operators):**  
   - Requires both `acq_digits` and `acq_ops`.  
   - Controls: number of terms, chance of parentheses and two‑digit numbers, overlap/spacing.  
   - The preview can be saved (file name includes both human label and **safe** label).

### 4.1 Segmentation presets & knobs
Presets are available in **Segmentation Hyperparameters**:
- **Default (report):** balanced
- **Tight / Overlapped:** aggressive split for touching digits
- **Well‑spaced / Conservative:** avoids over‑split
- **Aggressive Split:** strongest splitting

Key parameters (what to tweak when):
- `seg_score_floor` ↓ → more candidate cuts (helps **undersplit**).  
- `seg_nms_px` ↓ and `seg_min_gap_px` ↓ → allow denser cuts (helps **undersplit**).  
- `seg_max_span_factor` ↓ → prefer shorter spans (helps **undersplit**).  
- `per_cut_bias` ↓ or slightly negative → encourage adding cuts.  
- `recur_max_depth` ↑ and `recur_gain_thr` ↓/negative → deeper recursive fixes.

> If your predictions look right on single glyphs but sequences fail, try **“Tight / Overlapped”** then re‑run.

---

## 5) Best‑Practice Tips

- **Digits sequence accuracy** multiplies per‑glyph errors. Even +1% per‑digit often yields +5–10% sequence accuracy on long strings.
- Keep training **augments** strong for operators (jitter, thickness, shear), but use **digit‑weighted sampling** or **digit‑first fine‑tune** for better digit stability.
- For evaluation, use the **same preprocessing** pipeline: auto‑invert → /255 → resize 28×28.
- If you plan to deploy both “expression‑heavy” and “digit‑only” use‑cases, you can keep two heads or two checkpoints and switch in the GUI.

---

## 6) Troubleshooting

- **`UnidentifiedImageError`** during training: non‑image files in your folders. The loader now filters by extension; corrupt images still raise a clear error with the file path. Remove/replace those files.
- **“This GUI requires a 16‑class model…”**: you loaded a 10‑class checkpoint. Train/export with `num_classes=16` (this repo’s default).
- **Undersplit (missing cuts)**: use **Tight / Overlapped** preset and/or lower `seg_score_floor`, `seg_nms_px`, `seg_min_gap_px`; lower `per_cut_bias`; increase `recur_max_depth` and make `recur_gain_thr` ≤ 0.
- **Digits‑only accuracy lags**: run the **1‑epoch digit‑first fine‑tune** (`--ft_digits_focus 0.85`, `--ft_lr 1e-4`).

---

## 7) Reproducibility

We use Python’s and NumPy’s RNGs and random shuffles; results may vary slightly by run. For stricter reproducibility, set global seeds and disable non‑deterministic CuDNN features (may reduce speed).

---

## 8) Reference Commands (copy‑paste)

**Train + fine‑tune (PowerShell):**
```powershell
python .\expr16_train.py `
  --digits_root .\acq_digits `
  --ops_root .\acq_ops `
  --epochs 12 `
  --batch 256 `
  --out .\expr16_clean.pth `
  --ft_epochs 1 `
  --ft_lr 1e-4 `
  --ft_digits_focus 0.85
```

**Eval only (bash):**
```bash
python expr16_train.py --eval_only --weights ./expr16_ft.pth \
  --digits_root ./acq_digits --ops_root ./acq_ops
```

**Run GUI (with weights):**
```bash
python gui_acquisition_expr16.py --weights ./expr16_ft.pth
```

---

## 9) File Map

- `expr16_train.py` — training, eval, and optional **digit‑first fine‑tune**.
- `gui_acquisition_expr16.py` — end‑to‑end acquisition, segmentation, inference, evaluation, and data composers.
- `report_metric_hybrid.py` — segmentation functions (**required by GUI**):
  - Must expose `segment_by_psc_decode(arr_u8, model, device, **kwargs)`
  - Optional: `_shear_min_cov(arr_u8, max_abs=0.45)` for deskew before segmentation.
