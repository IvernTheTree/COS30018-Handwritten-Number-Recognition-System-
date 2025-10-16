# ============================
# MNIST CNN + Projection "Valley" Segmentation Pipeline
# ============================

import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import torch.nn.functional as F
import math

# --------------------------
# Reproducibility
# --------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

#set_seed(1337)  # <-- comment this out if you want different random canvases each run


# --------------------------
# Overlapped digits generator
# --------------------------
def create_segments(mnist_dataset, num_digits=3,
                    overlap_range=(6, 12),      # pixels of overlap between adjacent digits
                    spacing=0,                 # tiny gap after overlap
                    left_pad=10, right_pad=10, # margins
                    blend='max',               # 'max' or 'add' (clipped)
                    crop_threshold=0.1,        # crop columns that contain content
                    max_overlap_frac=0.12,     # cap overlap to 12% of the narrower digit
                    p_no_overlap=0.25):        # sometimes no overlap at all
    """
    Build a single-row canvas of digits with intentional horizontal overlap.
    Returns:
        canvas: (H, W) float32 image in [0,1]
        labels: list of digit labels
        true_segments: list of (start_col, end_col) where each original digit was placed
    """
    digits = []
    labels = []

    idxs = random.sample(range(len(mnist_dataset)), num_digits)
    for idx in idxs:
        img, label = mnist_dataset[idx]           # img = Tensor [1,28,28] in [0,1]
        img_np = img.squeeze().numpy()            # (28, 28)
        cols = np.where(img_np.max(axis=0) > crop_threshold)[0]
        if len(cols) == 0:
            continue
        start, end = cols[0], cols[-1]
        content = img_np[:, start:end+1]          # (28, w)
        digits.append(content.astype(np.float32))
        labels.append(label)

    if not digits:
        raise RuntimeError("No digits selected contained content.")

    overlaps = []
    if len(digits) > 1:
        lo_px, hi_px = overlap_range
        for i in range(len(digits) - 1):
            cap = int(max_overlap_frac * min(digits[i].shape[1], digits[i+1].shape[1]))
            cap = max(1, cap)
            upper = min(hi_px, cap)
            lower = min(lo_px, upper)
            ov = random.randint(lower, upper)
            if random.random() < p_no_overlap:
                ov = 0
            overlaps.append(ov)

    widths = [d.shape[1] for d in digits]
    total_width = left_pad + sum(widths) - sum(overlaps) + spacing * (len(digits) - 1) + right_pad
    canvas = np.zeros((28, total_width), dtype=np.float32)

    true_segments = []
    x = left_pad
    for i, content in enumerate(digits):
        w = content.shape[1]
        if x + w > canvas.shape[1]:
            pad_w = (x + w) - canvas.shape[1]
            canvas = np.pad(canvas, ((0, 0), (0, pad_w)), mode='constant')

        region = canvas[:, x:x+w]
        if blend == 'add':
            canvas[:, x:x+w] = np.clip(region + content, 0.0, 1.0)
        else:
            canvas[:, x:x+w] = np.maximum(region, content)

        true_segments.append((x, x + w))

        if i < len(digits) - 1:
            x = x + w - overlaps[i] + spacing

    return canvas, labels, true_segments


# --------------------------
# Projection "valley" segmentation
# --------------------------
def segment_by_valleys(
    image,
    threshold=0.1,        # binarization cutoff (image assumed in [0,1])
    smooth_sigma=1.5,     # Gaussian smoothing for the projection (higher = smoother, fewer cuts)
    min_prom=3,           # valley prominence; raise to be stricter about splits
    min_distance=2,       # minimum pixel distance between valleys
    min_width=4,          # minimum segment width (columns)
    min_area=25           # minimum pixel area in the binary mask to keep a segment
):
    """
    Split touching digits by finding local minima (valleys) in the smoothed vertical projection.
    Returns:
        segmented_digits: list of grayscale crops (H, w_i)
        segments:         list of (x0, x1) column spans per component (left->right)
        vertical_projection: np.ndarray of column sums of the binary mask
        binary_image:     np.uint8 mask used for projection (0/1)
    """
    # 1) Binary mask and vertical projection
    binary_image = (image > threshold).astype(np.uint8)
    vertical_projection = binary_image.sum(axis=0).astype(np.float32)
    W = binary_image.shape[1]

    if vertical_projection.sum() == 0:
        return [], [], vertical_projection, binary_image

    # 2) Smooth projection to stabilize valleys
    vp_s = gaussian_filter1d(vertical_projection, sigma=smooth_sigma)

    # 3) Find valleys as peaks on the inverted signal
    inv = vp_s.max() - vp_s
    valley_idx, _ = find_peaks(inv, prominence=min_prom, distance=min_distance)

    # 4) Turn valleys into cut positions (add image borders)
    cuts = [0] + valley_idx.tolist() + [W]

    # 5) Build segments between cuts; refine to actual ink span; filter tiny pieces
    segments, segmented_digits = [], []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a < min_width:
            continue

        # refine [a,b) to columns that actually contain ink to avoid blank margins
        col_sums = binary_image[:, a:b].sum(axis=0)
        nz = np.where(col_sums > 0)[0]
        if nz.size == 0:
            continue
        aa = a + int(nz[0])
        bb = a + int(nz[-1]) + 1
        if (bb - aa) < min_width:
            continue

        area = int(binary_image[:, aa:bb].sum())
        if area < min_area:
            continue

        segments.append((aa, bb))
        segmented_digits.append(image[:, aa:bb])

    # Left-to-right ordering (usually already ordered)
    if segments:
        order = np.argsort([s[0] for s in segments])
        segments = [segments[i] for i in order]
        segmented_digits = [segmented_digits[i] for i in order]

    return segmented_digits, segments, vertical_projection, binary_image


# --------------------------
# Visualiser
# --------------------------
def visualize(multi_digit_image, binary_image, vertical_projection,
              segmented_digits, true_labels, true_segments=None, found_segments=None,
              preds=None):
    # rows = 1 header row + enough rows to show all segments (3 per row)
    n = len(segmented_digits)
    digit_rows = max(1, math.ceil(n / 3))   # at least one row for consistency
    rows = 1 + digit_rows
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows))

    # If rows==2, axes is 2D; keep indexing uniform
    if rows == 2:
        axes = np.array(axes).reshape(rows, 3)

    # -------- first row: overview panels --------
    axes[0, 0].imshow(multi_digit_image, cmap='gray')
    axes[0, 0].set_title(f'1. Original Image\nSelected Digits: {true_labels}')
    axes[0, 0].axis('off')
    if true_segments:
        for s, e in true_segments:
            axes[0, 0].axvline(s, linestyle='--', linewidth=1)
            axes[0, 0].axvline(e, linestyle='--', linewidth=1)

    axes[0, 1].imshow(binary_image, cmap='gray')
    axes[0, 1].set_title('2. Binary Image\n(After thresholding)')
    axes[0, 1].axis('off')

    axes[0, 2].plot(vertical_projection)
    axes[0, 2].set_title('3. Vertical Projection')
    axes[0, 2].set_xlabel('Column (x-position)')
    axes[0, 2].set_ylabel('Sum of pixels')
    axes[0, 2].grid(True)
    if true_segments:
        for s, e in true_segments:
            axes[0, 2].axvspan(s, e, alpha=0.08)
    if found_segments:
        for s, e in found_segments:
            axes[0, 2].axvline(s, linestyle='--', linewidth=1)
            axes[0, 2].axvline(e, linestyle='--', linewidth=1)

    # -------- remaining rows: all segmented digits (3 per row) --------
    # place digit i at row = 1 + i//3, col = i%3
    for i, digit in enumerate(segmented_digits):
        r = 1 + i // 3
        c = i % 3
        axes[r, c].imshow(digit, cmap='gray')
        title = f'Digit {i+1}'
        if preds is not None and i < len(preds):
            title += f' → pred: {preds[i]}'
        axes[r, c].set_title(title)
        axes[r, c].axis('off')

    # turn off any unused cells in the last row
    last_row_used = (n % 3) if (n % 3) != 0 else 3
    for c in range(last_row_used, 3):
        axes[rows-1, c].axis('off')

    plt.tight_layout()
    plt.show()

# --------------------------
# CNN Model
# --------------------------
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# --------------------------
# Data loaders (normalized for CNN)
# --------------------------
MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)

def get_dataloaders(batch_size=128):
    tf_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD)
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD)
    ])
    train_set = torchvision.datasets.MNIST(root='../data', train=True, transform=tf_train, download=True)
    test_set  = torchvision.datasets.MNIST(root='../data', train=False, transform=tf_test,  download=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader

def get_raw_testset():  # For building overlapped canvas (no normalization needed)
    return torchvision.datasets.MNIST(root='../data', train=False, transform=transforms.ToTensor(), download=True)


# --------------------------
# Training / Evaluation
# --------------------------
def train_one_epoch(model, loader, opt, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()

        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


# --------------------------
# Helpers: pad/crop segment to 28x28 and normalize for model
# --------------------------
def segment_to_28x28_tensor(seg_np):
    """
    seg_np: numpy array (28, W) in [0,1]
    Returns a torch tensor [1,1,28,28] normalized with MNIST stats.
    """
    H, W = seg_np.shape
    assert H == 28, "Expected segment height 28."
    if W > 28:
        # center crop width to 28 if weirdly wide (rare)
        start = (W - 28) // 2
        seg_np = seg_np[:, start:start+28]
        W = 28

    # center-pad to width 28
    pad_left = (28 - W) // 2
    pad_right = 28 - W - pad_left
    padded = np.pad(seg_np, ((0, 0), (pad_left, pad_right)), mode='constant')

    # to tensor + normalize
    t = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).float()  # [1,1,28,28]
    # scale already [0,1]; normalize:
    t = (t - MNIST_MEAN[0]) / MNIST_STD[0]
    return t

@torch.no_grad()
def classify_segments(model, segments_list, device):
    """
    segments_list: list of numpy arrays (28, W) in [0,1]
    Returns: list of predicted ints
    """
    model.eval()
    preds = []
    for seg in segments_list:
        t = segment_to_28x28_tensor(seg).to(device)
        logits = model(t)
        pred = int(torch.argmax(logits, dim=1).item())
        preds.append(pred)
    return preds


# --------------------------
# Main
# --------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Train the CNN quickly
    train_loader, test_loader = get_dataloaders(batch_size=128)
    model = DigitCNN().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10 # bump to 5–10 for higher accuracy
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, device)
        te_loss, te_acc = evaluate(model, test_loader, device)
        print(f"[Epoch {ep:02d}] train loss {tr_loss:.4f} acc {tr_acc:.4f} | test loss {te_loss:.4f} acc {te_acc:.4f}")

    # 2) Build a RANDOM overlapped canvas from RAW (unnormalized) test set
    raw_test = get_raw_testset()

    # --- Randomize canvas parameters here ---
    num_digits     = random.randint(2, 4)                    # 2–4 digits
    overlap_choice = random.choice([(1, 3), (4, 6), (8, 16)])# gentle→harder
    spacing_choice = random.choice([0, 1, 2, 3])             # allow zero gap
    print(f"\n[Random canvas] num_digits={num_digits}, overlap_range={overlap_choice}, spacing={spacing_choice}")

    multi_digit_image, true_labels, true_segments = create_segments(
        raw_test,
        num_digits=num_digits,
        overlap_range=overlap_choice,
        spacing=spacing_choice,
        blend='max'
    )

    # 3) Segment by valleys
    segmented_digits, segments, vertical_projection, binary_image = segment_by_valleys(
        multi_digit_image,
        threshold=0.1,
        smooth_sigma=2.0,
        min_prom=3,
        min_distance=2,
        min_width=3,
        min_area=15
    )

    # 4) Classify each segment
    preds = classify_segments(model, segmented_digits, device)

    # 5) Summary
    print("\nRESULTS SUMMARY")
    print(f"True labels (left→right placement order): {true_labels}")
    print(f"Segments found: {len(segments)}")
    print(f"Predictions on segments (left→right):     {preds}")

    # 6) Visualize with predictions on crops
    visualize(multi_digit_image, binary_image, vertical_projection,
              segmented_digits, true_labels, true_segments=true_segments,
              found_segments=segments, preds=preds)
    torch.save(model.state_dict(), "mnist_cnn.pth")


if __name__ == "__main__":
    main()
