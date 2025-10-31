# export_mnist_test_to_folders.py
# Exports torchvision MNIST test set to class folders: out/0, out/1, ..., out/9
# Images are 28x28 grayscale PNGs, white-on-black (same as MNIST).
import os, argparse
from PIL import Image
import numpy as np
import torchvision
import torchvision.transforms as T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="acq_digits", help="root output folder")
    ap.add_argument("--limit", type=int, default=0, help="optional cap per class (0 = all)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for d in range(10):
        os.makedirs(os.path.join(args.out, str(d)), exist_ok=True)

    ds = torchvision.datasets.MNIST(root="./data", train=False,
                                    transform=T.ToTensor(), download=True)

    counts = [0]*10
    for i in range(len(ds)):
        img, label = ds[i]            # img: [1,28,28], 0..1
        arr = (img.squeeze(0).numpy()*255).astype(np.uint8)  # white-on-black
        if args.limit and counts[label] >= args.limit:
            continue
        out_path = os.path.join(args.out, str(label), f"img_{counts[label]:05d}.png")
        Image.fromarray(arr, mode="L").save(out_path)
        counts[label] += 1

    print("Exported per-class counts:", counts)
    print(f"Saved under: {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()
