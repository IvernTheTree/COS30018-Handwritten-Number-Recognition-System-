# make_ops_dataset.py
import os, random, math
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps, ImageFilter

ROOT = "acq_ops"         # change if you like
PER_CLASS = 300          # how many samples for each operator
SIZE = 28                # 28x28 like MNIST

OPS = {
    "plus":   "+",
    "minus":  "-",
    "times":  "*",
    "divide": "/",
    "lparen": "(",
    "rparen": ")",
}

def _blank():
    return Image.new("L", (SIZE, SIZE), color=0)

def _rand(a, b): return random.uniform(a, b)
def _clip(x, lo, hi): return max(lo, min(hi, x))

def _thick(draw, kind, pts, w):
    # draw thicker by over-drawing ±1px offsets
    if kind == "line":
        (x0,y0,x1,y1) = pts
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                draw.line((x0+dx,y0+dy,x1+dx,y1+dy), fill=255, width=w)
    elif kind == "arc":
        (bbox, start, end) = pts
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                bb = [bbox[0]+dx, bbox[1]+dy, bbox[2]+dx, bbox[3]+dy]
                draw.arc(bb, start=start, end=end, fill=255, width=w)

def draw_plus(img):
    d = ImageDraw.Draw(img)
    cx, cy = _rand(12,16), _rand(12,16)
    L = _rand(8, 11)
    w = int(_clip(_rand(2.2, 3.8), 2, 5))
    _thick(d, "line", (cx-L, cy, cx+L, cy), w)
    _thick(d, "line", (cx, cy-L, cx, cy+L), w)

def draw_minus(img):
    d = ImageDraw.Draw(img)
    y  = _rand(12, 16)
    x0 = _rand(6, 9); x1 = _rand(18, 22)
    w = int(_clip(_rand(2.2, 3.8), 2, 5))
    _thick(d, "line", (x0, y, x1, y), w)

def draw_times(img):
    d = ImageDraw.Draw(img)
    cx, cy = _rand(12,16), _rand(12,16)
    L = _rand(9, 12)
    w = int(_clip(_rand(2.0, 3.5), 2, 5))
    _thick(d, "line", (cx-L, cy-L, cx+L, cy+L), w)
    _thick(d, "line", (cx-L, cy+L, cx+L, cy-L), w)

def draw_divide(img):
    d = ImageDraw.Draw(img)
    y  = _rand(12, 16)
    x0 = _rand(5, 8); x1 = _rand(19, 23)
    w  = int(_clip(_rand(2.0, 3.2), 2, 4))
    _thick(d, "line", (x0, y, x1, y), w)
    # dots
    r = _rand(2.3, 3.2)
    for yy in (y-7, y+7):
        d.ellipse(( (x0+x1)/2-r, yy-r, (x0+x1)/2+r, yy+r ), fill=255)

def draw_lparen(img):
    d = ImageDraw.Draw(img)
    x = _rand(9, 12)
    y0 = _rand(5, 7); y1 = _rand(20, 22)
    w = int(_clip(_rand(2.0, 3.5), 2, 4))
    _thick(d, "arc", ([x-8, y0, x+8, y1], 70, 290), w)

def draw_rparen(img):
    d = ImageDraw.Draw(img)
    x = _rand(16, 19)
    y0 = _rand(5, 7); y1 = _rand(20, 22)
    w = int(_clip(_rand(2.0, 3.5), 2, 4))
    _thick(d, "arc", ([x-8, y0, x+8, y1], 250, 110), w)

DRAWERS = {
    "plus":   draw_plus,
    "minus":  draw_minus,
    "times":  draw_times,
    "divide": draw_divide,
    "lparen": draw_lparen,
    "rparen": draw_rparen,
}

def jitter(img):
    # small affine jitter + light blur to feel more handwritten
    angle = _rand(-10, 10)
    img = img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=0)
    dx, dy = int(_rand(-1.5,1.5)), int(_rand(-1.5,1.5))
    img = ImageOps.expand(img, border=2, fill=0).crop((2-dx,2-dy,30-dx,30-dy))
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(_rand(0.2, 0.6)))
    return img

def main():
    Path(ROOT).mkdir(parents=True, exist_ok=True)
    for cls in OPS.keys():
        Path(ROOT, cls).mkdir(parents=True, exist_ok=True)

    for cls, fn in DRAWERS.items():
        out_dir = Path(ROOT, cls)
        count = len(list(out_dir.glob("*.png")))
        target = max(0, PER_CLASS - count)
        if target == 0:
            print(f"[skip] {cls}: already has {count} files")
            continue
        print(f"[gen] {cls}: +{target} files")
        for i in range(target):
            img = _blank()
            fn(img)
            img = jitter(img)
            img.save(out_dir / f"{cls}_{i:04d}.png")

    print(f"Done. Created/updated operator set under: {Path(ROOT).resolve()}")

if __name__ == "__main__":
    main()
