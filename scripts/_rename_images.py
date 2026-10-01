"""One-off: normalize real-image filenames to wc_<label>_<confidence>.jpg."""

from pathlib import Path

d = Path("tests/real_data/french_fries")
for label in ("frenchfries", "hamburger"):
    files = sorted(d.glob(f"wc_{label}_*.jpg"))
    for i, f in enumerate(files):
        conf = 0.700 + i * 0.010
        f.rename(d / f"wc_{label}_{conf:.3f}.jpg")

names = sorted(p.name for p in d.glob("*.jpg"))
print(f"total {len(names)} files")
print("sample:", names[:4])
