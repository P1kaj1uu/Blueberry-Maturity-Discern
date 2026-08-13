# -*- coding: utf-8 -*-
"""统计蓝莓数据集：类别、目标数、空标注、框尺寸/位置分布、图片尺寸。"""
import os
import glob
from collections import Counter, defaultdict
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = ["train", "valid", "test"]

all_boxes = []          # (cls, w, h, cx, cy, img_w, img_h)
img_sizes = Counter()
cls_counts = Counter()
per_image_boxes = defaultdict(list)  # split -> [每图目标数]
empty_images = Counter()
label_files_total = 0
label_files_empty = 0

for split in SPLITS:
    img_dir = os.path.join(ROOT, "dataset", split, "images")
    lbl_dir = os.path.join(ROOT, "dataset", split, "labels")
    if not os.path.isdir(img_dir):
        print(f"[skip] {split} 不存在")
        continue
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    for imp in imgs:
        name = os.path.splitext(os.path.basename(imp))[0]
        lbp = os.path.join(lbl_dir, name + ".txt")
        label_files_total += 1
        with Image.open(imp) as im:
            iw, ih = im.size
        img_sizes[(iw, ih)] += 1
        if not os.path.exists(lbp) or os.path.getsize(lbp) == 0:
            empty_images[split] += 1
            label_files_empty += 1
            per_image_boxes[split].append(0)
            continue
        with open(lbp) as f:
            lines = [l.strip() for l in f if l.strip()]
        per_image_boxes[split].append(len(lines))
        for line in lines:
            c, cx, cy, w, h = map(float, line.split())
            cls_counts[int(c)] += 1
            all_boxes.append((int(c), w, h, cx, cy, iw, ih))

print("=" * 60)
print("图片尺寸分布:", dict(img_sizes))
print("类别ID计数:", dict(sorted(cls_counts.items())))
print("标签文件总数:", label_files_total, " 空标注数:", label_files_empty,
      f"({label_files_empty/max(label_files_total,1)*100:.1f}%)")
for split in SPLITS:
    if per_image_boxes[split]:
        n = len(per_image_boxes[split])
        tot = sum(per_image_boxes[split])
        print(f"  {split}: 图片 {n}, 目标 {tot}, 平均每图 {tot/n:.2f}, "
              f"空图 {empty_images[split]} ({empty_images[split]/n*100:.1f}%)")

# 框尺寸统计（相对比例）
if all_boxes:
    ws = [b[1] for b in all_boxes]
    hs = [b[2] for b in all_boxes]
    import statistics
    print("-" * 60)
    print(f"框总数: {len(all_boxes)}")
    print(f"相对宽: min={min(ws):.4f} mean={statistics.mean(ws):.4f} "
          f"p50={statistics.median(ws):.4f} max={max(ws):.4f}")
    print(f"相对高: min={min(hs):.4f} mean={statistics.mean(hs):.4f} "
          f"p50={statistics.median(hs):.4f} max={max(hs):.4f}")
    areas = [w * h * iw * ih for (c, w, h, cx, cy, iw, ih) in all_boxes]
    print(f"框像素面积: min={min(areas):.0f} p25={sorted(areas)[len(areas)//4]:.0f} "
          f"median={statistics.median(areas):.0f} p75={sorted(areas)[3*len(areas)//4]:.0f} "
          f"max={max(areas):.0f}")
