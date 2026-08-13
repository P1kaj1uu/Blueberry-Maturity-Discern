# -*- coding: utf-8 -*-
"""验证转换后的 YOLO 数据集: 类别分布/尺寸/空标注。"""
import os, glob, sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = os.path.join(ROOT, "dataset")
cls_cnt = Counter()
empty = 0
total = 0
for split in ["train", "valid", "test"]:
    lb = os.path.join(DS, split, "labels")
    n_img = len(glob.glob(os.path.join(DS, split, "images", "*")))
    n_lbl = 0
    e = 0
    for p in glob.glob(os.path.join(lb, "*.txt")):
        with open(p) as f:
            lines = [l for l in f if l.strip()]
        n_lbl += len(lines)
        total += len(lines)
        if not lines:
            e += 1
        for l in lines:
            cls_cnt[int(l.split()[0])] += 1
    empty += e
    print(f"{split}: images={n_img} boxes={n_lbl} empty_txt={e}")
print("class dist:", dict(cls_cnt), "total:", total, "empty:", empty)
with open(os.path.join(DS, "data.yaml")) as f:
    print(f.read())
