# -*- coding: utf-8 -*-
"""汇总各实验最终指标 -> 论文结果表。

用法: python analysis/make_results_table.py [out.md]
"""
import os
import glob
import csv
import sys
from collections import OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")

# (目录名, 论文名称, 组别)
EXPS = [
    ("baseline", "YOLOv8n (baseline)", "w/o pretrain"),
    ("base_wiou", "+ WIoUv3", "w/o pretrain"),
    ("bifpn", "+ BiFPN-Concat", "w/o pretrain"),
    ("p2", "+ P2", "w/o pretrain"),
    ("ca", "+ C2f-CA (neg. result)", "w/o pretrain"),
    ("bcmlite", "BCM-Lite (Ours)", "w/o pretrain"),
    ("bcm", "BCM full (neg. result)", "w/o pretrain"),
    ("baseline_pt", "YOLOv8n", "pretrain"),
    ("p2_pt", "+ P2", "pretrain"),
    ("bcm_pt_v3", "BCM full (CIoU)", "pretrain"),
]

HEADERS = ["Method", "Pretrain", "P", "R", "mAP50", "mAP50-95"]


def load_best(name):
    pat = os.path.join(RUNS, "**", name, "results.csv")
    hits = glob.glob(pat, recursive=True)
    if not hits:
        return None
    with open(hits[0]) as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "analysis", "RESULTS.md")
    lines = ["| " + " | ".join(HEADERS) + " |",
             "|" + "---|" * len(HEADERS)]
    for name, label, grp in EXPS:
        r = load_best(name)
        if r is None:
            print(f"[skip] {name}")
            continue
        g = lambda k: float(r[k]) if r.get(k, "") else 0.0
        lines.append(f"| {label} | {grp} | {g('metrics/precision(B)'):.3f} | "
                     f"{g('metrics/recall(B)'):.3f} | {g('metrics/mAP50(B)'):.3f} | "
                     f"{g('metrics/mAP50-95(B)'):.3f} |")
    with open(out, "w", encoding="utf-8") as f:
        f.write("# 实验结果汇总（真实数据）\n\n")
        f.write("- 数据集: Blueberry maturity (Ni et al. 2020), train 366 / valid 104 / test 54\n")
        f.write("- 输入 416×416, 30-60 epochs, AdamW, seed=42, CPU (i7-14700KF)\n\n")
        f.write("\n".join(lines) + "\n")
    print("written:", out)
    for l in lines:
        print(l)


if __name__ == "__main__":
    main()
