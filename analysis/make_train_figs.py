# -*- coding: utf-8 -*-
"""训练结果图。

从 runs/ 下各实验目录读取 results.csv:
  - fig_loss_curves.png    训练损失曲线 (box/cls/dfl) 各实验对比
  - fig_metric_curves.png  mAP50 / mAP50-95 随 epoch 变化
  - fig_ablation_bars.png  消融实验指标柱状图 (mAP50, mAP50-95)
  - fig_params_fps.png     参数量/FLOPs 对比 (来自模型结构)
"""
import os
import csv
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
OUTD = os.path.join(ROOT, "figs", "results")
os.makedirs(OUTD, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})
EXPS = ["baseline", "base_wiou", "ca", "bifpn", "p2", "bcmlite", "bcm"]
COLORS = {"baseline": "#868E96", "base_wiou": "#4C6EF5", "ca": "#F59F00",
          "bifpn": "#AE3EC9", "p2": "#2F9E44", "bcmlite": "#E03131", "bcm": "#212529"}
LABELS = {"baseline": "YOLOv8n", "base_wiou": "+WIoU", "ca": "+C2f-CA",
          "bifpn": "+BiFPN", "p2": "+P2", "bcmlite": "BCM-Lite (ours)", "bcm": "BCM (full)"}
ORDER = ["baseline", "base_wiou", "ca", "bifpn", "p2", "bcmlite"]


def find_results(name):
    pat = os.path.join(RUNS, "**", name, "results.csv")
    hits = glob.glob(pat, recursive=True)
    return hits[0] if hits else None


def load_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows


def get(rows, key, default=0.0):
    vals = []
    for r in rows:
        v = r.get(key, "")
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            vals.append(default)
    return np.array(vals)


def fig_loss_curves(all_data):
    keys = [("train/box_loss", "Box loss"), ("train/cls_loss", "Cls loss"),
            ("train/dfl_loss", "DFL loss"), ("val/box_loss", "Val box loss")]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8))
    for ax, (key, title) in zip(np.ravel(axes), keys):
        for name in ORDER:
            if name not in all_data:
                continue
            rows, _ = all_data[name]
            ep = get(rows, "epoch")
            y = get(rows, key)
            if len(y) == 0:
                continue
            ax.plot(ep, y, color=COLORS[name], lw=1.4, label=LABELS[name])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"({chr(97 + list(keys).index((key, title)))}) {title}", loc="left")
        ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTD, "fig_loss_curves.png"))
    plt.close(fig)
    print("saved fig_loss_curves.png")


def fig_metric_curves(all_data):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for idx, (ax, (key, title)) in enumerate(zip(axes, [("metrics/mAP50(B)", "mAP50"),
                                                        ("metrics/mAP50-95(B)", "mAP50-95")])):
        for name in ORDER:
            if name not in all_data:
                continue
            rows, _ = all_data[name]
            ep = get(rows, "epoch")
            y = get(rows, key)
            if len(y) == 0:
                continue
            ax.plot(ep, y, color=COLORS[name], lw=1.6, label=LABELS[name])
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(f"({chr(97 + idx)}) {title}", loc="left")
        ax.legend(frameon=False, ncol=3, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTD, "fig_metric_curves.png"))
    plt.close(fig)
    print("saved fig_metric_curves.png")


def fig_ablation_bars(all_data):
    f = lambda v: float(v) if not isinstance(v, (int, float)) else v
    names = [LABELS[n] for n in ORDER if n in all_data]
    m50 = [f(all_data[n][1]["metrics/mAP50(B)"]) for n in ORDER if n in all_data]
    m5095 = [f(all_data[n][1]["metrics/mAP50-95(B)"]) for n in ORDER if n in all_data]
    pr = [f(all_data[n][1]["metrics/precision(B)"]) for n in ORDER if n in all_data]
    rc = [f(all_data[n][1]["metrics/recall(B)"]) for n in ORDER if n in all_data]

    x = np.arange(len(names))
    w = 0.18
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.bar(x - 1.5 * w, m50, w, label="mAP50", color="#4C6EF5", edgecolor="white", linewidth=0.6)
    ax.bar(x - 0.5 * w, m5095, w, label="mAP50-95", color="#845EF7", edgecolor="white", linewidth=0.6)
    ax.bar(x + 0.5 * w, pr, w, label="Precision", color="#F783AC", edgecolor="white", linewidth=0.6)
    ax.bar(x + 1.5 * w, rc, w, label="Recall", color="#69DB7C", edgecolor="white", linewidth=0.6)
    for xi, v in zip(x - 1.5 * w, m50):
        ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
    for xi, v in zip(x - 0.5 * w, m5095):
        ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12)
    ax.set_ylabel("Score")
    ax.set_ylim(0, max(max(m50), max(m5095), max(pr), max(rc)) * 1.18 + 0.03)
    ax.legend(frameon=False, ncol=4, loc="upper left")
    ax.set_title("Ablation study on blueberry maturity detection (YOLOv8n, @416)", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTD, "fig_ablation_bars.png"))
    plt.close(fig)
    print("saved fig_ablation_bars.png")


def main():
    all_data = {}
    for name in EXPS:
        p = find_results(name)
        if not p:
            print(f"[skip] {name}: no results.csv")
            continue
        rows = load_csv(p)
        best = rows[-1] if rows else {}
        all_data[name] = (rows, best)
        print(f"[load] {name}: {len(rows)} epochs, "
              f"mAP50={best.get('metrics/mAP50(B)', '?'):s}, "
              f"mAP50-95={best.get('metrics/mAP50-95(B)', '?'):s}")
    if not all_data:
        print("no training results yet; run custom/batch_train.py first")
        return
    fig_loss_curves(all_data)
    fig_metric_curves(all_data)
    fig_ablation_bars(all_data)
    print("all result figures ->", OUTD)


if __name__ == "__main__":
    main()
