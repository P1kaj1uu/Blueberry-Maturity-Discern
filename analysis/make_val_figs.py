# -*- coding: utf-8 -*-
"""验证结果图: PR 曲线对比 / F1-Confidence / 混淆矩阵 / 检测可视化。

依赖: 各实验 best.pt 已训练完成。
用法: python analysis/make_val_figs.py [--testsplit test]
"""
import os
import sys
import glob
import shutil
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import custom.bbcm_modules as M
M.register()

RUNS = os.path.join(ROOT, "runs")
OUTD = os.path.join(ROOT, "figs", "results")
os.makedirs(OUTD, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})
EXPS = ["baseline", "base_wiou", "bifpn", "p2", "bcmlite"]
COLORS = {"baseline": "#868E96", "base_wiou": "#4C6EF5", "ca": "#F59F00",
          "bifpn": "#AE3EC9", "p2": "#2F9E44", "bcmlite": "#E03131", "bcm": "#212529"}
LABELS = {"baseline": "YOLOv8n", "base_wiou": "+WIoU", "ca": "+C2f-CA",
          "bifpn": "+BiFPN", "p2": "+P2", "bcmlite": "BCM-Lite (ours)", "bcm": "BCM (full)"}
ORDER = ["baseline", "base_wiou", "bifpn", "p2", "bcmlite"]
CMAP_CLS = ["#2B8A3E", "#E8590C"]   # Immature / Mature
NAMES = ["Immature", "Mature"]


def find_weights(name):
    pat = os.path.join(RUNS, "**", name, "weights", "best.pt")
    hits = glob.glob(pat, recursive=True)
    return hits[0] if hits else None


def fig_pr_curves(metrics):
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    for name in ORDER:
        if name not in metrics:
            continue
        px, py, xl, yl = metrics[name]["curves_results"][0]
        if py.ndim > 1:
            py = py.mean(axis=0)   # 多类平均 (all 曲线)
        ax.plot(px, py, color=COLORS[name], lw=1.6, label=LABELS[name])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower left", fontsize=8.5)
    ax.set_title("Precision-Recall curves (IoU=0.5)", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTD, "fig_pr_curves.png"))
    plt.close(fig)
    print("saved fig_pr_curves.png")


def fig_f1_curves(metrics):
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    for name in ORDER:
        if name not in metrics:
            continue
        px, py, xl, yl = metrics[name]["curves_results"][1]
        if py.ndim > 1:
            py = py.mean(axis=0)
        ax.plot(px, py, color=COLORS[name], lw=1.6, label=LABELS[name])
    ax.set_xlabel("Confidence")
    ax.set_ylabel("F1 score")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower left", fontsize=8.5)
    ax.set_title("F1-Confidence curves", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTD, "fig_f1_curves.png"))
    plt.close(fig)
    print("saved fig_f1_curves.png")


def fig_cm_comparison(cms):
    """对比 baseline vs bcm 的归一化混淆矩阵。"""
    names = [n for n in ("baseline", "bcm") if n in cms]
    if not names:
        return
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 4.2))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        cm = cms[name]
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                        fontsize=9, color="white" if cm[i, j] > 0.5 else "black")
        ncls = cm.shape[0]
        ticks = list(range(ncls))
        labels = NAMES[:ncls] + (["background"] if ncls > len(NAMES) else [])
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_xticklabels(labels, rotation=0)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(LABELS[name], loc="left")
    fig.colorbar(im, ax=axes.tolist(), shrink=0.85)
    fig.suptitle("Normalized confusion matrices", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTD, "fig_confusion_cmp.png"))
    plt.close(fig)
    print("saved fig_confusion_cmp.png")


def fig_detect_visual(model, split="test", n=6, seed=3):
    """真实图像 + GT 框 + 预测框对比。"""
    from PIL import Image
    import random
    from ultralytics.utils import ops
    import torch

    img_dir = os.path.join(ROOT, "dataset", split, "images")
    lbl_dir = os.path.join(ROOT, "dataset", split, "labels")
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.JPG")) + glob.glob(os.path.join(img_dir, "*.jpg")) +
                  glob.glob(os.path.join(img_dir, "*.png")))
    rng = random.Random(seed)
    rng.shuffle(imgs)
    imgs = imgs[:n]

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.0))
    for ax, imp in zip(np.ravel(axes), imgs):
        im = Image.open(imp).convert("RGB")
        arr = np.asarray(im)
        iw, ih = arr.shape[1], arr.shape[0]
        ax.imshow(arr)
        # GT
        lbp = os.path.splitext(imp)[0].replace("images", "labels") + ".txt"
        with open(lbp) as f:
            for line in f:
                c, cx, cy, w, h = map(float, line.split())
                x = (cx - w / 2) * iw; y = (cy - h / 2) * ih
                ax.add_patch(Rectangle((x, y), w * iw, h * ih, fill=False,
                                       edgecolor=CMAP_CLS[int(c)], lw=1.0, ls="-"))
        # 预测
        res = model.predict(imp, imgsz=416, conf=0.25, verbose=False)[0]
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            c = int(box.cls[0])
            conf = float(box.conf[0])
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                   edgecolor="red", lw=1.4, ls="--"))
            ax.text(x1, y1 - 4, f"{NAMES[c]} {conf:.2f}", fontsize=7.5,
                    color="red", va="bottom")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Detection on {split} split — solid: GT, dashed: prediction", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTD, "fig_detect_visual.png"))
    plt.close(fig)
    print("saved fig_detect_visual.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--models", default=",".join(ORDER))
    args = ap.parse_args()
    from ultralytics import YOLO

    metrics = {}
    cms = {}
    for name in ORDER:
        wp = find_weights(name)
        if not wp:
            print(f"[skip] {name}: no best.pt")
            continue
        m = YOLO(wp)
        res = m.val(data=os.path.join(ROOT, "dataset", "data.yaml"), imgsz=416,
                    conf=0.001, iou=0.6, split="val", plots=False, verbose=False)
        mb = res.box if hasattr(res, "box") else res.results_dict
        metrics[name] = {
            "mAP50": float(mb.mAP50) if hasattr(mb, "mAP50") else float(res.results_dict["metrics/mAP50(B)"]),
            "mAP5095": float(mb.mAP50_95) if hasattr(mb, "mAP50_95") else float(res.results_dict["metrics/mAP50-95(B)"]),
            "curves_results": mb.curves_results if hasattr(mb, "curves_results") else None,
        }
        # 混淆矩阵: DetMetrics.confusion_matrix
        cm = getattr(res, "confusion_matrix", None)
        if cm is not None and hasattr(cm, "matrix"):
            cms[name] = cm.matrix / (cm.matrix.sum(1, keepdims=True) + 1e-9)
        print(f"[val] {name}: mAP50={metrics[name]['mAP50']:.4f} mAP50-95={metrics[name]['mAP5095']:.4f}")

    fig_pr_curves(metrics)
    fig_f1_curves(metrics)
    fig_cm_comparison(cms)

    # 检测可视化 (用 BCM-Lite 模型)
    wp = find_weights("bcmlite")
    if wp:
        m = YOLO(wp)
        fig_detect_visual(m, split=args.split)
    print("all val figures ->", OUTD)


if __name__ == "__main__":
    main()
