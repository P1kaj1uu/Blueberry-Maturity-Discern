# -*- coding: utf-8 -*-
"""论文用数据集统计图。

输出 (300 dpi, Times New Roman 风格, 蓝紫色系):
  figs/dataset/fig_dataset_overview.png   3x2 综合面板
  figs/dataset/fig_class_dist.png         类别目标数
  figs/dataset/fig_boxes_per_image.png    每图目标数
  figs/dataset/fig_bbox_size.png          框尺寸散点+面积直方
  figs/dataset/fig_bbox_center_heatmap.png 中心位置热图
  figs/dataset/fig_bbox_ar.png            长宽比
  figs/dataset/fig_examples.png           带 GT 框的真实图像示例
"""
import os
import glob
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle
from PIL import Image
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = os.path.join(ROOT, "dataset")
OUTD = os.path.join(ROOT, "figs", "dataset")
os.makedirs(OUTD, exist_ok=True)

# ---- 论文风格设置 ----
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})
# 蓝紫色系（蓝莓主题）
C_MAIN = "#4C6EF5"
C_ACC = "#845EF7"
C_WARM = "#F783AC"
C_GREEN = "#69DB7C"
C_GRAY = "#868E96"
SPLIT_COLORS = {"train": "#4C6EF5", "valid": "#845EF7", "test": "#F783AC"}


def load_yaml_names():
    p = os.path.join(DS, "data.yaml")
    names = []
    with open(p, encoding="utf-8") as f:
        txt = f.read()
    import re
    m = re.search(r"names:\s*(\[.*?\])", txt)
    if m:
        names = json.loads(m.group(1))
    return names


def collect():
    names = load_yaml_names()
    rows = []  # dict: split, img, cls, w, h, cx, cy, iw, ih
    splits = ["train", "valid", "test"]
    for split in splits:
        img_dir = os.path.join(DS, split, "images")
        if not os.path.isdir(img_dir):
            continue
        seen = set()
        for imp in (glob.glob(os.path.join(img_dir, "*.jpg")) +
                    glob.glob(os.path.join(img_dir, "*.png")) +
                    glob.glob(os.path.join(img_dir, "*.JPG")) +
                    glob.glob(os.path.join(img_dir, "*.jpeg"))):
            key = os.path.normcase(imp)
            if key in seen:
                continue
            seen.add(key)
            lbp = os.path.splitext(imp)[0].replace("images", "labels") + ".txt"
            with Image.open(imp) as im:
                iw, ih = im.size
            if not os.path.exists(lbp):
                continue
            with open(lbp) as f:
                lines = [l.strip() for l in f if l.strip()]
            if not lines:
                continue
            for line in lines:
                c, cx, cy, w, h = map(float, line.split())
                rows.append(dict(split=split, img=imp, cls=int(c),
                                 w=w, h=h, cx=cx, cy=cy, iw=iw, ih=ih))
    return names, rows


def fig_class_dist(names, rows):
    cnt = Counter(r["cls"] for r in rows)
    labels = [f"{names[i]}\n(cls {i})" if i < len(names) else f"cls {i}"
              for i in range(max(len(names), max(cnt) + 1))]
    vals = [cnt.get(i, 0) for i in range(len(labels))]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    bars = ax.bar(range(len(vals)), vals, color=[C_MAIN, C_ACC, C_WARM][:len(vals)],
                  edgecolor="white", linewidth=0.8, width=0.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, str(v),
                ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Number of instances")
    ax.set_ylim(0, max(vals) * 1.15)
    ax.set_title("(a) Instance count per class", loc="left")
    fig.savefig(os.path.join(OUTD, "fig_class_dist.png"))
    plt.close(fig)


def fig_boxes_per_image(names, rows):
    per = defaultdict(list)
    for r in rows:
        per[r["split"]].append(r)
    data = defaultdict(list)
    for split, rs in per.items():
        cnt = Counter(r["img"] for r in rs)
        data[split] = list(cnt.values())
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    bp = ax.boxplot([data[s] for s in ("train", "valid", "test")],
                    patch_artist=True, widths=0.55, showfliers=True,
                    flierprops=dict(marker="o", markersize=3, alpha=0.5,
                                    markerfacecolor=C_GRAY, markeredgecolor="none"))
    for patch, s in zip(bp["boxes"], ("train", "valid", "test")):
        patch.set_facecolor(SPLIT_COLORS[s])
        patch.set_alpha(0.75)
    for med in bp["medians"]:
        med.set_color("black")
    ax.set_xticklabels(["Train", "Valid", "Test"])
    ax.set_ylabel("Instances per image")
    ax.set_title("(b) Instances per image", loc="left")
    fig.savefig(os.path.join(OUTD, "fig_boxes_per_image.png"))
    plt.close(fig)


def fig_bbox_size(names, rows):
    ws = np.array([r["w"] * r["iw"] for r in rows])
    hs = np.array([r["h"] * r["ih"] for r in rows])
    areas = ws * hs
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.0))
    # 散点(抽样子集)
    rng = np.random.default_rng(7)
    idx = rng.choice(len(ws), size=min(4000, len(ws)), replace=False)
    ax[0].scatter(ws[idx], hs[idx], s=8, alpha=0.35, c=C_MAIN, edgecolors="none")
    lim = max(ws.max(), hs.max()) * 1.05
    ax[0].plot([0, lim], [0, lim], ls="--", lw=1, color=C_GRAY)
    ax[0].set_xlabel("Box width (px)")
    ax[0].set_ylabel("Box height (px)")
    ax[0].set_xlim(0, lim); ax[0].set_ylim(0, lim)
    ax[0].set_title("(c) Box size distribution", loc="left")
    ax[1].hist(areas, bins=40, color=C_ACC, alpha=0.8, edgecolor="white", linewidth=0.4)
    ax[1].axvline(np.median(areas), color=C_WARM, ls="--", lw=1.5,
                  label=f"median = {np.median(areas):.0f} px²")
    ax[1].set_xlabel("Box area (px²)")
    ax[1].set_ylabel("Count")
    ax[1].set_title("(d) Box area distribution", loc="left")
    ax[1].legend(frameon=False)
    fig.savefig(os.path.join(OUTD, "fig_bbox_size.png"))
    plt.close(fig)


def fig_bbox_center_heatmap(names, rows):
    xs = np.array([r["cx"] for r in rows])
    ys = np.array([r["cy"] for r in rows])
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    hb = ax.hexbin(xs, ys, gridsize=32, cmap="Purples",
                   extent=(0, 1, 0, 1), mincnt=1)
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label("Instances")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Normalized x (center)")
    ax.set_ylabel("Normalized y (center)")
    ax.set_title("(e) Object center heatmap", loc="left")
    fig.savefig(os.path.join(OUTD, "fig_bbox_center_heatmap.png"))
    plt.close(fig)


def fig_bbox_ar(names, rows):
    ar = np.array([r["w"] / max(r["h"], 1e-6) for r in rows])
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.hist(ar, bins=40, color=C_MAIN, alpha=0.8, edgecolor="white", linewidth=0.4)
    ax.axvline(np.median(ar), color=C_WARM, ls="--", lw=1.5,
               label=f"median = {np.median(ar):.2f}")
    ax.set_xlabel("Aspect ratio (w / h)")
    ax.set_ylabel("Count")
    ax.set_title("(f) Aspect ratio distribution", loc="left")
    ax.legend(frameon=False)
    fig.savefig(os.path.join(OUTD, "fig_bbox_ar.png"))
    plt.close(fig)


def fig_examples(names, rows, n_cols=4, n_rows=2, seed=1):
    rng = np.random.default_rng(seed)
    # 按类别挑图：确保覆盖各成熟度类别
    per_img = {}
    for r in rows:
        per_img.setdefault(r["img"], []).append(r)
    imgs = sorted(per_img.keys())
    rng.shuffle(imgs)
    picked = []
    need = set(range(len(names)))
    for im in imgs:
        cls = {r["cls"] for r in per_img[im]}
        if cls & need:
            picked.append(im)
            need -= cls
            if not need:
                break
    # 补足数量
    for im in imgs:
        if len(picked) >= n_cols * n_rows:
            break
        if im not in picked:
            picked.append(im)
    picked = picked[:n_cols * n_rows]

    cmap = ["#2B8A3E", "#E8590C", "#1971C2", "#9C36B5"]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 3.1 * n_rows))
    for ax, imp in zip(np.ravel(axes), picked):
        with Image.open(imp) as im:
            arr = np.asarray(im.convert("RGB"))
        ax.imshow(arr)
        iw, ih = arr.shape[1], arr.shape[0]
        for r in per_img[imp]:
            x = (r["cx"] - r["w"] / 2) * iw
            y = (r["cy"] - r["h"] / 2) * ih
            w, h = r["w"] * iw, r["h"] * ih
            col = cmap[r["cls"] % len(cmap)]
            ax.add_patch(Rectangle((x, y), w, h, fill=False,
                                   edgecolor=col, lw=1.1))
        ax.set_xticks([]); ax.set_yticks([])
    handles = [plt.Line2D([0], [0], color=cmap[i % len(cmap)], lw=2,
                          label=names[i] if i < len(names) else f"cls {i}")
               for i in range(len(names))]
    fig.legend(handles=handles, loc="lower center", ncol=len(names),
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.savefig(os.path.join(OUTD, "fig_examples.png"))
    plt.close(fig)


def fig_dataset_overview(names, rows):
    """综合面板: 类别+每图目标+中心热图+尺寸散点(4合一)"""
    per = defaultdict(list)
    for r in rows:
        per[r["split"]].append(r)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2))
    # (a) 类别
    cnt = Counter(r["cls"] for r in rows)
    labels = [names[i] if i < len(names) else f"cls{i}" for i in range(len(names))]
    vals = [cnt.get(i, 0) for i in range(len(names))]
    bars = axes[0, 0].bar(range(len(vals)), vals, color=[C_MAIN, C_ACC, C_WARM][:len(vals)],
                          edgecolor="white", linewidth=0.8, width=0.6)
    for b, v in zip(bars, vals):
        axes[0, 0].text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, str(v),
                        ha="center", va="bottom", fontsize=10)
    axes[0, 0].set_xticks(range(len(vals)))
    axes[0, 0].set_xticklabels(labels)
    axes[0, 0].set_ylabel("Instances")
    axes[0, 0].set_ylim(0, max(vals) * 1.15)
    axes[0, 0].set_title("(a) Instances per class", loc="left")
    # (b) 每图目标数
    data = {s: [len([x for x in rs if x["img"] == im])
                for im in {r["img"] for r in rs}]
            for s, rs in per.items()}
    bp = axes[0, 1].boxplot([data[s] for s in ("train", "valid", "test")],
                            patch_artist=True, widths=0.55, showfliers=False)
    for patch, s in zip(bp["boxes"], ("train", "valid", "test")):
        patch.set_facecolor(SPLIT_COLORS[s]); patch.set_alpha(0.75)
    for med in bp["medians"]:
        med.set_color("black")
    axes[0, 1].set_xticklabels(["Train", "Valid", "Test"])
    axes[0, 1].set_ylabel("Instances / image")
    axes[0, 1].set_title("(b) Instances per image", loc="left")
    # (c) 中心热图
    xs = np.array([r["cx"] for r in rows]); ys = np.array([r["cy"] for r in rows])
    hb = axes[1, 0].hexbin(xs, ys, gridsize=28, cmap="Purples", extent=(0, 1, 0, 1), mincnt=1)
    fig.colorbar(hb, ax=axes[1, 0], pad=0.02)
    axes[1, 0].set_xlim(0, 1); axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_xlabel("Normalized center x")
    axes[1, 0].set_ylabel("Normalized center y")
    axes[1, 0].set_title("(c) Center heatmap", loc="left")
    # (d) 尺寸散点(像素)
    ws = np.array([r["w"] * r["iw"] for r in rows])
    hs = np.array([r["h"] * r["ih"] for r in rows])
    rng = np.random.default_rng(7)
    idx = rng.choice(len(ws), size=min(4000, len(ws)), replace=False)
    axes[1, 1].scatter(ws[idx], hs[idx], s=8, alpha=0.35, c=C_MAIN, edgecolors="none")
    lim = max(ws.max(), hs.max()) * 1.05
    axes[1, 1].plot([0, lim], [0, lim], ls="--", lw=1, color=C_GRAY)
    axes[1, 1].set_xlim(0, lim); axes[1, 1].set_ylim(0, lim)
    axes[1, 1].set_xlabel("Box width (px)")
    axes[1, 1].set_ylabel("Box height (px)")
    axes[1, 1].set_title("(d) Box size distribution", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTD, "fig_dataset_overview.png"))
    plt.close(fig)


def main():
    names, rows = collect()
    if not rows:
        print("no annotations found, run coco_to_yolo.py first")
        return
    print(f"names={names}  instances={len(rows)}")
    fig_dataset_overview(names, rows)
    fig_class_dist(names, rows)
    fig_boxes_per_image(names, rows)
    fig_bbox_size(names, rows)
    fig_bbox_center_heatmap(names, rows)
    fig_bbox_ar(names, rows)
    fig_examples(names, rows)
    print("figures ->", OUTD)


if __name__ == "__main__":
    main()
