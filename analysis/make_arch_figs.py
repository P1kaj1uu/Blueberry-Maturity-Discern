# -*- coding: utf-8 -*-
"""YOLOv8-BCM 网络结构示意图。

输出:
  figs/arch/fig_arch_bcm.png      整体结构图 (骨干+颈部+检测头)
  figs/arch/fig_c2f_ca.png        C2f-CA 模块内部结构
  figs/arch/fig_bifpn.png         BiFPN 加权融合示意
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import matplotlib.patches as mpatches

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTD = os.path.join(ROOT, "figs", "arch")
os.makedirs(OUTD, exist_ok=True)

# 配色
C_IN = "#495057"        # 输入
C_CONV = "#4C6EF5"      # 常规卷积 (蓝)
C_C2F = "#74C0FC"       # C2f (浅蓝)
C_CA = "#F59F00"        # C2f_CA (橙, 改进)
C_SPPF = "#22B8CF"      # SPPF (青)
C_BIFPN = "#AE3EC9"     # BiFPN (紫, 改进)
C_P2 = "#2F9E44"        # P2 检测头 (绿, 改进)
C_DET = "#E8590C"       # 检测头 (橙红)
C_OUT = "#212529"
C_TEXT = "#343A40"

# 层块: (x, y, w, h, 颜色, 标签)
# 坐标系统: 0-16 宽, 0-9 高


def draw_block(ax, x, y, w, h, color, label, fs=9, lw=1.2, ec="white", zorder=3):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.06",
                         linewidth=lw, edgecolor=ec, facecolor=color, alpha=0.92, zorder=zorder)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, color="white", fontweight="bold", zorder=zorder + 1)
    return box


def draw_arrow(ax, x1, y1, x2, y2, color=C_TEXT, lw=1.3, style="-|>", zorder=2):
    ar = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12,
                         linewidth=lw, color=color, zorder=zorder)
    ax.add_patch(ar)


def fig_arch_bcm():
    fig, ax = plt.subplots(figsize=(13.5, 7.6))
    ax.set_xlim(0, 16); ax.set_ylim(0, 9.4); ax.axis("off")

    # ---- 输入 ----
    draw_block(ax, 0.15, 3.9, 1.15, 1.3, C_IN, "Input\n640×640×3", fs=8.5)

    # ---- 骨干 (列 1.6 - 5.6) ----
    bx = 1.65
    bw, bh = 1.55, 0.72
    bs = 0.30
    draw_block(ax, bx, 4.05, bw, bh, C_CONV, "Conv k3 s2\n320×320", fs=8)
    draw_block(ax, bx, 3.30, bw, bh, C_CONV, "Conv k3 s2\n160×160", fs=8)
    draw_block(ax, bx, 2.55, bw, bh, C_C2F, "C2f\n160×160\n×3", fs=8)
    draw_block(ax, bx, 1.80, bw, bh, C_CONV, "Conv k3 s2\n80×80", fs=8)
    draw_block(ax, bx, 1.05, bw, bh, C_CA, "C2f-CA\n80×80 ×6", fs=8)
    draw_block(ax, bx, 0.30, bw, bh, C_CONV, "Conv k3 s2\n40×40", fs=8)
    draw_block(ax, bx + bw + bs, 4.05, bw, bh, C_CA, "C2f-CA\n40×40 ×6", fs=8)
    draw_block(ax, bx + bw + bs, 3.30, bw, bh, C_CONV, "Conv k3 s2\n20×20", fs=8)
    draw_block(ax, bx + bw + bs, 2.55, bw, bh, C_CA, "C2f-CA\n20×20 ×3", fs=8)
    draw_block(ax, bx + bw + bs, 1.80, bw, bh, C_SPPF, "SPPF\n20×20", fs=8)

    # 骨干箭头
    xs = [bx + bw] * 6
    ys = [4.0, 3.25, 2.5, 1.75, 1.0, 0.25]
    for i in range(5):
        draw_arrow(ax, xs[i] + 0.02, ys[i], bx + bw + bs - 0.02, ys[i])
    # 第二列向下
    draw_arrow(ax, bx + bw + bs + bw + 0.02, 4.0, bx + bw + bs + bw + 0.02, 3.25)
    draw_arrow(ax, bx + bw + bs + bw + 0.02, 3.25, bx + bw + bs + bw + 0.02, 2.5)
    draw_arrow(ax, bx + bw + bs + bw + 0.02, 2.5, bx + bw + bs + bw + 0.02, 1.75)

    # 骨干标注
    ax.text(bx + bw / 2, 8.55, "Backbone", ha="center", fontsize=13, fontweight="bold", color=C_TEXT)
    ax.text(bx + bw / 2, 8.1, "C2f-CA: Coordinate Attention", ha="center", fontsize=9.5, color=C_CA)

    # ---- 颈部 (列 6.4 - 10.6) ----
    nx = 6.55
    nw, nh = 1.5, 0.72
    # 上行
    draw_block(ax, nx, 2.55, nw, nh, C_SPPF, "Up×2", fs=9)
    draw_block(ax, nx, 1.80, nw, nh, C_BIFPN, "BiFPN-Concat\n40×40", fs=8)
    draw_block(ax, nx, 1.05, nw, nh, C_CA, "C2f-CA\n40×40", fs=8)
    draw_block(ax, nx, 0.30, nw, nh, C_SPPF, "Up×2", fs=9)
    draw_block(ax, nx + nw + 0.30, 2.55, nw, nh, C_BIFPN, "BiFPN-Concat\n80×80", fs=8)
    draw_block(ax, nx + nw + 0.30, 1.80, nw, nh, C_CA, "C2f-CA\n80×80", fs=8)
    draw_block(ax, nx + nw + 0.30, 1.05, nw, nh, C_SPPF, "Up×2", fs=9)
    draw_block(ax, nx + nw + 0.30, 0.30, nw, nh, C_BIFPN, "BiFPN-Concat\n160×160", fs=8)
    draw_block(ax, nx + 2 * (nw + 0.30), 2.55, nw, nh, C_CA, "C2f-CA\n160×160", fs=8)
    # 下行
    draw_block(ax, nx + 2 * (nw + 0.30), 1.80, nw, nh, C_CONV, "Conv s2", fs=8.5)
    draw_block(ax, nx + 2 * (nw + 0.30), 1.05, nw, nh, C_BIFPN, "BiFPN-Concat\n80×80", fs=8)
    draw_block(ax, nx + 2 * (nw + 0.30), 0.30, nw, nh, C_CA, "C2f-CA\n80×80", fs=8)
    draw_block(ax, nx + 3 * (nw + 0.30), 2.55, nw, nh, C_CONV, "Conv s2", fs=8.5)
    draw_block(ax, nx + 3 * (nw + 0.30), 1.80, nw, nh, C_BIFPN, "BiFPN-Concat\n40×40", fs=8)
    draw_block(ax, nx + 3 * (nw + 0.30), 1.05, nw, nh, C_CA, "C2f-CA\n40×40", fs=8)
    draw_block(ax, nx + 4 * (nw + 0.30), 2.55, nw, nh, C_CONV, "Conv s2", fs=8.5)
    draw_block(ax, nx + 4 * (nw + 0.30), 1.80, nw, nh, C_BIFPN, "BiFPN-Concat\n20×20", fs=8)
    draw_block(ax, nx + 4 * (nw + 0.30), 1.05, nw, nh, C_CA, "C2f-CA\n20×20", fs=8)
    ax.text(nx + 2 * nw + 0.3, 8.55, "Neck", ha="center", fontsize=13, fontweight="bold", color=C_TEXT)
    ax.text(nx + 2 * nw + 0.3, 8.1, "BiFPN: learnable weighted fusion", ha="center", fontsize=9.5, color=C_BIFPN)

    # ---- 检测头 (列 13.6 - 15.3) ----
    dx = 13.85
    dw, dh = 1.5, 0.62
    heads = [
        ("P5 20×20", C_DET, 4.35),
        ("P4 40×40", C_DET, 3.55),
        ("P3 80×80", C_DET, 2.75),
        ("P2 160×160", C_P2, 1.95),   # 改进: 新增
    ]
    for label, col, yy in heads:
        draw_block(ax, dx, yy, dw, dh, col, label, fs=9)
    ax.text(dx + dw / 2, 8.55, "Head", ha="center", fontsize=13, fontweight="bold", color=C_TEXT)
    ax.text(dx + dw / 2, 8.1, "P2: high-res small-object layer (new)", ha="center", fontsize=9.5, color=C_P2)

    # ---- 输出 ----
    draw_block(ax, dx, 0.5, dw, 0.8, C_OUT, "Output\nImmature / Mature", fs=8.5)

    # 颈部到检测头连线
    draw_arrow(ax, 10.55, 2.9, dx, 4.55, lw=1.0)
    draw_arrow(ax, 10.55, 2.1, dx, 3.75, lw=1.0)
    draw_arrow(ax, 10.55, 1.3, dx, 2.95, lw=1.0)
    draw_arrow(ax, 10.55, 0.5, dx, 2.15, lw=1.0, color=C_P2)
    # 检测头到输出
    for yy in (4.5, 3.7, 2.9, 2.1):
        draw_arrow(ax, dx + dw / 2, yy, dx + dw / 2, 1.35, lw=0.8, color="#ADB5BD")

    # 图例
    leg = [
        mpatches.Patch(color=C_C2F, label="C2f"),
        mpatches.Patch(color=C_CA, label="C2f-CA (Coordinate Attention)"),
        mpatches.Patch(color=C_BIFPN, label="BiFPN-Concat (learnable weights)"),
        mpatches.Patch(color=C_P2, label="P2 high-resolution head"),
        mpatches.Patch(color=C_DET, label="Detect"),
    ]
    ax.legend(handles=leg, loc="lower center", ncol=5, frameon=False,
              fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.savefig(os.path.join(OUTD, "fig_arch_bcm.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved fig_arch_bcm.png")


def fig_c2f_ca():
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5.2); ax.axis("off")
    x, w = 0.3, 1.1
    y1, y2, y3 = 3.6, 2.4, 1.2
    h = 1.0
    # 主路径
    draw_block(ax, x, y1, w, h, C_C2F, "Conv\n1×1", fs=8.5)
    draw_block(ax, x + 1.5, y1, w, h, C_C2F, "Conv\n3×3", fs=8.5)
    draw_block(ax, x + 3.0, y1, w, h, C_C2F, "Conv\n3×3", fs=8.5)
    draw_block(ax, x + 4.5, y1, w, h, C_C2F, "Conv\n1×1", fs=8.5)
    draw_block(ax, x + 6.6, y1, w, h, C_CA, "Coord\nAtt", fs=9)
    draw_block(ax, x + 8.3, y1, w, h, C_OUT, "Out", fs=9)
    for i in range(4):
        draw_arrow(ax, x + i * 1.5 + w, y1 + h / 2, x + (i + 1) * 1.5, y1 + h / 2)
    draw_arrow(ax, x + 6.0 + w, y1 + h / 2, x + 6.6, y1 + h / 2)
    draw_arrow(ax, x + 7.6 + w, y1 + h / 2, x + 8.3, y1 + h / 2)
    # 分支 (identity)
    draw_arrow(ax, x + w / 2, y1, x + w / 2, y3, lw=0.9, color="#ADB5BD")
    draw_arrow(ax, x + w / 2, y3, x + 8.3 + w / 2, y3, lw=0.9, color="#ADB5BD")
    draw_arrow(ax, x + 8.3 + w / 2, y3, x + 8.3 + w / 2, y1 + 0.02, lw=0.9, color="#ADB5BD")
    ax.text(x + w / 2 + 0.3, y3 - 0.2, "Bottleneck × n  (shortcut)", fontsize=8.5, color="#868E96")
    # CoordAtt 细节
    ax.text(x + 6.6 + w / 2, y1 + h + 0.28, "X-AvgPool → 1×1 Conv → split → Sigmoid", fontsize=8, ha="center", color=C_CA)
    ax.text(x + 6.6 + w / 2, y1 + h - 0.05, "Y-AvgPool → 1×1 Conv → Sigmoid", fontsize=8, ha="center", color=C_CA)
    ax.text(4.4, 4.95, "C2f-CA: CSP bottleneck with Coordinate Attention", fontsize=12,
            fontweight="bold", ha="center", color=C_TEXT)
    fig.savefig(os.path.join(OUTD, "fig_c2f_ca.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved fig_c2f_ca.png")


def fig_bifpn():
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.4); ax.axis("off")
    # 三个输入特征 P5 P4 P3
    feats = [("P5 (20×20)", 3.3), ("P4 (40×40)", 2.2), ("P3 (80×80)", 1.1)]
    w, h = 1.9, 0.85
    for label, yy in feats:
        draw_block(ax, 0.4, yy, w, h, C_CONV, label, fs=9)
    # 加权融合
    draw_block(ax, 4.3, 1.5, 2.4, 1.9, C_BIFPN, "Σ wᵢ·fᵢ\n(1×1 proj + resize)", fs=8.5)
    for _, yy in feats:
        draw_arrow(ax, 2.4, yy + h / 2, 4.3, 2.45)
    draw_block(ax, 7.9, 1.9, 1.6, 1.1, C_CA, "C2f-CA", fs=9)
    draw_arrow(ax, 6.7, 2.45, 7.9, 2.45)
    ax.text(5.5, 3.85, "BiFPN-Concat:  out = Σ wᵢ/(ε+Σwⱼ) · Resize(fᵢ)", fontsize=9.5,
            ha="center", color=C_BIFPN, fontweight="bold")
    ax.text(5.5, 3.35, "wᵢ ≥ 0 (ReLU),  ε = 1e-4", fontsize=8.5, ha="center", color="#868E96")
    fig.savefig(os.path.join(OUTD, "fig_bifpn.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved fig_bifpn.png")


if __name__ == "__main__":
    fig_arch_bcm()
    fig_c2f_ca()
    fig_bifpn()
    print("all arch figures ->", OUTD)
