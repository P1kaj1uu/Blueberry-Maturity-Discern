# -*- coding: utf-8 -*-
"""特征热力图: 骨干 C2f_CA 输出的类激活可视化。

用法: python analysis/make_cam_figs.py
输出: figs/results/fig_cam.png  (3 张测试图: 原图 / 热力图 / 叠加)
"""
import os
import sys
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import custom.bbcm_modules as M
M.register()

OUTD = os.path.join(ROOT, "figs", "results")
os.makedirs(OUTD, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def find_best(name="bcm"):
    hits = glob.glob(os.path.join(ROOT, "runs", "**", name, "weights", "best.pt"), recursive=True)
    return hits[0] if hits else None


def cam_feature(model, img_tensor):
    """提取骨干最后 C2f_CA 层输出并返回均值激活图。"""
    x = img_tensor
    feat = None
    layers = model.model.model
    # 逐层前向 backbone (0..8), 记录最后一个 C2f_CA 输出
    for i in range(9):
        x = layers[i](x)
        if type(layers[i]).__name__ == "C2f_CA":
            feat = x
    if feat is None:
        raise RuntimeError("no C2f_CA layer found in backbone")
    f = feat
    act = f.mean(dim=1).squeeze()                      # (H, W)
    act = (act - act.min()) / (act.max() - act.min() + 1e-8)
    act = torch.nn.functional.interpolate(
        act.unsqueeze(0).unsqueeze(0), size=(416, 416), mode="bilinear", align_corners=False
    ).squeeze().numpy()
    act = gaussian_filter(act, sigma=2.0)
    act = (act - act.min()) / (act.max() - act.min() + 1e-8)
    return act


def main():
    from ultralytics import YOLO
    wp = find_best()
    if not wp:
        print("no bcm best.pt yet")
        return
    model = YOLO(wp)
    model.model.eval()

    img_dir = os.path.join(ROOT, "dataset", "test", "images")
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.JPG")) + glob.glob(os.path.join(img_dir, "*.jpg")))
    pick = [imgs[1], imgs[3], imgs[7]] if len(imgs) > 8 else imgs[:3]

    fig, axes = plt.subplots(len(pick), 3, figsize=(9.6, 3.4 * len(pick)))
    axes = np.atleast_2d(axes)
    for r, imp in enumerate(pick):
        im = Image.open(imp).convert("RGB")
        im416 = im.resize((416, 416))
        arr = np.asarray(im416).astype(np.float32) / 255.0
        t = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
        act = cam_feature(model, t)

        axes[r, 0].imshow(im416)
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(os.path.basename(imp)[:28], fontsize=8)
        if r == 0:
            axes[r, 0].set_title("Input", loc="left")

        axes[r, 1].imshow(act, cmap="jet", vmin=0, vmax=1)
        axes[r, 1].set_xticks([]); axes[r, 1].set_yticks([])
        if r == 0:
            axes[r, 1].set_title("C2f-CA activation", loc="left")

        axes[r, 2].imshow(im416)
        axes[r, 2].imshow(act, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        axes[r, 2].set_xticks([]); axes[r, 2].set_yticks([])
        if r == 0:
            axes[r, 2].set_title("Overlay", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTD, "fig_cam.png"))
    plt.close(fig)
    print("saved fig_cam.png")


if __name__ == "__main__":
    main()
