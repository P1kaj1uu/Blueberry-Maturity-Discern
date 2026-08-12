# -*- coding: utf-8 -*-
"""训练/验证入口: 注册自定义模块 + 可选 WIoU 损失。

用法:
  python custom/train.py --cfg custom/yolov8n-bcm.yaml --name bcm \
         --data dataset/data.yaml --epochs 30 --imgsz 416 --batch 8 --wiou
"""
import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import custom.bbcm_modules as M  # noqa: E402   (注册 C2f_CA / BiFPN_Concat)
M.register()  # 注入 ultralytics yaml 解析命名空间
import custom.bbcm_loss as LOSS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="custom/yolov8n-bcm.yaml")
    ap.add_argument("--data", default="dataset/data.yaml")
    ap.add_argument("--name", default="exp")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--wiou", action="store_true", help="use WIoUv3 loss")
    ap.add_argument("--pretrained", action="store_true", help="start from pretrained weights")
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights", "yolov8n.pt"),
                    help="pretrained weight path")
    ap.add_argument("--device", default="")
    ap.add_argument("--project", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    torch.set_num_threads(max(4, torch.get_num_threads()))
    print(f"threads={torch.get_num_threads()} cuda={torch.cuda.is_available()}")

    if args.wiou:
        LOSS.patch_wiou(True)
        print("[loss] WIoUv3 enabled")
    else:
        LOSS.patch_wiou(False)
        print("[loss] CIoU (default)")

    from ultralytics import YOLO
    cfg = os.path.join(ROOT, args.cfg) if not os.path.isabs(args.cfg) else args.cfg
    data = os.path.join(ROOT, args.data) if not os.path.isabs(args.data) else args.data

    model = YOLO(cfg)
    if args.pretrained:
        model = YOLO(cfg).load(args.weights)   # 按层匹配加载预训练权重(跳过不匹配层)
        print("[init] loaded pretrained weights from", args.weights)

    model.train(
        data=data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or (0 if torch.cuda.is_available() else "cpu"),
        project=args.project,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        seed=42,
        deterministic=True,
        plots=True,
        val=True,
        cache=False,
        amp=False,          # CPU 上关闭 AMP 更稳
        close_mosaic=5,
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3.0,
        hsv_h=0.015,        # 颜色域增强(成熟度依赖颜色, 适度增强)
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
    )
    print("[done] training finished:", args.name)


if __name__ == "__main__":
    main()
