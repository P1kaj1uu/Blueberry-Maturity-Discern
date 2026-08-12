# -*- coding: utf-8 -*-
"""决定性实验: BCM-Lite (P2+BiFPN+WIoU, 无 CA) 从头训练 40 epochs。
对比 baseline (scratch): mAP50=0.662, mAP50-95=0.386
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log = os.path.join(ROOT, "runs", "bcmlite.log")
cmd = [sys.executable, os.path.join(ROOT, "custom", "train.py"),
       "--cfg", os.path.join(ROOT, "custom", "yolov8n-bcmlite.yaml"),
       "--name", "bcmlite", "--epochs", "40",
       "--imgsz", "416", "--batch", "8", "--wiou"]
t0 = time.time()
with open(log, "w", encoding="utf-8") as lf:
    rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
print(f"bcmlite done rc={rc} {(time.time()-t0)/60:.1f} min")
