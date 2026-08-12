# -*- coding: utf-8 -*-
"""BCM v2: 预训练 + ReZero 门控 C2f-CA + BiFPN + P2 + WIoU, 60 epochs。"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log = os.path.join(ROOT, "runs", "bcm_v2.log")
cmd = [sys.executable, os.path.join(ROOT, "custom", "train.py"),
       "--cfg", os.path.join(ROOT, "custom", "yolov8n-bcm.yaml"),
       "--name", "bcm_pt_v2", "--epochs", "60",
       "--imgsz", "416", "--batch", "8", "--pretrained", "--wiou"]
t0 = time.time()
with open(log, "w", encoding="utf-8") as lf:
    rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
print(f"bcm_pt_v2 done rc={rc} {(time.time()-t0)/60:.1f} min")
