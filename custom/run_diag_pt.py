# -*- coding: utf-8 -*-
"""诊断实验: p2_pt (预训练+P2+CIoU) 与 bcm_pt_v3 (预训练+CA+BiFPN+P2+CIoU)。"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = [
    ("p2_pt", "yolov8n-p2.yaml", False),
    ("bcm_pt_v3", "yolov8n-bcm.yaml", False),
]
EPOCHS = 60
log = os.path.join(ROOT, "runs", "diag_pt.log")
results = []
t0 = time.time()
with open(log, "w", encoding="utf-8") as lf:
    for name, cfg, wiou in RUNS:
        t1 = time.time()
        msg = f"\n===== {name} start ====="
        print(msg, flush=True)
        lf.write(msg + "\n"); lf.flush()
        cmd = [sys.executable, os.path.join(ROOT, "custom", "train.py"),
               "--cfg", os.path.join(ROOT, "custom", cfg),
               "--name", name, "--epochs", str(EPOCHS),
               "--imgsz", "416", "--batch", "8", "--pretrained"]
        if wiou:
            cmd.append("--wiou")
        rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
        dt = (time.time() - t1) / 60
        results.append((name, rc, dt))
        msg = f"----- {name} done rc={rc} {dt:.1f} min -----"
        print(msg, flush=True)
        lf.write(msg + "\n"); lf.flush()
msg = f"\nALL DONE in {(time.time()-t0)/60:.1f} min: {results}"
print(msg, flush=True)
with open(log, "a", encoding="utf-8") as lf:
    lf.write(msg + "\n")
