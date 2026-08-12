# -*- coding: utf-8 -*-
"""预训练微调组: baseline 与 BCM 均用 yolov8n.pt 初始化, 50 epochs。"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = [
    ("baseline_pt", "yolov8n.yaml", False),
    ("bcm_pt", "yolov8n-bcm.yaml", True),
]
EPOCHS = 50
IMGSZ = 416
BATCH = 8

log = os.path.join(ROOT, "runs", "batch_train_pt.log")
results = []
t0 = time.time()
with open(log, "w", encoding="utf-8") as lf:
    for name, cfg, wiou in RUNS:
        t1 = time.time()
        msg = f"\n===== {name} ({cfg}, wiou={wiou}) start ====="
        print(msg, flush=True)
        lf.write(msg + "\n")
        lf.flush()
        cmd = [sys.executable, os.path.join(ROOT, "custom", "train.py"),
               "--cfg", os.path.join(ROOT, "custom", cfg),
               "--name", name, "--epochs", str(EPOCHS),
               "--imgsz", str(IMGSZ), "--batch", str(BATCH), "--pretrained"]
        if wiou:
            cmd.append("--wiou")
        rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
        dt = (time.time() - t1) / 60
        results.append((name, rc, dt))
        msg = f"----- {name} done rc={rc} {dt:.1f} min -----"
        print(msg, flush=True)
        lf.write(msg + "\n")
        lf.flush()
msg = f"\nALL DONE in {(time.time()-t0)/60:.1f} min: {results}"
print(msg, flush=True)
with open(log, "a", encoding="utf-8") as lf:
    lf.write(msg + "\n")
