# -*- coding: utf-8 -*-
"""批量训练: 消融矩阵 6 个配置, 串行执行。

  baseline  : YOLOv8n + CIoU
  base_wiou : YOLOv8n + WIoUv3
  ca        : + C2f_CA
  bifpn     : + BiFPN_Concat
  p2        : + P2 检测层
  bcm       : 完整 (C2f_CA + BiFPN + P2 + WIoUv3)
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = [
    ("baseline", "yolov8n.yaml", False),
    ("base_wiou", "yolov8n.yaml", True),
    ("ca", "yolov8n-ca.yaml", False),
    ("bifpn", "yolov8n-bifpn.yaml", False),
    ("p2", "yolov8n-p2.yaml", False),
    ("bcm", "yolov8n-bcm.yaml", True),
]
EPOCHS = 30
IMGSZ = 416
BATCH = 8

log = os.path.join(ROOT, "runs", "batch_train.log")
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
               "--imgsz", str(IMGSZ), "--batch", str(BATCH)]
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
