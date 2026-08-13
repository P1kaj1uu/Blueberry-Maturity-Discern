# -*- coding: utf-8 -*-
"""验证自定义 yaml 能被 ultralytics 解析并构建(不训练)。"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import custom.bbcm_modules as M  # noqa
M.register()
from ultralytics import YOLO

for name in ["yolov8n-bcm", "yolov8n-ca", "yolov8n-p2", "yolov8n-bifpn"]:
    try:
        m = YOLO(os.path.join(ROOT, "custom", name + ".yaml"))
        print(f"[OK]   {name}: layers={len(m.model.model)}  "
              f"params={sum(p.numel() for p in m.model.parameters())/1e6:.2f}M")
        # 前向冒烟测试
        import torch
        x = torch.zeros(1, 3, 416, 416)
        with torch.no_grad():
            y = m.model(x)
        if isinstance(y, (list, tuple)):
            print(f"       forward ok, outputs={len(y)}")
    except Exception as e:
        import traceback
        print(f"[FAIL] {name}: {e}")
        traceback.print_exc()
