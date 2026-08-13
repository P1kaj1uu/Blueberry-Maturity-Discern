# -*- coding: utf-8 -*-
"""完整验证: yaml 构建 + 前向 + WIoU 损失冒烟测试。"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import custom.bbcm_modules as M
M.register()
import custom.bbcm_loss as LOSS
from ultralytics import YOLO
import torch

fails = 0
for name in ["yolov8n-bcm", "yolov8n-ca", "yolov8n-p2", "yolov8n-bifpn"]:
    try:
        m = YOLO(os.path.join(ROOT, "custom", name + ".yaml"))
        nparams = sum(p.numel() for p in m.model.parameters()) / 1e6
        x = torch.zeros(1, 3, 416, 416)
        with torch.no_grad():
            y = m.model(x)
        # 8.4.x Detect 前向输出为 dict (含 'one2one'/'one2many' 等键)
        if isinstance(y, dict):
            outs = next(iter(y.values()))
        elif isinstance(y, (list, tuple)):
            outs = y[0]
        else:
            outs = y
        shapes = [tuple(o.shape) for o in (outs if isinstance(outs, (list, tuple)) else [outs])]
        print(f"[OK] {name}: layers={len(m.model.model)} params={nparams:.2f}M out_shapes={shapes}")
    except Exception as e:
        fails += 1
        print(f"[FAIL] {name}: {e}")
        import traceback; traceback.print_exc()

# WIoU 冒烟测试
try:
    LOSS.patch_wiou(True)
    box1 = torch.rand(16, 4) * 0.8
    box2 = torch.rand(16, 4) * 0.8
    iou = LOSS.wiou_v3_bbox_iou(box1, box2, xywh=True)
    assert iou.shape[0] == 16, iou.shape
    assert bool((iou >= -1).all() and (iou <= 1).all())
    LOSS.patch_wiou(False)
    print("[OK] WIoUv3 smoke test passed")
except Exception as e:
    fails += 1
    print(f"[FAIL] WIoUv3: {e}")
    import traceback; traceback.print_exc()

print("RESULT:", "ALL OK" if fails == 0 else f"{fails} FAILURES")
sys.exit(1 if fails else 0)
