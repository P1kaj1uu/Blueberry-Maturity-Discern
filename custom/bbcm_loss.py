# -*- coding: utf-8 -*-
"""Wise-IoU v3 (WIoUv3) 损失, 用于替换 YOLOv8 的 CIoU 回归损失。

原理 (Tong et al., "Wise-IoU: Bounding Box Regression Loss with Dynamic
Focusing Mechanism", 2023):
  - WIoUv1: L = R_WIoU * exp((中心距)^2 / (外接框对角线)^2), R_WIoU ∈ [1, e)
  - WIoUv3: 按"离群度" beta = IoU* / IoU 动态分配非单调聚焦系数 r,
    对低质量/离群样本降权, 对普通样本提权 —— 适配密集遮挡场景。

集成方式: 仅替换 ultralytics.utils.loss 命名空间中的 bbox_iou,
只影响训练回归损失; 验证/指标计算仍用官方 CIoU, 不受影响。
"""
import math
import torch
import torch.nn.functional as F

# ---- WIoUv3 超参 (与论文一致) ----
_ALPHA = 1.9
_DELTA = 3.0
_MOMENTUM = 0.9
_EPS = 1e-7

# 离群度统计的运行均值 (跨 batch 更新)
_iou_mean = None


def _reset():
    global _iou_mean
    _iou_mean = None


def wiou_v3_bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """WIoUv3. 兼容 bbox_iou 的调用签名 (box1/box2 最后一维为 4)。"""
    global _iou_mean
    # 坐标解析 (与官方 bbox_iou 一致)
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2 = x1 - w1_, x1 + w1_
        b1_y1, b1_y2 = y1 - h1_, y1 + h1_
        b2_x1, b2_x2 = x2 - w2_, x2 + w2_
        b2_y1, b2_y2 = y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    # ---- WIoUv1: 距离度量 ----
    cx1, cy1 = (b1_x1 + b1_x2) / 2, (b1_y1 + b1_y2) / 2
    cx2, cy2 = (b2_x1 + b2_x2) / 2, (b2_y1 + b2_y2) / 2
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    dist2 = (cx2 - cx1).pow(2) + (cy2 - cy1).pow(2)
    u = (dist2 / (cw.pow(2) + ch.pow(2) + eps)).exp()   # R ∈ [1, e)

    # ---- WIoUv3: 动态非单调聚焦 ----
    with torch.no_grad():
        b = iou.detach()
        if _iou_mean is None:
            _iou_mean = b.mean()
        else:
            _iou_mean = _MOMENTUM * _iou_mean + (1 - _MOMENTUM) * b.mean()
        beta = b / (_iou_mean + eps)                    # 离群度
        r = beta / (_DELTA * _ALPHA ** (beta - _DELTA))  # 聚焦系数
        r = r.clamp_(min=0)

    loss = r * u * (1 - iou)
    # 返回与 bbox_iou 同语义: loss_iou = 1 - iou
    return 1 - loss


def patch_wiou(enable=True):
    """将 WIoUv3 挂载到训练损失路径 (仅 loss 模块命名空间)。"""
    import ultralytics.utils.loss as L
    if enable:
        L.bbox_iou = wiou_v3_bbox_iou
    else:
        from ultralytics.utils.metrics import bbox_iou
        L.bbox_iou = bbox_iou
    _reset()
