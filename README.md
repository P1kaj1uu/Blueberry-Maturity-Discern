# Blueberry Maturity Detection — YOLOv8-BCM

基于 YOLOv8 的蓝莓成熟度检测。

## 数据集

- 来源：Ni et al. (2020, _Horticulture Research_, doi:10.1038/s41438-020-0323-3)
  蓝莓果分割数据集（HuggingFace `c-tan/Blueberry-fruit-segmentation`，CC BY-NC-SA 4.0）
- 格式：COCO 实例分割 → YOLO bbox（`analysis/coco_to_yolo.py`）
- 类别：`Immature`（未成熟） / `Mature`（成熟），共 7480 实例
- 划分：train 366 / valid 104 / test 54（seed=42，7:2:1）

## 模型改进（YOLOv8-BCM / BCM-Lite）

| 模块         | 改进                                                         | 文件                          |
| ------------ | ------------------------------------------------------------ | ----------------------------- |
| 检测头       | **P2 高分辨率层（stride 4）** ← 贡献最大                     | `custom/yolov8n-bcmlite.yaml` |
| 颈部         | **BiFPN_Concat（可学习加权融合）**                           | `custom/bbcm_modules.py`      |
| 损失         | **WIoUv3 动态聚焦回归损失**                                  | `custom/bbcm_loss.py`         |
| 骨干(分析用) | C2f-CA 坐标注意力（本数据集为负贡献，ReZero 门控变体已实现） | `custom/bbcm_modules.py`      |

**最终模型 BCM-Lite = P2 + BiFPN + WIoU**（从头训练 40 ep @416）：mAP50 **0.741**（+0.079 vs 基线 0.662）、mAP50-95 **0.477**（+0.091）。完整真实结果见 `analysis/RESULTS.md`。

消融变体 yaml：`custom/yolov8n.yaml`（基线）、`yolov8n-bifpn.yaml`、`yolov8n-p2.yaml`、`yolov8n-bcmlite.yaml`（提出）、`yolov8n-bcm.yaml`（含 CA 全量）、`yolov8n-ca.yaml`。

## 使用

```bash
# 训练全部 6 个配置（消融矩阵，CPU 串行，约 4h @ 416px / 30 epochs）
python custom/batch_train.py

# 单配置训练
python custom/train.py --cfg custom/yolov8n-bcm.yaml --name bcm --epochs 30 --imgsz 416 --batch 8 --wiou

# 验证出图（PR 曲线 / F1 / 混淆矩阵 / 检测可视化）
python analysis/make_val_figs.py

# 数据集统计图
python analysis/make_dataset_figs.py

# 网络结构示意图
python analysis/make_arch_figs.py

# 特征热力图（C2f-CA 激活可视化）
python analysis/make_cam_figs.py
```

## 目录

- `dataset/` — YOLO 格式数据集（`data.yaml`）
- `custom/` — 改进模型代码（模块、损失、yaml、训练入口）
- `analysis/` — 数据处理与出图脚本、`IMPROVEMENT_PLAN.md`（论文方案）
- `figs/` — 论文图（`dataset/` 统计图、`arch/` 结构图、`results/` 训练/验证结果图）
- `runs/` — 训练输出（各实验 best.pt / results.csv / 自动 plots）
