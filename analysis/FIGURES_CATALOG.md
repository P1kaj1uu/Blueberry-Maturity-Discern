# 论文素材清单

> 生成方式：数据来自真实蓝莓数据集 + 真实训练/验证结果；每个脚本可一键重跑。

## 一、数据集统计图（figs/dataset/，脚本 `analysis/make_dataset_figs.py`）

| 文件                          | 内容                                                        | 论文用途               |
| ----------------------------- | ----------------------------------------------------------- | ---------------------- |
| `fig_dataset_overview.png`    | 4 合一：类别实例数 / 每图实例数 / 中心热图 / 框尺寸散点     | Figure 1（数据集总览） |
| `fig_class_dist.png`          | 每类实例数（Immature 5244 / Mature 2236）                   | 类别不平衡说明         |
| `fig_boxes_per_image.png`     | 每图实例数箱线图（train/valid/test）                        | 密度说明               |
| `fig_bbox_size.png`           | 框宽高散点 + 面积直方图（中位数约 5217 px²@640 → 中等目标） | 小目标挑战证据         |
| `fig_bbox_center_heatmap.png` | 目标中心位置热图                                            | 空间分布               |
| `fig_bbox_ar.png`             | 长宽比分布                                                  | 目标形态               |
| `fig_examples.png`            | 真实图像 + GT 框（按类别着色）                              | Figure 1 示例图        |

## 二、网络结构图（figs/arch/，脚本 `analysis/make_arch_figs.py`）

| 文件               | 内容                                                  |
| ------------------ | ----------------------------------------------------- |
| `fig_arch_bcm.png` | YOLOv8-BCM 整体结构（骨干/颈部/检测头，改进模块高亮） |
| `fig_c2f_ca.png`   | C2f-CA 模块内部结构（Bottleneck + CoordAtt）          |
| `fig_bifpn.png`    | BiFPN 加权融合示意                                    |

## 三、训练/验证结果图（figs/results/）

训练中自动生成（`runs/<exp>/` 内）：results.png（损失+指标）、PR_curve.png、confusion_matrix.png、val_batch\*.jpg。
脚本重绘（更统一风格）：

| 文件                    | 内容                                        | 脚本                 |
| ----------------------- | ------------------------------------------- | -------------------- |
| `fig_loss_curves.png`   | 各实验 train box/cls/dfl + val box 损失曲线 | `make_train_figs.py` |
| `fig_metric_curves.png` | mAP50 / mAP50-95 随 epoch 变化              | `make_train_figs.py` |
| `fig_ablation_bars.png` | 消融矩阵柱状图（mAP50/mAP50-95/P/R）        | `make_train_figs.py` |
| `fig_pr_curves.png`     | 6 模型 PR 曲线对比（IoU=0.5）               | `make_val_figs.py`   |
| `fig_f1_curves.png`     | F1-Confidence 对比                          | `make_val_figs.py`   |
| `fig_confusion_cmp.png` | baseline vs BCM 归一化混淆矩阵              | `make_val_figs.py`   |
| `fig_detect_visual.png` | 测试图检测可视化（GT 实线 vs 预测虚线）     | `make_val_figs.py`   |
| `fig_cam.png`           | C2f-CA 特征热力图（注意力聚焦果实）         | `make_cam_figs.py`   |

## 四、实验配置（runs/，最终报告组）

| 目录           | 配置                                 | 说明                           |
| -------------- | ------------------------------------ | ------------------------------ |
| `baseline`     | YOLOv8n + CIoU（30 ep, scratch）     | 基线                           |
| `base_wiou`    | + WIoUv3（30 ep）                    | 损失改进消融                   |
| `bifpn`        | + BiFPN_Concat（30 ep）              | 融合消融                       |
| `p2`           | + P2 层（30 ep）                     | 小目标层消融                   |
| `ca`           | + C2f-CA（30 ep）                    | 注意力消融（负面结果，分析用） |
| `bcmlite`      | **BCM-Lite（P2+BiFPN+WIoU, 40 ep）** | **提出模型（最优）**           |
| `bcm`          | BCM full（30 ep）                    | 组合负面结果                   |
| `baseline_pt`  | YOLOv8n 预训练微调（50 ep）          | 微调组基线                     |
| `p2_pt`        | + P2 预训练微调（60 ep）             | 微调组                         |
| `bcm_pt_v2/v3` | BCM 预训练微调（60 ep）              | 微调组（ReZero/CIoU 变体）     |

## 五、论文写作注意

1. 所有图均由本仓库脚本从真实数据生成，**不要**用 AI 绘图工具伪造训练曲线/检测效果；
2. PR/混淆矩阵/损失曲线必须与 `results.csv` 数字一致；报告指标时给出 ± 或至少固定 seed（42）可复现；
3. 数据集来自 Ni et al. (2020)，正文必须引用并遵守 CC BY-NC-SA 4.0（注明来源、非商业用途）；
4. 本实验为 CPU 小规模（416px、30 epoch），投稿前建议在 GPU 上按 640px、100+ epoch 复跑以提升指标竞争力，脚本已支持；
5. 若目标是农业类一区（CAE/Biosystems Engineering 等），建议补充：与经典方法（Faster R-CNN、SSD、YOLOv5/v7/v9/v10）的对比实验 + 小/中/大目标分尺寸 mAP + 光照/遮挡鲁棒性分析。
