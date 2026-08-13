# 实验结果汇总

- 数据集: Blueberry maturity (Ni et al. 2020), train 366 / valid 104 / test 54
- 输入 416×416, 30-60 epochs, AdamW, seed=42, CPU (i7-14700KF)

| Method                 | Pretrain     | P     | R     | mAP50 | mAP50-95 |
| ---------------------- | ------------ | ----- | ----- | ----- | -------- |
| YOLOv8n (baseline)     | w/o pretrain | 0.756 | 0.575 | 0.662 | 0.386    |
| + WIoUv3               | w/o pretrain | 0.753 | 0.579 | 0.671 | 0.393    |
| + BiFPN-Concat         | w/o pretrain | 0.807 | 0.578 | 0.686 | 0.401    |
| + P2                   | w/o pretrain | 0.816 | 0.578 | 0.708 | 0.432    |
| + C2f-CA (neg. result) | w/o pretrain | 0.745 | 0.531 | 0.625 | 0.329    |
| BCM-Lite (Ours)        | w/o pretrain | 0.797 | 0.650 | 0.741 | 0.477    |
| BCM full (neg. result) | w/o pretrain | 0.718 | 0.557 | 0.628 | 0.382    |
| YOLOv8n                | pretrain     | 0.919 | 0.839 | 0.908 | 0.697    |
| + P2                   | pretrain     | 0.921 | 0.841 | 0.910 | 0.684    |
| BCM full (CIoU)        | pretrain     | 0.911 | 0.816 | 0.896 | 0.656    |
