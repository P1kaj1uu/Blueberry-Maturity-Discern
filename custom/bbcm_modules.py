# -*- coding: utf-8 -*-
"""YOLOv8-BCM 自定义模块。

包含:
  1. CoordAtt      —— Coordinate Attention (Hou et al., CVPR 2021)
  2. C2f_CA        —— 融入坐标注意力的 C2f (骨干)
  3. BiFPN_Concat  —— 带可学习权重的双向特征融合 (Tan et al., CVPR 2020)
用法: import bbcm_modules 后, 自定义 yaml 即可引用这些名字。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import C2f, Conv, Concat
from ultralytics.nn.modules.conv import autopad
from ultralytics.nn.modules.block import Bottleneck


class CoordAtt(nn.Module):
    """Coordinate Attention: 沿 H/W 两个方向编码位置信息的通道注意力。"""
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mip, oup, 1, bias=False)
        self.conv_w = nn.Conv2d(mip, oup, 1, bias=False)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)                       # (n,c,h,1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)   # (n,c,1,w)
        y = torch.cat([x_h, x_w], dim=2)           # (n,c,h+w,1)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)              # (n,c,1,w)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_w * a_h


class C2f_CA(C2f):
    """C2f + 输出端坐标注意力 (ReZero 门控)。

    设计动机: 成熟度判别的核心是"颜色-空间"联合信息; 坐标注意力
    同时保留通道关系与果实空间位置, 对密集簇生的小果更友好。
    ReZero 门控: gate 初始化为 0, 输出 = y + gate*(att(y)-y),
    初始完全等价于普通 C2f —— 不破坏预训练权重/特征尺度, 训练中
    逐步开启注意力 (解决新模块在微调场景下收敛慢的问题)。
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.att = CoordAtt(c2, c2)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        y = super().forward(x)
        return y + self.gate * (self.att(y) - y)


class BiFPN_Concat(nn.Module):
    """带可学习归一化权重的特征融合 (BiFPN 风格)。

    对每个输入特征: 先经 1x1 投影统一到 c2 通道 (通道不同时),
    再按可学习权重 w_i 加权: out = Σ w_i/(ε+Σw) · Resize(feat_i)。
    权重经 ReLU 保证非负, 用 ε=1e-4 归一化。
    """
    def __init__(self, c1, c2):
        super().__init__()
        self.c2 = c2
        self.project = None     # 延迟构建: ModuleList of 1x1 Conv / Identity
        self.weights = None
        self.nin = 0
        self.eps = 1e-4

    def forward(self, x):
        xs = list(x) if isinstance(x, (list, tuple)) else [x]
        # 延迟构建投影层与权重 (model 构建阶段的首次 forward 即完成)
        if self.project is None or len(self.project) != len(xs):
            self.project = nn.ModuleList()
            for xi in xs:
                if xi.shape[1] == self.c2:
                    self.project.append(nn.Identity())
                else:
                    self.project.append(Conv(xi.shape[1], self.c2, 1))
            self.weights = nn.Parameter(torch.ones(len(xs), device=xs[0].device))
            self.nin = len(xs)
        w = F.relu(self.weights)
        norm = torch.sum(w) + self.eps
        target = max(x.shape[-2:] for x in xs)
        out = None
        for wi, proj, xi in zip(w, self.project, xs):
            xi = proj(xi)
            if xi.shape[-2:] != target:
                xi = F.interpolate(xi, size=target, mode="nearest")
            xi = xi * (wi / norm)
            out = xi if out is None else out + xi
        return out


class C2f_WCA(C2f):
    """C2f + 内部 Bottleneck 替换为带坐标注意力的变体(在瓶颈分支后增强)。"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.att = CoordAtt(c2, c2)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        y = super().forward(x)
        return y + self.gate * (self.att(y) - y)


def register():
    """把自定义模块注册进 ultralytics 的 yaml 解析命名空间。

    ultralytics 8.4.x 的 parse_model 内部用局部 frozenset base_modules/
    repeat_modules 判定"哪些模块按 C2f 家族方式解析参数", 自定义类必须
    进入这两个集合。由于集合是函数局部变量, 采用源码级 patch:
    读取 parse_model 源码 -> 注入自定义类 -> exec 回编译 -> 替换原函数。
    不改动 site-packages 文件, 可复现。
    """
    import inspect
    import ultralytics.nn.tasks as T

    for cls_name in ("CoordAtt", "C2f_CA", "BiFPN_Concat", "C2f_WCA"):
        setattr(T, cls_name, globals()[cls_name])
        import ultralytics.nn.modules as M
        setattr(M, cls_name, globals()[cls_name])

    src = inspect.getsource(T.parse_model)
    marker = "    base_modules = frozenset(\n        {\n"
    assert marker in src, "parse_model source marker not found"
    inject = ("            C2f_CA, C2f_WCA,\n"
              "            CoordAtt,\n")
    src2 = src.replace(marker, marker + inject, 1)
    marker2 = "    repeat_modules = frozenset(  # modules with 'repeat' arguments\n        {\n"
    if marker2 in src2:
        src2 = src2.replace(marker2, marker2 + "            C2f_CA, C2f_WCA,\n", 1)
    # BiFPN_Concat: 类似 Concat 的独立分支 (f 为 list, c2 取 args[0] 并按 width 缩放)
    concat_marker = "        elif m is Concat:\n            c2 = sum(ch[x] for x in f)\n"
    assert concat_marker in src2, "concat marker not found"
    bifpn_branch = (
        concat_marker +
        "        elif m is BiFPN_Concat:\n"
        "            c2 = make_divisible(min(args[0], max_channels) * width, 8)\n"
        "            args = [ch[f[-1]] if isinstance(f, list) else ch[f], c2]\n"
    )
    src2 = src2.replace(concat_marker, bifpn_branch, 1)
    ns = dict(T.__dict__)
    for name in ("CoordAtt", "C2f_CA", "BiFPN_Concat", "C2f_WCA"):
        ns[name] = globals()[name]
    exec(compile(src2, "<patched_parse_model>", "exec"), ns)
    T.parse_model = ns["parse_model"]
    return True
