<div align="center">

# SARNet

**Structure-aware refinement for YOLO26 object detection**

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![Base: Ultralytics YOLO26](https://img.shields.io/badge/Base-Ultralytics%20YOLO26-0b7285.svg)](https://github.com/ultralytics/ultralytics)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776ab.svg)](https://www.python.org/)

[中文说明](#中文说明) · [English](#english)

</div>

---

## 中文说明

### 一、简介

SARNet 是一个面向目标检测的**结构感知（structure-aware）**检测网络。它以 Ultralytics **YOLO26** 为基线，
针对两个被广泛忽略的问题做了改进：

| 问题 | SARNet 的解决方案 | 代码位置 |
| --- | --- | --- |
| 常规 stem 使用大步长卷积，早期就把**边缘 / 轮廓 / 细纹理**等高频结构信息丢掉，小目标与细长目标特征被削弱 | **LoGStem**：高斯拉普拉斯（Laplacian-of-Gaussian）边缘增强 stem | [`ultralytics/nn/extra_modules/stem/LoG.py`](ultralytics/nn/extra_modules/stem/LoG.py) |
| 常规下采样直接使用 stride=2 卷积，**语义信息与结构信息被混在一起**，无法按需保留方向性结构 | **SDS**：语义引导双路径结构下采样（Semantic-guided Dual-path Structure downsampling） | [`ultralytics/nn/extra_modules/my_module/sds.py`](ultralytics/nn/extra_modules/my_module/sds.py) |

本仓库提供的模型配置：[`ultralytics/cfg/models/improve/my_module/yolo26-LoG-SDS.yaml`](ultralytics/cfg/models/improve/my_module/yolo26-LoG-SDS.yaml)

### 二、创新点

#### 1. LoGStem —— 高斯拉普拉斯边缘增强 Stem

常规 stem 只做一次 stride-2 卷积，纹理与边缘响应在进入主干前就已衰减。LoGStem 由四个部件组成，
在 **不引入额外监督** 的前提下把高频结构信息显式地保留下来：

1. **LoGFilter**：`7×7` 卷积做特征强化（通道 `in_chans → stem_dim/4`），随后接一个**固定的**
   Laplacian-of-Gaussian 卷积核（深度可分离、`bias=False`、权重不可学习）提取边缘；
   再以残差形式 `norm2(x + SiLU(norm1(LoG(x))))` 融合，避免边缘响应覆盖原始纹理。
2. **2× 下采样层**：`depthwise 3×3` + `Conv(..., 3, stride=2, groups=...)`，把特征图降到 `H/2`。
3. **Gaussian**：固定高斯平滑分支（`9×9`, `sigma=0.5`），与下采样结果残差相加，
   抑制 LoG 带来的高频噪声。
4. **DRFD_LoG**：双路径结构感知下采样——一条深度可分离卷积路径（`stride=2`）与一条最大池化路径
   （`3×3`, `stride=2`）分别提取“可学习结构”与“显著响应”，沿通道拼接后用 `1×1` 融合，
   并把特征图降到 `H/4`，直接接 P2 层。

> 输出：`[B, C, H/4, W/4]`。

#### 2. SDS —— 语义引导双路径结构下采样

SDS 替换主干中用于产生 **P3 / P4 / P5** 的 stride-2 下采样卷积，将“抽象语义”与“方向性结构”
解耦到两条独立路径，再用语义信息对结构路径做门控：

```
                       ┌──────────────── 语义路径 ────────────────┐
                       │ Conv(c1→c2/2, k=3, s=2) → Conv(k=1)      │──► semantic
       输入 x ─────────┤                                          │
                       │                                          │
                       └──────────────── 结构路径 ────────────────┐
                          AvgPool2d(2) → Conv(c1→c2/2, k=1)       │
                          → dw3×3 + dw(1×5) + dw(5×1) 相加         │──► structure
                          → Conv(k=1)                             │
                                                                  │
        SemanticGate(semantic) = SE 风格通道门控 ──► structure × gate
                                                                  │
        Concat[semantic, structure] ──► Conv(c2→c2, k=1) ──► 输出
```

* **语义路径**：`3×3` stride-2 卷积 + `1×1` 卷积，负责抽象语义与上下文。
* **结构路径**：`AvgPool2d(2)` 温和下采样以保留位置精度，`1×1` 投影后并联 `3×3`、
  **水平 `(1,5)`**、**垂直 `(5,1)`** 三个深度卷积，显式建模方向性结构（细长目标、道路、杆状物等）。
* **SemanticGate**：在语义路径上做通道注意力（`AdaptiveAvgPool2d → 1×1 → SiLU → 1×1 → Sigmoid`），
  用语义显著性对结构特征逐通道重加权，抑制与语义无关的背景纹理。
* 最后 `Concat` + `1×1` 融合，输出通道为 `c2`。

### 三、模型结构

主干（`yolo26-LoG-SDS.yaml`）：

| 层 | from | n | 模块 | 参数 | 输出 |
| --- | --- | --- | --- | --- | --- |
| 0 | -1 | 1 | **LoGStem** | `[128]` | P2/4 |
| 1 | -1 | 2 | C3k2 | `[256, False, 0.25]` | P2/4 |
| 2 | -1 | 1 | **SDS** | `[256]` | P3/8 |
| 3 | -1 | 2 | C3k2 | `[512, False, 0.25]` | P3/8 |
| 4 | -1 | 1 | **SDS** | `[512]` | P4/16 |
| 5 | -1 | 2 | C3k2 | `[512, True]` | P4/16 |
| 6 | -1 | 1 | **SDS** | `[1024]` | P5/32 |
| 7 | -1 | 2 | C3k2 | `[1024, True]` | P5/32 |
| 8 | -1 | 1 | SPPF | `[1024, 5, 3, True]` | P5/32 |
| 9 | -1 | 2 | C2PSA | `[1024]` | P5/32 |

检测头沿用 YOLO26 的 PAN 结构（`Upsample → Concat → C3k2 ×2` 与 `Conv(s=2) → Concat → C3k2 ×2`），
最终由 `Detect` 在 **P3/8、P4/16、P5/32** 三个尺度输出。

实测参数量（`imgsz=640`，检测头 `nc=80`）：

| 模型 | 参数量 | 相对基线 |
| --- | --- | --- |
| YOLO26n（基线） | 2,572,280 | — |
| **SARNet-n** | **2,548,112** | **−0.94%** |
| YOLO26s（基线） | 10,009,784 | — |
| **SARNet-s** | **9,771,752** | **−2.38%** |
| YOLO26m（基线） | 21,896,248 | — |
| **SARNet-m** | **20,539,032** | **−6.20%** |

> 即：在替换 stem 与三处下采样模块后，SARNet 的参数量**略低于**同规模基线。

### 四、环境配置

要求 Python ≥ 3.8，推荐 PyTorch ≥ 1.8（CUDA 版本按显卡驱动选择）。

```bash
# 1) 克隆仓库
git clone https://github.com/yujianfwq/SARNet.git
cd SARNet

# 2) 建议使用虚拟环境 / conda 环境
conda create -n sarnet python=3.10 -y
conda activate sarnet

# 3) 安装依赖
pip install -r requirements.txt

# 4) 以可编辑方式安装本仓库（让 `import ultralytics` 指向本仓库代码）
pip install -e .
```

验证环境与模型是否可用：

```bash
python -c "from ultralytics import YOLO; m = YOLO('ultralytics/cfg/models/improve/my_module/yolo26-LoG-SDS.yaml'); m.info()"
```

### 五、数据集准备

按 Ultralytics 的 YOLO 格式组织数据，并编写 `data.yaml`（`path` / `train` / `val` / `names`）：

```
dataset/
├── images/{train,val}
├── labels/{train,val}
└── data.yaml
```

`data.yaml` 示例：

```yaml
path: /path/to/dataset
train: images/train
val: images/val
names:
  0: class_a
  1: class_b
```

### 六、训练 / 验证 / 推理 / 导出

```bash
# 训练（默认 scale=n，可用 --scale 选择 n/s/m/l/x）
python train.py --data /path/to/dataset/data.yaml --scale n \
                --epochs 300 --batch 8 --imgsz 640 --device 0

# 从预训练权重初始化
python train.py --data data.yaml --scale m --pretrained yolo26m.pt

# 验证
python val.py --weights runs/sarnet/train/weights/best.pt --data /path/to/dataset/data.yaml

# 推理
python detect.py --weights runs/sarnet/train/weights/best.pt \
                 --source ultralytics/assets/bus.jpg

# 导出 ONNX
python export.py --weights runs/sarnet/train/weights/best.pt --format onnx
```

也可以完全按 Ultralytics 的原生接口使用：

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/improve/my_module/yolo26-LoG-SDS.yaml")
model.train(data="data.yaml", epochs=300, imgsz=640, batch=8, device=0)
```

### 七、实验结果

> 完整的对比实验、消融实验与可视化结果将在论文正式发表后同步更新。

| 方法 | 参数量 | mAP@50 | mAP@50-95 |
| --- | --- | --- | --- |
| YOLO26 | 2.57 M | — | — |
| YOLO26 + LoGStem | — | — | — |
| YOLO26 + SDS | — | — | — |
| **SARNet (YOLO26 + LoGStem + SDS)** | 2.55 M | — | — |

### 八、项目结构

```
SARNet/
├── train.py / val.py / detect.py / export.py     # 训练、验证、推理、导出入口
├── requirements.txt
├── ultralytics/
│   ├── cfg/
│   │   ├── default.yaml                          # 默认超参数
│   │   ├── datasets/                             # 常用数据集配置
│   │   └── models/
│   │       ├── 26/yolo26*.yaml                   # YOLO26 基线配置
│   │       └── improve/my_module/
│   │           └── yolo26-LoG-SDS.yaml           # ★ SARNet 模型配置
│   └── nn/
│       ├── extra_modules/
│       │   ├── stem/LoG.py                       # ★ LoGStem
│       │   └── my_module/sds.py                  # ★ SDS
│       └── tasks.py                              # 模型解析（已注册 LoGStem / SDS）
└── LICENSE                                       # AGPL-3.0
```

### 九、引用

如果本工作对你有帮助，欢迎引用（论文信息待补充）：

```bibtex
@article{sarnet,
  title   = {SARNet: Structure-Aware Refinement for Object Detection},
  author  = {<authors>},
  journal = {<journal>},
  year    = {<year>}
}
```

### 十、致谢

* 本仓库基于 [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO26 构建，遵循 **AGPL-3.0** 许可。
* `LoGFilter` 中高斯拉普拉斯滤波器的设计参考：<https://arxiv.org/abs/2503.14012>。

### 十一、许可

本项目采用 **AGPL-3.0** 许可，详见 [LICENSE](LICENSE)。
如用于商业用途，请遵守 AGPL-3.0 的相关条款。

---

## English

### Overview

**SARNet** is a structure-aware object detector built on **Ultralytics YOLO26**. It keeps the
high-frequency structure that plain strided convolutions discard, using two drop-in modules:

| Module | Role | File |
| --- | --- | --- |
| **LoGStem** | Laplacian-of-Gaussian edge-enhanced stem. A 7×7 projection, a fixed LoG kernel, a Gaussian smoothing branch and a dual-path structure downsampler produce a `H/4` feature map that preserves edges and contours. | [`LoG.py`](ultralytics/nn/extra_modules/stem/LoG.py) |
| **SDS** | Semantic-guided Dual-path Structure downsampling. Decouples abstract semantics (strided conv) from directional structure (average pooling + 3×3 / horizontal / vertical depthwise convs), and re-weights the structure path with a channel-wise semantic gate. Replaces the strided conv at P3/P4/P5. | [`sds.py`](ultralytics/nn/extra_modules/my_module/sds.py) |

Model config: [`yolo26-LoG-SDS.yaml`](ultralytics/cfg/models/improve/my_module/yolo26-LoG-SDS.yaml)

### Quick start

```bash
git clone https://github.com/yujianfwq/SARNet.git && cd SARNet
pip install -r requirements.txt && pip install -e .

python train.py --data /path/to/dataset/data.yaml --scale n --epochs 300 --batch 8 --device 0
python val.py   --weights runs/sarnet/train/weights/best.pt --data /path/to/dataset/data.yaml
python detect.py --weights runs/sarnet/train/weights/best.pt --source ultralytics/assets/bus.jpg
```

### Measured parameters

| Model | Params | Δ vs. baseline |
| --- | --- | --- |
| YOLO26n / **SARNet-n** | 2,572,280 / **2,548,112** | −0.94% |
| YOLO26s / **SARNet-s** | 10,009,784 / **9,771,752** | −2.38% |
| YOLO26m / **SARNet-m** | 21,896,248 / **20,539,032** | −6.20% |

### License and acknowledgements

Released under **AGPL-3.0** (see [LICENSE](LICENSE)), inheriting the license of the
[Ultralytics](https://github.com/ultralytics/ultralytics) YOLO26 codebase it is built upon.
The Laplacian-of-Gaussian filter design references <https://arxiv.org/abs/2503.14012>.
