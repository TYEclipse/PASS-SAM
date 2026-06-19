<p align="center">
  <img src="assets/ori.png" width="80%">
</p>

# PASS-SAM: Pixel Attention Self-Supervised SAM

**🏆 1st Place Solution for the Jittor Large-scale Unsupervised Semantic Segmentation Challenge**

[![Paper](https://img.shields.io/badge/Paper-CVMJ%202024-blue)](https://doi.org/10.26599/CVM.2025.9450432)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Jittor](https://img.shields.io/badge/Framework-Jittor-red)](https://github.com/Jittor/jittor)

PASS-SAM integrates the Segment Anything Model (SAM) into the PASS self-supervised framework for large-scale unsupervised semantic segmentation. Our approach won **1st place** in the 3rd Jittor AI Challenge (2023).

## 🔥 Highlights

- **SAM-powered pseudo-label refinement** — leverages SAM's zero-shot masks to dramatically improve pseudo-label quality
- **Dual-model ensemble** — ResNet18 + ResNet34 with complementary attention mechanisms
- **Self-attention enhanced pixel attention** — improved feature representation for fine-grained segmentation
- **CRF post-processing** — conditional random fields for boundary refinement
- **409.2M params / 97.8 GMACs** — efficient enough to run on a single GPU

## 🚀 Quick Start

### Installation

```bash
pip install jittor
git clone https://github.com/TYEclipse/PASS-SAM.git
cd PASS-SAM
```

### Demo: Single Image Inference

```python
import jittor as jt
from demo import PASS_SAM

# Load model (downloads checkpoint automatically)
model = PASS_SAM.from_pretrained("TYEclipse/PASS-SAM")

# Run inference
mask = model.segment("your_image.jpg")
mask.save("output.png")
```

Or from command line:

```bash
python demo.py --image your_image.jpg --output output.png
```

### Full Evaluation

```bash
# Download ImageNet-S dataset and place in ./data/test/
python test.py
```

## 📊 Architecture

<p align="center">
  <img src="assets/model1.png" width="45%">
  <img src="assets/model2.png" width="45%">
</p>

**Model 1 (R18)**: PASS framework with ResNet18 backbone, CRF + SAM refined pseudo-labels.

**Model 2 (R34)**: Self-Attention enhanced PASS with ResNet34 backbone.

<p align="center">
  <img src="assets/infer.png" width="70%">
</p>

**Inference pipeline**: R18 + R34 ensemble → CRF → SAM → PerSAM → final prediction.

## 📈 Results

| Method | Backbone | mIoU (50 classes) | 
|--------|----------|-------------------|
| PASS (baseline) | ResNet18 | 23.4 |
| PASS-SAM (ours) | ResNet18 | 28.1 |
| PASS-SAM (ours) | ResNet34 | 29.3 |
| **PASS-SAM Ensemble** | **R18 + R34** | **30.7** |

## 📦 Pretrained Models

| Model | Backbone | Link |
|-------|----------|------|
| PASS-SAM R18 | ResNet18 | [Download](https://github.com/TYEclipse/PASS-SAM/releases) |
| PASS-SAM R34 | ResNet34 | [Download](https://github.com/TYEclipse/PASS-SAM/releases) |

Place checkpoints in:
```
./weight/pass50_r18_bz128_ep400/pixel_finetuning_ep40_lr0.6_sz256/checkpoint.pth.tar
./weight/pass50_r34_bz128_ep400/pixel_finetuning_ep40_lr0.6_sz384/checkpoint.pth.tar
```

## 📝 Citation

If you use PASS-SAM in your research, please cite:

```bibtex
@article{tang2024passsam,
  title={PASS-SAM: Integration of Segment Anything Model for Large-scale Unsupervised Semantic Segmentation},
  author={Tang, Yin and others},
  journal={Computational Visual Media (CVMJ)},
  year={2024},
  doi={10.26599/CVM.2025.9450432}
}
```

## 🛠 Training

```bash
# Prepare ImageNet-S dataset in ./data/train/
bash train.sh
```

Training configuration: see `options.py` and `scripts/pass_*.sh`.

## 📚 Reference

- [PASS](https://github.com/LUSSeg/PASS) — pixel attention self-supervised baseline
- [SAM](https://github.com/facebookresearch/segment-anything) — Segment Anything Model
- [PerSAM](https://github.com/ZrrSkywalker/Personalize-SAM) — Personalize SAM
- [EANet](https://github.com/MenghaoGuo/EANet) — efficient attention segmentation head

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>中文 | <a href="#中文版">跳转到中文版</a></sub>
</p>

---

<a name="中文版"></a>

## 🇨🇳 中文版

PASS-SAM 将 Segment Anything Model (SAM) 集成到 PASS 自监督框架中，用于大规模无监督语义分割。本方案在**第三届计图人工智能挑战赛**中获得**第一名**（2023年），论文发表于 CVMJ 2024。

### 方案核心

- **SAM 伪标签优化**：利用 SAM 零样本分割能力提升伪标签质量
- **双模型集成**：ResNet18（PASS框架）+ ResNet34（Self-Attention增强）
- **CRF 后处理**：条件随机场优化边界
- **多阶段推理**：集成→CRF→SAM→PerSAM 四步优化

### 快速开始

```bash
pip install jittor
git clone https://github.com/TYEclipse/PASS-SAM.git
python demo.py --image your_image.jpg
```

### 引用

```bibtex
@article{tang2024passsam,
  title={PASS-SAM: Integration of Segment Anything Model for Large-scale Unsupervised Semantic Segmentation},
  author={唐印 等},
  journal={计算可视媒体 (CVMJ)},
  year={2024}
}
```
