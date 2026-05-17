# Verite!: Cross-Domain Deception Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.20256648-blue.svg)](https://doi.org/10.5281/zenodo.20256648)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![HuggingFace](https://img.shields.io/badge/Model-HuggingFace-orange.svg)](https://huggingface.co/Daxlia/verite)

> **Verite!** is a cross-domain deception detection system built on [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base), combining spectral features, hyperspherical classification, local consistency modeling, and multi-task domain learning.  
> Evaluated on the [DIFrauD](https://huggingface.co/datasets/difraud/difraud) benchmark (7 domains, ~103K samples).

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Citation](#citation)
- [AI Disclosure](#ai-disclosure)
- [License](#license)

---

## Overview

Deception detection is a challenging NLP task that requires generalizing across radically different domains (fake news, phishing, product reviews, SMS spam, political statements, job scams, Twitter rumours). Verite! addresses this by combining a powerful pre-trained encoder with domain-aware multi-task learning and several auxiliary objectives designed to capture both semantic and structural deception signals.

**Key contributions:**

- **Spectral features**: top-*k* FFT magnitudes + spectral centroid + entropy over the token sequence, capturing frequency-domain patterns of deception
- **Local Consistency Module**: segment-level cross-attention to detect internal contradictions
- **HypersphericalHead**: prototype-based classification on the unit hypersphere, robust to inter-domain shifts
- **Domain MTL**: shared encoder learns domain-specific cues through a positive multi-task head (complementary to adversarial DANN)
- **Multi-sample dropout** (5×) + **EMA** + **AWP** (epochs 4–5): strong regularization stack

---

## Architecture

```text
Input Text ────────────────────────── Linguistic Features (8-d)
    │                                          │
    ▼                                          ▼
ModernBERT-base (149M params, fp32)       ling_proj → feat_emb (512-d)
    │                                          │
    ├── AttentionPooling ──► semantic_proj ──► sem_emb  (512-d) ──┐
    ├── LocalConsistencyModule (4 segs)  ──► cons_emb (256-d) ────┤
    └── SpectralFeatures (top-8 FFT + centroid + entropy) ─► spec (10-d)
                                                                   │
                  Concatenate [sem_emb | feat_emb | cons_emb | spec]  (1290-d)
                                         │
                               LayerNorm → Linear(512) → GELU
                                         │
                         Multi-sample Dropout (5×) → HypersphericalHead
                                         │
                                    Logits (2 classes)
```

**Training objectives:**

- Focal loss (γ=2.0) with class-balanced weights and label smoothing (ε=0.05)
- Supervised Contrastive loss (λ=0.1, τ=0.07) on semantic embeddings
- Domain MTL cross-entropy (λ=0.1) on 7 domain heads

**Optimization:**

- AdamW with Layer-wise LR Decay (LLRD, decay=0.9): encoder LR=1e-5, head LR=1e-4
- Cosine schedule with 8% linear warmup
- Gradient accumulation (×4), gradient clipping (0.7)
- Adversarial Weight Perturbation (AWP, ε=0.001) from epoch 4 onward

---

## Results

Evaluated on the DIFrauD test set (macro-F1, higher is better).

| System | Macro-F1 | AUC-ROC |
| ------ | -------- | ------- |
| Majority class | 0.3792 | 0.5000 |
| TF-IDF + LR | 0.8094 | 0.9079 |
| ModernBERT-base (fine-tuned) | ~0.82 | — |
| **Verite! (ours)** | **0.8512** | **0.9487** |
| SOTA (DIFrauD leaderboard) | 0.904 | — |

> Results obtained with a single seed (seed=42) on 2×NVIDIA T4 GPUs.  
> Multi-seed ensemble (`multi_seed=True`, 3 seeds) is expected to close the gap further.

---

## Installation

```bash
git clone https://github.com/Daxlia/Verite.git
cd Verite
pip install torch>=2.1.0 transformers>=4.47.0 safetensors sentencepiece
pip install scikit-learn pandas numpy tqdm datasets huggingface_hub
```

**Hardware requirements:** 1–2 GPUs with ≥15GB VRAM (tested on 2×T4 16GB).

---

## Usage

### Inference

```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from safetensors.torch import load_file

from VeriteTrainer import DeceptionReasoningModel, Config, DeceptionDataset, collate_fn

cfg       = Config()
tokenizer = AutoTokenizer.from_pretrained("Daxlia/verite")

model = DeceptionReasoningModel(cfg)
model.load_state_dict(load_file("model.safetensors"))
model.eval()

texts = ["This is a suspicious message claiming you've won a prize."]

dataset = DeceptionDataset(texts, [0] * len(texts), tokenizer, cfg)
loader  = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

with torch.no_grad():
    for batch in loader:
        out  = model(input_ids=batch["input_ids"],
                     attention_mask=batch["attention_mask"],
                     ling_feats=batch["ling_feats"])
        prob = torch.softmax(out["logits"], dim=-1)[:, 1]
        for t, p in zip(texts, prob.tolist()):
            print(f"P(deceptive) = {p:.4f} | {t}")
```

### Training from scratch

Training was run on a Kaggle notebook with the following setup:

- Accelerator: **2×T4 GPU**
- Dataset input: `difraud/difraud` (added via HuggingFace Hub integration)
- `VeriteTrainer.py` uploaded as a private Kaggle dataset input
- Runtime: **~13 h** (< 14 h total session)

**Cell 1** — install dependencies:

```python
!pip install transformers>=4.47.0 safetensors sentencepiece
```

**Cell 2** — run training:

```python
exec(open("/kaggle/input/verite/VeriteTrainer.py").read())
```

Then: **Save Version → Run All**.

---

## Training

Training was performed on 2×NVIDIA T4 (16GB each) via Kaggle.

| Hyperparameter | Value |
| --- | --- |
| Base encoder | `answerdotai/ModernBERT-base` |
| Total runtime | ~13 h on 2×T4 (< 14 h) |
| Max sequence length | 256 |
| Batch size (effective) | 64 (8 × 2 GPUs × 4 accum steps) |
| Epochs | 5 |
| Encoder LR | 1e-5 |
| Head LR | 1e-4 |
| LLRD decay | 0.9 |
| Warmup | 8% |
| Weight decay | 0.02 |
| Focal γ | 2.0 |
| SupCon λ | 0.1 |
| Domain MTL λ | 0.1 |
| AWP start | Epoch 4 |
| EMA decay | 0.995 |

---

## Citation

If you use Verite! in your research, please cite:

```bibtex
@misc{verite2026,
  author    = {Daxlia},
  title     = {Verite!: Cross-Domain Deception Detection with ModernBERT},
  year      = {2026},
  doi       = {10.5281/zenodo.20256648},
  url       = {https://doi.org/10.5281/zenodo.20256648}
}
```

---

## AI Disclosure

This project was developed with assistance from AI tools:

- **ChatGPT (OpenAI)** — Bug identification, algorithmic suggestions, and research ideation
- **Claude (Anthropic)** — Code implementation, debugging, and writing of documentation files

All AI-generated content was reviewed, validated, and adapted by the author.

---

## License

This project is licensed under the [MIT License](LICENSE).

The base encoder [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) is licensed under Apache 2.0.  
The [DIFrauD dataset](https://huggingface.co/datasets/difraud/difraud) is subject to its own license; refer to the dataset repository.
