# Information Bottleneck-Driven Cross-Receiver RF Fingerprinting for Physical-Layer Security

## Overview

This work revisits cross-receiver RF fingerprinting (RFF) from an information-theoretic perspective. The proposed framework combines:

- a lightweight **Frequency-Aware Network (FAN)** as the backbone, and
- an **Information Bottleneck (IB)**-based regularization with HSIC as a tractable surrogate for mutual information,

trained with a **two-stage scheme** (pre-training + IB fine-tuning). The framework requires no receiver labels and is validated on the WiSig and LoRa datasets.

---

## Repository Structure

```
IB-RFF/
├── backbones/      # FAN backbone and baseline models
├── dataset/        # Data loading and preprocessing
├── utils/          # HSIC computation, metrics, helpers
├── weights/        # Pre-trained model weights
├── logs/           # Training logs
├── main.py         # Training and evaluation entry
└── run_all.sh      # Script to reproduce all experiments
```

---

## Environment

- Python 3.8
- PyTorch 1.11 + CUDA 11.3
- NVIDIA RTX 3080 Ti (or equivalent)

---

## Datasets

The two public datasets used in this work should be downloaded from their original sources:

- **WiSig**: https://cores.ee.ucla.edu/downloads/datasets/wisig/
- **LoRa** (receiver_drift_dataset): https://ieee-dataport.org/documents/radio-frequency-fingerprint-lora-dataset-multiple-receivers

Please follow the licenses of the original dataset providers.

---

## Quick Start

Reproduce all experiments:

```bash
bash run_all.sh
```

## License

The source code in this repository is released under the **MIT License**.

- The **WiSig and LoRa datasets** are subject to the licenses of their original providers and are **not** redistributed here.
- The **pre-trained weights** in `weights/` are released for **academic and research use only**. Commercial use requires prior written consent from the author.
- The **figures, tables, and text of the associated paper** are © IEEE upon publication.

This code is provided "as is", without warranty of any kind. The author is not liable for any damages arising from its use.

---
