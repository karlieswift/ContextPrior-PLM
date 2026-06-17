<div align="center">

# ContextPrior-PLM

### Natural Vector-guided local-context adaptation of frozen ESM-2 representations for protein stability prediction

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python\&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch\&logoColor=white)]()
[![ESM--2](https://img.shields.io/badge/Encoder-ESM--2%20650M-4B8BBE)]()
[![Task](https://img.shields.io/badge/Task-Protein%20Stability%20Prediction-2E8B57)]()
[![Status](https://img.shields.io/badge/Status-Research%20Code-purple)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey)]()

**ContextPrior-PLM** is a frozen-protein-language-model adapter for protein variant stability prediction.
It preserves ESM-2 residue states in overlapping local-context units and injects sequence-statistical prior information before sequence-level readout.

</div>

---

## Project overview

<p align="center">
  <a href="docs/overview.pdf">
    <img src="assets/overview.png" alt="ContextPrior-PLM overview" width="92%">
  </a>
</p>

<p align="center">
  <b>Click the figure to open the full project overview PDF.</b>
</p>

---

## Why ContextPrior-PLM?

Protein engineering often requires selecting a small number of candidate variants for costly experimental validation. A useful stability predictor should therefore support both accurate stability-score regression and stable-variant prioritization.

Many frozen PLM-based predictors compress residue-level representations into sequence-level summaries before prediction. This early aggregation can weaken local sequence-context signals that are relevant to protein stability.

ContextPrior-PLM addresses this by introducing an intermediate local-context representation before sequence-level readout.

---

## Method at a glance

ContextPrior-PLM combines three components:

1. **Frozen ESM-2 residue representations**
   Residue-level states are extracted from a frozen ESM-2 encoder without fine-tuning the PLM backbone.

2. **Overlapping local-context units**
   Projected residue states are grouped into overlapping windows, preserving local PLM-derived context before global aggregation.

3. **Natural Vector-guided context prior graph**
   Local contexts are connected using alignment-free sequence-statistical similarity derived from Natural Vector descriptors. The graph does not rely on structural coordinates or annotated motifs.

The resulting graph information is injected into the local-context field before sequence-level prediction.

---

## Main results

On the SaProtHub protein stability benchmark, ContextPrior-PLM with frozen ESM-2 650M achieved:

| Model                         |   Spearman |       RMSE | Precision@1000 |
| ----------------------------- | ---------: | ---------: | -------------: |
| Strongest frozen-PLM baseline |     0.8427 |     0.5992 |          0.519 |
| **ContextPrior-PLM**          | **0.9255** | **0.4181** |      **0.728** |

Compared with the strongest frozen-PLM baseline, ContextPrior-PLM improved Spearman correlation by **0.0828**, reduced RMSE by **30.2%**, and improved top-stable-variant prioritization from **0.519** to **0.728** in Precision@1000.

---

## Repository structure

```text
ContextPrior-PLM/
├── README.md
├── docs/
│   └── overview.pdf
├── assets/
│   └── overview.png
├── configs/
│   └── contextprior_esm2_650m.yaml
├── data/
│   └── README.md
├── scripts/
│   ├── train_contextprior.py
│   ├── evaluate_contextprior.py
│   └── reproduce_figures.py
├── src/
│   ├── models/
│   ├── data/
│   ├── graph/
│   ├── metrics/
│   └── utils/
├── results/
│   └── README.md
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/karlieswift/ContextPrior-PLM.git
cd ContextPrior-PLM

conda create -n contextprior-plm python=3.10 -y
conda activate contextprior-plm

pip install -r requirements.txt
```

---

## Quick start

### 1. Prepare data

Place the processed SaProtHub stability files under:

```text
data/saprothub/
```

The expected format is described in:

```text
data/README.md
```

### 2. Train ContextPrior-PLM

```bash
python scripts/train_contextprior.py \
  --config configs/contextprior_esm2_650m.yaml
```

### 3. Evaluate the trained model

```bash
python scripts/evaluate_contextprior.py \
  --config configs/contextprior_esm2_650m.yaml \
  --checkpoint checkpoints/contextprior_esm2_650m.pt
```

### 4. Reproduce figures

```bash
python scripts/reproduce_figures.py \
  --result_dir results/
```

---

## Configuration

The main configuration used in the study is provided in:

```text
configs/contextprior_esm2_650m.yaml
```

Key settings include:

| Setting                   |      Value |
| ------------------------- | ---------: |
| Frozen encoder            | ESM-2 650M |
| Window length             |         10 |
| Stride                    |          5 |
| Maximum number of windows |         15 |
| Graph top-k               |          6 |
| Hidden dimension          |        384 |
| Optimizer                 |      AdamW |
| Learning rate             |       2e-5 |
| Batch size                |         16 |
| Random seed               |         42 |

---

## Reproducibility

This repository provides:

* model implementation;
* training and evaluation scripts;
* configuration files;
* figure-reproduction scripts;
* benchmark result tables;
* documentation for expected data formats.

The ESM-2 backbone is kept frozen in all main experiments. Only the ContextPrior-PLM adapter and prediction head are trained.

---

## Data and code availability

Data and code used in this study are available at:

```text
https://github.com/karlieswift/ContextPrior-PLM
```

An archived version of the repository and accompanying data is available from Zenodo:

```text
https://sandbox.zenodo.org/records/514784
```

For the final publication version, the sandbox Zenodo record should be replaced by a formal Zenodo DOI.

---

## Citation

If you use ContextPrior-PLM in your research, please cite:

```bibtex
@article{wang2026contextpriorplm,
  title   = {Natural Vector-guided local-context adaptation of frozen ESM-2 representations for protein stability prediction},
  author  = {Wang, Hao and Hu, Guoqing and Zhao, Xin and Yau, Stephen S.-T.},
  journal = {Bioinformatics},
  year    = {2026}
}
```

---

## Contact

For questions, issues, or collaboration requests, please open a GitHub issue or contact the corresponding authors listed in the manuscript.

---

<div align="center">

**ContextPrior-PLM: preserving local PLM-derived context for stability prediction and stable-variant prioritization.**

</div>
