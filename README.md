# Protein Variant Classifier with ESM2

This repository contains the implementation of a deep learning model for classifying protein variants as **Gain-of-Function (GOF)** or **Loss-of-Function (LOF)** using the **ESM2 (Evolutionary Scale Modeling)** protein language model.

## Overview

Accurately predicting the functional impact of genetic variants is crucial for clinical genomics. This project leverages the power of pre-trained protein language models to capture the semantic differences between wild-type and mutant protein sequences.

### Key Features
- **ESM2 Backbone**: Utilizes `facebook/esm2_t33_650M_UR50D` for robust feature extraction.
- **Difference Embedding**: Explicitly models the direction of change ($E_{mut} - E_{wt}$) to distinguish GOF from LOF.
- **Class Imbalance Handling**: Implements weighted loss functions to handle real-world data skew (e.g., 9:1 LOF:GOF).
- **Multi-GPU Support**: Built with PyTorch `DistributedDataParallel` (DDP) for efficient training on A100 clusters.

## Installation

```bash
conda create -n protein_classifier python=3.10
conda activate protein_classifier
pip install torch transformers pandas scikit-learn
```

## Usage

### 1. Training the Classifier

You can run the training script locally or on a multi-GPU cluster.

**Local Training (Single GPU/CPU):**
```bash
python code/train_esm_classifier.py --batch_size 2 --max_len 512 --epochs 3
```

**Distributed Training (4x A100):**
```bash
torchrun --nproc_per_node=4 code/train_esm_classifier.py --batch_size 8 --max_len 1024 --epochs 10
```

### 2. Analyzing Predictor Performance

Use the analysis script to evaluate different pathogenicity scores, focusing on patient-centric metrics like Top-1 Accuracy.

```bash
python code/analyze_scores.py path/to/scores.csv
```

## License

MIT License
