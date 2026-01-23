# PRU-Net: Prior-Regularized Uncertainty-aware Network

**Dense 3D Point Cloud Reconstruction from Sparse Tactile Sequences**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

> **Abstract:** Reconstructing dense 3D geometry from sparse tactile feedback is a fundamental challenge in robotics, particularly in vision-denied environments. This repository contains the official implementation of **PRU-Net**, a data-driven framework that infers **dense 3D point clouds of contacted surface regions** from extremely sparse tactile sequences (e.g., 2-5 touches). We introduce a **Prior-Regularized Uncertainty Weighting (PR-UW)** strategy to resolve optimization conflicts in Multi-Task Learning (MTL), achieving state-of-the-art reconstruction accuracy and robustness against sensory noise.

## 🌟 Key Features

* **Sparse-to-Dense Reconstruction:** Recovers high-fidelity 3D geometry ($N=800$ points) from as few as 2-5 tactile interactions.
* **Tactile Transformer Backbone:** Efficiently encodes temporal dependencies in sparse tactile sequences using self-attention mechanisms and positional encodings.
* **PR-UW Loss Strategy:** A novel uncertainty-weighting loss with prior regularization to prevent "uncertainty collapse" (the "lazy learner" problem) and ensure balanced convergence between global reconstruction and local attribute estimation.
* **Robust Generalization:** Validated on a diverse dataset containing primitive shapes and complex non-convex objects (e.g., bottles, mugs, toys), demonstrating strong performance on unseen objects and independent interaction trials.

## 🛠️ Installation

### Prerequisites
* Linux or Windows
* Python 3.8+
* PyTorch 1.10+ (CUDA recommended)

### Setup
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/PRU-Net.git](https://github.com/yourusername/PRU-Net.git)
    cd PRU-Net
    ```

2.  **Install dependencies:**
    ```bash
    pip install torch numpy
    # Please install chamferdist and pcgrad manually if not found via pip
    # pip install chamferdist
    # pip install pcgrad
    ```

## 📂 Data Preparation

### Dataset Structure
The project expects the data to be stored in `.pkl` format. Please organize your data directory as follows:

```text
data/
└── dataset2/
    ├── dataset2_addedlabel.pkl  # Main data file containing tactile & geometric data
    └── hand20_ind.pkl           # Index file for sensor channels (mask)
