# Spiking Neural Networks for Video-Based Left Ventricular Ejection Fraction (LVEF) Estimation

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5%2Bcu124-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-EchoNet--Dynamic-orange.svg)](https://echonet.github.io/dynamic/)

This repository implements a **Convolutional-to-Spiking Neural Network (CNN-to-SNN) conversion and fine-tuning pipeline** for continuous Left Ventricular Ejection Fraction (LVEF) regression from echocardiogram videos using the **EchoNet-Dynamic** dataset.

The project evaluates whether converting a 3D video CNN (R(2+1)D-18) into a Leaky Integrate-and-Fire (LIF) Spiking Neural Network can preserve diagnostic accuracy while achieving sparse, energy-efficient temporal processing.

---

## 📌 Key Findings & Executive Summary

1. **Accuracy Gap & Fine-Tuning Recovery**:
   * Pretrained CNN baseline achieves **$R^2 = 0.7791$** ($\text{MAE} = 4.329$) on the 1,276-video held-out test split.
   * Direct SNN conversion causes significant performance loss: **$R^2 = 0.4798$** ($\text{MAE} = 6.850$) at $T = 50$.
   * Surrogate-gradient fine-tuning recovers ~38% of the conversion loss at $T = 50$ ($R^2 = 0.5950$, $\text{MAE} = 5.822$), and achieves peak performance at **$T = 10$** with **$R^2 = 0.6927$** and **$\text{MAE} = 4.975$** (within $0.646$ MAE points of the CNN).

2. **Non-Monotonic Timestep Behavior**:
   * SNN accuracy does not monotonically improve with larger $T$. Error is lowest at $T = 10$ and degrades toward $T = 50$.
   * Below $T = 6$, predictions collapse catastrophically ($R^2 < 0$ for $T \le 4$), as deep LIF layers fail to accumulate sufficient charge to emit spikes.

3. **The Energy Discrepancy: Modeled vs. Measured**:
   * **Analytical Proxy (Modeled)**: Predicts a **45.6% (FP32)** and **83.1% (INT32)** compute energy reduction at $T = 10$ based on synaptic Accumulate (AC) vs. Multiply-Accumulate (MAC) theoretical cost.
   * **Physical Telemetry (Measured on NVIDIA RTX 4090)**: Hardware power sampling reveals the fine-tuned SNN consumes **$45.03\text{ J/video}$** vs. **$2.89\text{ J/video}$** for the CNN (**15.6× higher**). Standard GPUs process timesteps sequentially via dense CUDA kernels without sparse event-driven hardware acceleration.

---

## 🏗️ Architecture & Conversion Pipeline

The workflow converts the Stanford EchoNet-Dynamic R(2+1)D-18 CNN into an SNN with temporal LIF dynamics:

![CNN-to-SNN Pipeline](docs/images/conversion_pipeline.png)

### End-to-End Methodology

![End-to-End Workflow](docs/images/workflow.png)

1. **Data Preprocessing**: Echocardiogram clips downsampled to $112 \times 112$, sampled at period 2 (32 frames spanning $\sim 1.3\text{ s}$ / one cardiac cycle), normalized by channel statistics ($\mu = [32.17, 32.34, 32.70]$, $\sigma = [49.57, 49.71, 50.08]$).
2. **BatchNorm Folding**: 37 BatchNorm layers folded into preceding convolutions prior to activation swapping. Verified maximum numerical difference: $< 0.00023$.
3. **LIF Neuron Insertion**: 34 ReLU activations replaced with Leaky Integrate-and-Fire neurons:
   $$m_t = \beta \, m_{t-1} + I_t, \quad s_t = \mathbb{1}[m_t \ge \theta], \quad m_t \leftarrow m_t \, (1 - s_t)$$
   * Threshold $\theta = 0.5$, membrane decay $\beta = 0.9$, hard reset to zero.
4. **Input Presentation**: Direct static repetition of the normalized tensor across all $T$ timesteps (Bernoulli encoding was evaluated and discarded due to catastrophic information collapse).
5. **Surrogate Gradient Fine-Tuning**:
   * Fast-sigmoid surrogate gradient on backward pass; exact binary spikes on forward pass.
   * Freezing strategy: `stem`, `layer1`, `layer2`, and `layer3` frozen; only `layer4` and the linear regression head are trainable (20 parameter tensors). Training Layer 3 + Layer 4 diverged at scale.

---

## 📊 Benchmark Results

### 1. Main Performance Comparison (Held-Out Test Split, $N = 1,276$)

All models evaluated on identical test clips (batch size 1, 1,276 videos):

| Model | Timesteps ($T$) | MAE (% EF) ↓ | RMSE (% EF) ↓ | $R^2$ ↑ | Latency (ms/video) | Active Spike Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CNN Baseline** (`best.pt`) | — | **4.329** | **5.747** | **0.7791** | **9.4 ± 0.0** | Dense (Analog) |
| **Direct SNN** (Converted) | 50 | 6.850 | 8.820 | 0.4798 | 648.3 ± 0.8 | 18.55% |
| **Fine-Tuned SNN** (`l4_40ep_best_e24.pt`) | 50 | 5.822 | 7.782 | 0.5950 | 648.7 ± 0.8 | 18.57% |
| **Fine-Tuned SNN (Optimal)** | **10** | **4.975** | **6.779** | **0.6927** | **131.5 ± 0.3** | **18.30%** |

---

### 2. Timestep Ablation Study ($T = 1$ to $50$)

Evaluating the fine-tuned checkpoint across inference horizons reveals that $T = 10$ provides the optimal trade-off between temporal integration and error accumulation:

<p align="center">
  <img src="docs/images/timestep_ablation_mae.png" width="48%" />
  <img src="docs/images/timestep_ablation_r2_sub10.png" width="48%" />
</p>

| Inference Timesteps ($T$) | Fine-Tuned $R^2$ | Direct SNN $R^2$ | Test MAE | Test RMSE | Latency (ms) | Spikes/Video ($10^6$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | -4.6663 | — | 27.12 | 29.11 | — | — |
| **2** | -2.6304 | — | 21.48 | 23.30 | — | — |
| **4** | -0.1832 | — | 11.84 | 13.30 | — | — |
| **6** | 0.5302 | — | 6.72 | 8.38 | — | — |
| **8** | 0.6742 | — | 5.21 | 6.98 | — | — |
| **10** | **0.6927** | 0.3447 | **4.975** | **6.779** | **131.5 ± 0.3** | **226.4** |
| **20** | 0.6375 | 0.4396 | 5.481 | 7.363 | 261.2 ± 1.0 | 457.0 |
| **30** | 0.6125 | 0.4634 | 5.694 | 7.613 | 390.3 ± 0.6 | 687.5 |
| **50** | 0.5950 | 0.4798 | 5.822 | 7.782 | 648.4 ± 0.9 | 1,148.8 |

> **Takeaway**: Moving from $T = 50$ to $T = 10$ reduces inference latency by **79.7%**, cuts spike generation by **80.3%**, and **improves $R^2$ by +0.0977**.

<p align="center">
  <img src="docs/images/timestep_ablation_metrics.png" width="48%" />
  <img src="docs/images/latency_and_spikes.png" width="48%" />
</p>

---

## ⚡ Energy Analysis: Theory vs. Reality

<p align="center">
  <img src="docs/images/analytical_energy_proxy_T10.png" width="48%" />
  <img src="docs/images/gpu_measured_energy_T10.png" width="48%" />
</p>

### 1. Analytical Operation-Based Energy Proxy (Theoretical Model)

Calculated using standard 45 nm CMOS energy weights (Horowitz, 2014):
* $\text{MAC}_{FP32} = 4.6\text{ pJ}$, $\text{AC}_{FP32} = 0.9\text{ pJ}$ (5.1× ratio)
* $\text{MAC}_{INT32} = 3.2\text{ pJ}$, $\text{AC}_{INT32} = 0.1\text{ pJ}$ (32.0× ratio)

$$E_{CNN} = N_{MAC} \cdot E_{MAC}, \quad E_{SNN} = T \cdot \left(N_{MAC}^{SNN} E_{MAC} + N_{AC}^{SNN} E_{AC}\right) + N_{upd} E_{MAC}$$

| Model | Timesteps ($T$) | FP32 Proxy (mJ/video) | vs. CNN | INT32 Proxy (mJ/video) | vs. CNN |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CNN Baseline** | — | 372.8 | Baseline | 259.3 | Baseline |
| **Direct SNN** | 10 | 202.1 | **-45.8%** | 43.6 | **-83.2%** |
| **Fine-Tuned SNN** | 10 | 202.8 | **-45.6%** | 43.7 | **-83.1%** |
| **Direct SNN** | 50 | 1008.0 | +170.4% | 217.9 | -16.0% |
| **Fine-Tuned SNN** | 50 | 1009.9 | +170.9% | 218.1 | -15.9% |

### 2. Physical GPU-Board Energy Telemetry (NVIDIA RTX 4090)

Continuously sampled via `nvidia-smi` at 100 ms intervals over the full 1,276 test videos with a 30-second idle baseline (48.6 W) subtracted:

| Model | Average Power (W) | Duration (s) | Gross Energy (J/video) | Idle-Subtracted Energy (J/video) | Energy vs. CNN |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CNN Baseline** | 156.3 W | 34.2 s | 4.19 J | **2.89 J** | **1.0×** |
| **Direct SNN ($T=10$)** | 312.4 W | 211.7 s | 51.83 J | **43.76 J** | **15.2×** |
| **Fine-Tuned SNN ($T=10$)** | 319.5 W | 212.1 s | 53.11 J | **45.03 J** | **15.6×** |

> **Critical Insight**: On non-neuromorphic hardware (GPUs), simulated SNNs require unrolling dense matrix multiplications for each timestep $T$. Because standard GPUs do not provide event-driven zero-skipping, the computational overhead of multiple iterations dominates, making GPU energy scale unfavorably with $T$. SNN energy advantages require dedicated neuromorphic or event-driven hardware (e.g., Intel Loihi, SynSense).

---

## 📁 Repository Structure

```text
SNN-video-LVEF/
├── echonet/                      # Core EchoNet-Dynamic package
│   ├── datasets/                 # Video dataloaders & EchoNet dataset parser
│   │   └── echo.py
│   ├── snn/                      # SNN architecture & conversion components
│   │   ├── conversion.py         # BatchNorm folding & ReLU-to-LIF mapping
│   │   ├── lif.py                # Leaky Integrate-and-Fire neuron implementation
│   │   ├── residual.py           # Spiking residual block definitions
│   │   ├── spiking_model.py      # Converted SpikingModel wrapper
│   │   └── surrogate.py          # Fast-sigmoid surrogate gradient functions
│   └── utils/                    # Training, evaluation & ablation runners
│       ├── eval_cnn.py           # Baseline CNN evaluator
│       ├── eval_snn.py           # SNN evaluator
│       ├── eval_snn_timesteps.py # Timestep evaluation (T=1..10)
│       ├── snn_common.py         # Shared metric calculation & energy models
│       ├── test_compare.py       # Multi-model comparative test runner
│       └── train_snn_v2.py       # Surrogate-gradient fine-tuning engine
│
├── results/                      # Stored experimental evaluation outputs
│   ├── energy/                   # Physical power & analytical ops JSONs
│   │   ├── energy_T10_measured_ops.json
│   │   └── gpu_energy_T10_full_test.json
│   ├── eval/                     # Test set summary reports & timestep sweeps
│   │   ├── smoke_compare_summary.json
│   │   ├── test_compare_e24_summary.json
│   │   └── test_timestep_e24_summary.json
│   └── logs/                     # Evaluation stdout logs
│       └── snn_eval_log.txt
│
├── scripts/                      # Standalone measurement and plotting scripts
│   ├── measure_energy_T10.py     # Analytical operation & proxy calculator
│   ├── measure_gpu_energy_T10.py # Physical nvidia-smi power profiler
│   ├── test_gpu.py               # GPU availability & CUDA sanity check
│   ├── plot_complexity.py        # Model complexity plotting utility
│   └── run_experiments.sh       # Batch experiment bash runner
│
├── docs/                         # Documentation assets & figures
│   └── images/                   # High-resolution benchmark figures
│
├── requirements.txt              # Environment dependencies
├── setup.py                      # Package installation config
└── README.md                     # Project documentation
```

---

## 🚀 Getting Started

### 1. Installation & Environment Setup

Clone the repository and set up a Python 3.10 virtual environment:

```bash
git clone https://github.com/<your-username>/SNN-video-LVEF.git
cd SNN-video-LVEF

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# Install project dependencies and local package in editable mode
pip install -r requirements.txt
pip install -e .
```

### 2. Dataset Preparation

Download the **EchoNet-Dynamic** dataset from the [Stanford AIMI Center](https://echonet.github.io/dynamic/):
* Place the videos under `/path/to/EchoNet-Dynamic/Videos/`
* Ensure `FileList.csv` is present in the dataset root directory.

### 3. Evaluating the Baseline CNN

```bash
python -m echonet.utils.eval_cnn \
  --data_dir /path/to/EchoNet-Dynamic \
  --weights /path/to/checkpoints/best.pt \
  --split test \
  --batch_size 16
```

### 4. Running Timestep Ablations ($T = 10, 20, 30, 50$)

```bash
python -m echonet.utils.test_compare \
  --data_dir /path/to/EchoNet-Dynamic \
  --weights /path/to/checkpoints/best.pt \
  --finetuned_checkpoint /path/to/checkpoints/l4_40ep_best_e24.pt \
  --split test \
  --out results/eval/test_compare_summary.json
```

### 5. Profiling GPU-Board Power & Energy

To measure physical GPU-board power on the full test split using `nvidia-smi`:

```bash
python scripts/measure_gpu_energy_T10.py \
  --data_dir /path/to/EchoNet-Dynamic \
  --weights /path/to/checkpoints/best.pt \
  --finetuned_checkpoint /path/to/checkpoints/l4_40ep_best_e24.pt \
  --split test \
  --sample_interval 0.1 \
  --idle_seconds 30 \
  --out results/energy/gpu_energy_T10_full_test.json
```

---

## 🔬 Citation & Reference

If you use this work or the EchoNet-Dynamic dataset, please cite the primary dataset paper and this study:

```bibtex
@article{ouyang2020video,
  title={Video-based AI for beat-to-beat assessment of cardiac function},
  author={Ouyang, David and He, Bryan and Ghorbani, Amirata and Yuan, Neal and Ebinger, Joseph and Langlotz, Curtis P and Heidenreich, Paul A and Harrington, Robert A and Liang, David H and Ashley, Euan A and Zou, James Y},
  journal={Nature},
  volume={580},
  number={7802},
  pages={252--256},
  year={2020},
  publisher={Nature Publishing Group}
}
```

```bibtex
@misc{snn_video_lvef_2026,
  author = {Priyanka and collaborators},
  title = {Spiking Neural Networks for Video-Based Left Ventricular Ejection Fraction Estimation},
  year = {2026},
  howpublished = {\url{https://github.com/<your-username>/SNN-video-LVEF}}
}
```
