# Spiking Neural Networks for Video-Based Left Ventricular Ejection Fraction (LVEF) Estimation

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5%2Bcu124-EE4C2C.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-EchoNet--Dynamic-orange.svg)](https://echonet.github.io/dynamic/)

This repository implements a **Convolutional-to-Spiking Neural Network (CNN-to-SNN) conversion and fine-tuning pipeline** for continuous Left Ventricular Ejection Fraction (LVEF) estimation from echocardiogram videos using the **EchoNet-Dynamic** dataset.

The project investigates the conversion of a pretrained 3D video CNN (R(2+1)D-18) into a Leaky Integrate-and-Fire (LIF) Spiking Neural Network, exploring surrogate-gradient fine-tuning, temporal integration, and spike-based computation.

---

## Architecture

The pipeline converts the original R(2+1)D-18 architecture into a spiking network with temporal LIF neuron dynamics.

![CNN-to-SNN Conversion Pipeline](docs/images/conversion_pipeline.png)

### Methodology

1. **Video Preprocessing:** Echocardiogram clips are resized to 112 × 112 and sampled into 32-frame sequences.
2. **BatchNorm Folding:** Batch normalization parameters are folded into preceding convolutional layers to preserve the original network's numerical behavior.
3. **LIF Neuron Insertion:** ReLU activations are replaced with Leaky Integrate-and-Fire neurons.
4. **Input Encoding:** Normalized video tensors are repeatedly presented across simulation timesteps.
5. **Surrogate-Gradient Fine-Tuning:** The converted network is fine-tuned using surrogate gradients to recover predictive performance.

### LIF Neuron Dynamics

The membrane potential evolves according to:

$$
m_t = \beta m_{t-1} + I_t
$$

Spike generation and membrane reset are defined as:

$$
s_t = \mathbb{1}[m_t \geq \theta]
$$

$$
m_t \leftarrow m_t(1-s_t)
$$

The implementation uses a threshold of 0.5, membrane decay of 0.9, and hard reset dynamics.

### End-to-End Workflow

![End-to-End Workflow](docs/images/workflow.png)

---

## Repository Structure

```text
SNN-video-LVEF/
├── echonet/
│   ├── datasets/                 # Video dataloaders and dataset parser
│   ├── snn/                      # SNN architecture and conversion
│   │   ├── conversion.py
│   │   ├── lif.py
│   │   ├── residual.py
│   │   ├── spiking_model.py
│   │   └── surrogate.py
│   └── utils/                    # Training and evaluation utilities
├── scripts/                      # Experiment runners and profiling
├── results/                      # Evaluation outputs and logs
├── docs/
│   └── images/                   # Architecture and workflow figures
├── requirements.txt
├── setup.py
└── README.md
```

---

## Getting Started

### 1. Installation

Clone the repository and create a Python 3.10 environment.

```bash
git clone https://github.com/<your-username>/SNN-video-LVEF.git
cd SNN-video-LVEF

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

### 2. Dataset Preparation

Download the **EchoNet-Dynamic** dataset from the [Stanford AIMI Center](https://echonet.github.io/dynamic/).

Place the videos under:

```text
/path/to/EchoNet-Dynamic/Videos/
```

Ensure `FileList.csv` is present in the dataset root directory.

### 3. Evaluating the CNN Baseline

```bash
python -m echonet.utils.eval_cnn \
  --data_dir /path/to/EchoNet-Dynamic \
  --weights /path/to/checkpoints/best.pt \
  --split test \
  --batch_size 16
```

### 4. Running SNN Evaluation

```bash
python -m echonet.utils.test_compare \
  --data_dir /path/to/EchoNet-Dynamic \
  --weights /path/to/checkpoints/best.pt \
  --finetuned_checkpoint /path/to/checkpoints/l4_40ep_best_e24.pt \
  --split test \
  --out results/eval/test_compare_summary.json
```

### 5. GPU Energy Profiling

GPU-board power consumption can be profiled using `nvidia-smi`:

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

## References

Ouyang, D. et al. (2020). *Video-based AI for beat-to-beat assessment of cardiac function*. Nature, 580, 252–256.

```bibtex
@article{ouyang2020video,
  title={Video-based AI for beat-to-beat assessment of cardiac function},
  author={Ouyang, David and He, Bryan and Ghorbani, Amirata and Yuan, Neal and Ebinger, Joseph and Langlotz, Curtis P and Heidenreich, Paul A and Harrington, Robert A and Liang, David H and Ashley, Euan A and Zou, James Y},
  journal={Nature},
  volume={580},
  number={7802},
  pages={252--256},
  year={2020}
}
```

## License

This project is licensed under the Apache 2.0 License. See [LICENSE](LICENSE).
