# DTFNet: A Dual-Modal Time-Frequency Fusion Network for Non-Stationary Time Series Modeling

Official implementation of:

> **DTFNet: A Dual-Modal Time-Frequency Fusion Network for Non-Stationary Time Series Modeling**  
> Caixia Wang, Fan Zhang, Xiaofeng Zhang, Hua Wang  
> *Knowledge-Based Systems*, Vol. 343, Article 116022, 2026  
> DOI: https://doi.org/10.1016/j.knosys.2026.116022

DTFNet is designed for **non-stationary multivariate time series forecasting**. It jointly models frequency-domain coupling, time-varying multi-scale temporal patterns, and robustness to abnormal disturbances.

---

## Highlights

DTFNet contains three main components:

1. **WFD-MHA: Wavelet-Domain Frequency-Band Decoupled Multi-Head Attention**
   - Decomposes the input sequence into multiple frequency bands using DWT.
   - Models correlations within each frequency band.
   - Explicitly captures nonlinear interactions across frequency bands.
   - Preserves inter-variable dependency information while reducing interference between heterogeneous frequency components.

2. **D3SConv: Dynamic Dilated Depthwise Separable Convolution**
   - Divides sequences into overlapping temporal patches.
   - Uses parallel dilated depthwise convolutions to capture local patterns at multiple temporal scales.
   - Aggregates multi-scale information with learnable weights.
   - Uses pointwise convolution to propagate information across temporal segments.

3. **Dynamic Huber Loss**
   - Dynamically adjusts the loss curvature through a learnable/adaptive threshold.
   - Reduces the influence of abnormal disturbances while maintaining fitting accuracy on normal samples.

The paper evaluates DTFNet on **11 real-world forecasting datasets**. DTFNet achieves the best performance in **60 out of 72 experimental settings** reported in the paper.

---

## Repository Structure

```text
DTFNet/
├── data_provider/        # Data loading and preprocessing
├── dataset/              # Dataset directory
├── exp/                  # Experiment pipeline
├── layers/               # Core network layers and modules
├── models/               # Forecasting models, including DTFNet
├── scripts/              # Reproduction scripts for benchmark datasets
├── utils/                # Utility functions
├── requirements.txt      # Python dependencies
├── result.txt            # Experiment/result records
├── run_longExp.py        # Main entry for long-term forecasting experiments
└── README.md
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/luckycaicai1224/DTFNet.git
cd DTFNet
```

### 2. Install dependencies

The repository provides a `requirements.txt` file.

```bash
pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu118
```

For the most reliable reproduction, we recommend creating a clean Python environment before installing the dependencies.

---

## Data Preparation

The experiments follow commonly used long-term time series forecasting benchmarks.

You can download the benchmark datasets from the Autoformer dataset collection:

https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy

Create a dataset directory if it does not already exist:

```bash
mkdir -p dataset
```

Place the downloaded CSV files under `./dataset/` following the paths expected by the scripts in `./scripts/DTFNet/long_term_forecast/`.

Before running an experiment, please check the corresponding shell script and confirm that `root_path` and `data_path` match your local dataset location.

---

## Quick Start

All benchmark scripts are provided under:

```text
./scripts/DTFNet/long_term_forecast/
```

For example, to run DTFNet on **ETTh1**:

```bash
bash ./scripts/DTFNet/long_term_forecast/etth1.sh
```

We recommend reproducing the provided scripts **without changing the hyperparameters first**. After confirming the baseline result, you can modify prediction length, input length, batch size, learning rate, or other settings for additional experiments.

---

## Reproducing the Paper Experiments

A practical reproduction workflow is:

1. Install dependencies from `requirements.txt`.
2. Download and place the datasets under `./dataset/`.
3. Select the corresponding script in `./scripts/DTFNet/long_term_forecast/`.
4. Verify dataset paths in the script.
5. Run the script without modifying the paper configuration.
6. Compare the reproduced results with the records in `result.txt` and the results reported in the paper.

Example:

```bash
bash ./scripts/DTFNet/long_term_forecast/etth1.sh
```

If your reproduced numbers differ noticeably from the reported results, we recommend checking:

- dataset version and file path;
- data split and normalization settings;
- Python / PyTorch / CUDA environment;
- GPU numerical differences;
- random initialization;
- command-line arguments and script hyperparameters.

---

## Main Experimental Findings

The paper reports that:

- DTFNet is evaluated on **11 real-world datasets** covering multiple forecasting scenarios.
- It achieves the best result in **60 / 72** reported experimental configurations against the compared baselines.
- The dynamic Huber strategy improves robustness under noisy conditions, with reported average reductions of **2.54% in MSE** and **3.01% in MAE** compared with the traditional MSE objective in the corresponding noise-robustness study.
- The results support the complementary roles of frequency-band decoupling, multi-scale temporal modeling, cross-variable interaction, and robust optimization.

For complete dataset-level and horizon-level results, please refer to the paper and `result.txt`.

---

## Model Overview

Conceptually, DTFNet follows a dual-path design:

```text
                         ┌─────────────────────────────┐
                         │      Input Time Series      │
                         └──────────────┬──────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    │                                       │
          Frequency-domain Path                    Time-domain Path
                    │                                       │
              DWT Decomposition                     Temporal Patching
                    │                                       │
                WFD-MHA                                D3SConv
                    │                                       │
         Band / Cross-band Modeling              Multi-scale Local Modeling
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        │
                              Dual-modal Fusion
                                        │
                               Forecasting Output
                                        │
                             Dynamic Huber Objective
```

For implementation details, please refer to the code in `models/` and `layers/`.

---

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@article{wang2026dtfnet,
  title={DTFNet: A dual-modal time-frequency fusion network for non-stationary time series modeling},
  author={Wang, Caixia and Zhang, Fan and Zhang, Xiaofeng and Wang, Hua},
  journal={Knowledge-Based Systems},
  pages={116022},
  year={2026},
  publisher={Elsevier}
}
```

---

## Acknowledgements

We sincerely thank the authors of the following open-source repositories for their valuable codebases and benchmark implementations:

- [CPAT](https://github.com/linxi20/CPAT)
- [TimeFilter](https://github.com/TROUBADOUR000/TimeFilter)
- [TimeKAN](https://github.com/huangst21/TimeKAN)
- [TFP-Mixer](https://github.com/SDUYanDong/TFP-Mixer)
- [PDF](https://github.com/Hank0626/PDF)
- [PatchTST](https://github.com/yuqinie98/PatchTST)
- [LTSF-Linear](https://github.com/cure-lab/LTSF-Linear)

---

## Contact

For questions about the implementation or reproducibility, please open a GitHub issue or contact:

**Caixia Wang**  
ORCID: 0009-0006-9499-1491

---

## Reproducibility Note

This repository is intended to make the main implementation and experimental workflow of DTFNet easier to inspect and reproduce. If you encounter an issue when reproducing the reported results, please include the following information when opening an issue:

- dataset name;
- prediction horizon;
- command or script used;
- Python / PyTorch / CUDA versions;
- GPU model;
- obtained MSE / MAE;
- relevant error log or configuration.
