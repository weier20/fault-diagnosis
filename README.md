# Code Description

This repository contains the official PyTorch implementation for the multi-modal bearing fault diagnosis framework (integrating Self-Supervised Contrastive Learning and Knowledge Distillation) presented in the paper.

---

## Datasets & Strict Cross-Condition Split Manifests

To strictly prevent data leakage and evaluate cross-condition generalization, both datasets are partitioned at the **source-file level** across distinct operating conditions (loads/speeds) into mutually exclusive Training, Validation, and Test sets[cite: 14, 15, 16, 17]. Each raw vibration signal is segmented into non-overlapping windows of length `1024`[cite: 14, 15].

### 1. CWRU Bearing Dataset

The Case Western Reserve University (CWRU) Bearing Dataset (12k Drive End Bearing Fault Data, Fault Diameter `0.007"`) is used for primary fault diagnosis, few-shot learning, and ablation experiments[cite: 11, 13, 15, 16].

* **Official download page:** [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter/download-data-file)

#### CWRU Data Split Manifest (File Name / Window Count)[cite: 16]

| Motor Load (HP) / Speed (rpm) | Normal (Label 0) | Inner Race (Label 1) | Outer Race @6:00 (Label 2) | Outer Race @3:00 (Label 2) | Outer Race @12:00 (Label 2) | Ball (Label 3) | Dataset Split / Total Windows |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0 HP / 1797** | `97.mat` / 238 | `105.mat` / 118 | `130.mat` / 119 | `144.mat` / 119 | `156.mat` / 119 | `118.mat` / 119 | **Training Set**<br>(Total: 1897) |
| **1 HP / 1772** | `98.mat` / 472 | `106.mat` / 119 | `131.mat` / 119 | `145.mat` / 118 | `158.mat` / 119 | `119.mat` / 118 | *(Combined 0 & 1 HP)* |
| **2 HP / 1750** | `99.mat` / 472 | `107.mat` / 119 | `132.mat` / 118 | `146.mat` / 118 | `159.mat` / 119 | `120.mat` / 118 | **Validation Set**<br>(Total: 1064) |
| **3 HP / 1730** | `100.mat` / 474 | `108.mat` / 120 | `133.mat` / 119 | `147.mat` / 119 | `160.mat` / 119 | `121.mat` / 118 | **Test Set**<br>(Total: 1069) |

---

### 2. Paderborn University (PU) Bearing Dataset

The Paderborn University (PU) Bearing Dataset (64 kHz vibration data) is used to validate the generalization capability of the proposed framework under complex real-world damages[cite: 14, 17].

* **Official download page:** [PU Bearing Data Center](https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter/data-sets-and-download)

#### PU Data Split Manifest (File Path / Window Count)[cite: 14, 17]

| Speed (rpm) / Torque (Nm) / Force (N) | Normal `K001` (Label 0) | Inner Race `KI04` (Label 1) | Outer Race `KA04` (Label 2) | Compound Fault `KB23` (Label 3) | Dataset Split / Total Windows |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **900 / 0.7 / 1000** | `K001\N09_M07_F10_K001_1.mat`<br>(250) | `KI04\N09_M07_F10_KI04_1.mat`<br>(250) | `KA04\N09_M07_F10_KA04_1.mat`<br>(250) | `KB23\N09_M07_F10_KB23_1.mat`<br>(250) | **Training Set**<br>(Total: 2000) |
| **1500 / 0.1 / 1000** | `K001\N15_M01_F10_K001_1.mat`<br>(250) | `KI04\N15_M01_F10_KI04_1.mat`<br>(250) | `KA04\N15_M01_F10_KA04_1.mat`<br>(250) | `KB23\N15_M01_F10_KB23_1.mat`<br>(250) | *(Combined 2 conditions)* |
| **1500 / 0.7 / 400** | `K001\N15_M07_F04_K001_1.mat`<br>(252) | `KI04\N15_M07_F04_KI04_1.mat`<br>(250) | `KA04\N15_M07_F04_KA04_1.mat`<br>(250) | `KB23\N15_M07_F04_KB23_1.mat`<br>(250) | **Validation Set**<br>(Total: 1002) |
| **1500 / 0.7 / 1000** | `K001\N15_M07_F10_K001_1.mat`<br>(250) | `KI04\N15_M07_F10_KI04_1.mat`<br>(250) | `KA04\N15_M07_F10_KA04_1.mat`<br>(250) | `KB23\N15_M07_F10_KB23_1.mat`<br>(250) | **Test Set**<br>(Total: 1000) |

---

## File Description

### `main_cwru_tuning.py`
Main implementation for the CWRU bearing fault diagnosis pipeline[cite: 15]. This script includes strict cross-condition data partitioning (0HP/1HP for training, 2HP for validation, 3HP for testing), multi-modal time-frequency feature extraction (1D TS, STFT, Wavelet, GAF), self-supervised contrastive pre-training, supervised teacher fine-tuning, automated grid search for knowledge distillation hyperparameters ($T_d, \alpha$) on the validation set, and final evaluation on the unseen test set[cite: 15].

### `main_pu_tuning.py`
Main implementation for the Paderborn University (PU) bearing dataset[cite: 14]. It incorporates Zero-Crossing Rate (ZCR) vibration signal extraction, strict cross-operating-condition validation splitting, automated hyperparameter tuning via grid search, and comprehensive multi-metric evaluation (accuracy, F1-score, inference latency, model size, and noise robustness)[cite: 14].

### `exp_cwru_few_shot.py`
Implementation of the few-shot learning robustness evaluation on the CWRU dataset across varying labeled data fractions (`1%`, `5%`, `10%`, `20%`, and `100%`) over 5 independent random seeds (`42, 137, 256, 512, 1024`) to analyze label efficiency[cite: 11].

### `exp_cwru_paired_ttest.py`
Implementation of the paired statistical significance tests (`scipy.stats.ttest_rel`) under few-shot settings[cite: 12]. This script compares the lightweight baseline model with the proposed SSL+KD student model across 5 random seeds and outputs the mean, standard deviation, and one-tailed paired t-test P-value[cite: 12].

### `exp_cwru_modality_ablation.py`
Implementation of the comprehensive modality and architectural component ablation study on the CWRU dataset[cite: 13]. It evaluates 11 distinct configurations (including `w/o KD`, `w/o SSL`, single-modality baselines, and leave-one-modality-out variants) and generates bar and radar charts[cite: 13].

### `exp_cwru_kd_ablation.py`
Implementation of the controlled Knowledge Distillation (KD) ablation experiment under the 5%-labeled ultra-lightweight student setting (`[64, 32]`)[cite: 10]. It systematically compares hard-label-only training ($\alpha=0.0$), soft-label-only training ($\alpha=1.0$), and various combinations of distillation temperatures ($T_d \in \{1.0, 3.0, 5.0, 10.0\}$) and loss weights ($\alpha \in \{0.2, 0.6, 0.8\}$)[cite: 10].

### `Synth_System.py` & `Synth_Ablation.py`
Implementations of the main fault diagnosis experiments and component ablation studies under controlled synthetic dataset conditions.

---

## Summary of Scripts

| File | Function |
| :--- | :--- |
| `main_cwru_tuning.py` | Main CWRU fault diagnosis pipeline & validation hyperparameter tuning[cite: 15] |
| `main_pu_tuning.py` | Main PU fault diagnosis pipeline & validation hyperparameter tuning[cite: 14] |
| `exp_cwru_few_shot.py` | Few-shot label efficiency evaluation across 5 random seeds[cite: 11] |
| `exp_cwru_paired_ttest.py` | Paired t-test statistical significance analysis[cite: 12] |
| `exp_cwru_modality_ablation.py` | Complete modality & SSL/KD component ablation study[cite: 13] |
| `exp_cwru_kd_ablation.py` | Controlled KD temperature ($T_d$) and weight ($\alpha$) ablation study[cite: 10] |
| `Synth_System.py` | Synthetic dataset main experiments |
| `Synth_Ablation.py` | Synthetic dataset ablation experiments |
