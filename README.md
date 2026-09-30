# Code Description

This repository contains the implementation code for the experiments presented in the paper.

## File Description

### `CWRU_System.py`

Main implementation for the CWRU bearing fault diagnosis experiments.

This script includes the complete proposed framework, including data preprocessing, multimodal feature extraction, self-supervised representation pre-training, supervised fault classification, knowledge distillation, and performance evaluation.

---

### `CWRU_Ablation.py`

Implementation of the ablation experiments on the CWRU dataset.

This script evaluates the contribution of different components of the proposed framework through controlled model comparisons.

---

### `PU_System.py`

Implementation of the fault diagnosis experiments on the Paderborn University (PU) bearing dataset.

This script applies the proposed diagnosis framework to the PU dataset for additional experimental evaluation.

---

### `Synth_System.py`

Implementation of the main experiments on the synthetic dataset.

This script is used to evaluate the proposed framework under controlled synthetic-data conditions.

---

### `Synth_Ablation.py`

Implementation of ablation experiments on the synthetic dataset.

This script evaluates the effects of different components of the proposed method under controlled experimental conditions.

---

### `Few_Label.py`

Implementation of the few-label learning experiments.

This script evaluates the performance of the proposed method under different amounts of labeled training data and is used to analyze label efficiency.

---

### `Paired t-test.py`

Implementation of the paired statistical significance tests.

This script compares the experimental results of different methods using paired t-tests and reports the corresponding statistical significance results.

---

## Summary

| File | Function |
|---|---|
| `CWRU_System.py` | Main CWRU fault diagnosis experiments |
| `CWRU_Ablation.py` | CWRU ablation experiments |
| `PU_System.py` | PU dataset fault diagnosis experiments |
| `Synth_System.py` | Synthetic dataset main experiments |
| `Synth_Ablation.py` | Synthetic dataset ablation experiments |
| `Few_Label.py` | Few-label learning experiments |
| `Paired t-test.py` | Paired statistical significance analysis |
