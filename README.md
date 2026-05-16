<<<<<<< HEAD
# TSAC 2025/2026: Home Appliance Power Signature Recognition

**Group Members:** Anis Si Salem, Abderrahmane Dilmi, Yacine Kaizra, Khaled Zaabat

---

## Overview

Classify **10 home appliances** from their electrical power consumption time-series — 100 training samples, 1,460 time steps each. A high-dimensional, low-sample-size classification problem.

| Class | Appliance                |
|:-----:|:-------------------------|
| 0     | Mobile Phone (charger)   |
| 1     | Coffee Machine           |
| 2     | Computer Station         |
| 3     | Fridge/Freezer           |
| 4     | Hi-Fi / CD Player        |
| 5     | CFL Lamp                 |
| 6     | Laptop (charger)         |
| 7     | Microwave Oven           |
| 8     | Printers                 |
| 9     | Televisions (LCD/LED)    |

## Approaches

| # | Method                              | Output File               | Accuracy |
|:-:|:------------------------------------|:--------------------------|:--------:|
| 1 | Statistical Features + Random Forest | `predictions_rf.csv`      | ~57%     |
| 2 | ROCKET + Ridge Classifier            | `predictions_rocket.csv`  | ~78%     |
| 3 | **DWT + FFT + XGBoost**              | **`predictions_xgb.csv`** | **~95%** |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows

pip install numpy pandas matplotlib seaborn scipy PyWavelets scikit-learn xgboost numba sktime tsfresh
```

## Data

Datasets are downloaded automatically from Google Drive when running the notebook. Expected structure:

```
data/
├── train.csv    (100 × 1,461 — first column = label 0–9)
└── test.csv     (100 × 1,460 — features only)
```

## Usage

Run the Jupyter notebook:

```bash
jupyter notebook TSAC_Group_Project.ipynb
```

The three prediction CSV files will be generated in the project root.

---
*ENSIA TSAC — 2025/2026*
=======
# readme file <3
>>>>>>> 0340086f8387ae4475a2ae869be9850476a80d5c
