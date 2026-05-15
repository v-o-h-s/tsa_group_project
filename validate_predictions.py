"""
Validation Script: Compare prediction CSVs against validation.csv ground truth.

Usage:
    python validate_predictions.py predictions_rf.csv
    python validate_predictions.py predictions_rocket.csv predictions_xgb.csv
    python validate_predictions.py  (validates all prediction files found)
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Ground truth
VAL_PATH = os.path.join(os.path.dirname(__file__), 'data', 'validation.csv')


def load_ground_truth():
    df = pd.read_csv(VAL_PATH, header=None)
    return df.iloc[:, 0].values.astype(int)


def validate(pred_path, y_true):
    if not os.path.exists(pred_path):
        print(f"  [ERROR] File not found: {pred_path}")
        return

    df = pd.read_csv(pred_path)
    if 'Predicted' in df.columns:
        y_pred = df['Predicted'].values.astype(int)
    else:
        # Assume single column or second column
        y_pred = df.iloc[:, -1].values.astype(int)

    if len(y_pred) != len(y_true):
        print(f"  [WARNING] Length mismatch: predictions={len(y_pred)}, truth={len(y_true)}")
        n = min(len(y_pred), len(y_true))
        y_pred = y_pred[:n]
        y_true_trimmed = y_true[:n]
    else:
        y_true_trimmed = y_true

    acc = accuracy_score(y_true_trimmed, y_pred)
    print(f"\n{'='*60}")
    print(f"  File: {pred_path}")
    print(f"  Accuracy: {acc:.4f} ({int(acc * len(y_true_trimmed))}/{len(y_true_trimmed)})")
    print(f"{'='*60}")
    print(classification_report(y_true_trimmed, y_pred, digits=4, zero_division=0))

    cm = confusion_matrix(y_true_trimmed, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print()


def main():
    y_true = load_ground_truth()
    print(f"Ground truth loaded: {len(y_true)} samples")

    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        # Auto-find prediction CSVs in notebooks/ folder
        nb_dir = os.path.join(os.path.dirname(__file__), 'notebooks')
        files = [
            os.path.join(nb_dir, f)
            for f in os.listdir(nb_dir)
            if f.startswith('predictions_') and f.endswith('.csv')
        ]
        if not files:
            print("No prediction files found. Pass file paths as arguments.")
            return

    for f in sorted(files):
        validate(f, y_true)


if __name__ == '__main__':
    main()
