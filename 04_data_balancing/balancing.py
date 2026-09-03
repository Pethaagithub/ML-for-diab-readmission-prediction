"""
04 - Data Balancing
======================
Splits each task's dataset into stratified 80/20 train/test, then balances
the TRAINING set only (test sets are left untouched so evaluation reflects
real-world class distribution):

  - Necessity classification   -> TOMEK links undersampling
  - Criticality classification -> SMOTE oversampling

Outputs (written to 04_data_balancing/output/):
    - fig5_class_distribution.png  : Fig. 5, before/after class distribution for both tasks
    - balancing_summary.txt        : class counts before/after per task

Also writes the balanced training data (+ untouched test data) to
data/processed/ for use by the model building stage.
"""

import os
import sys
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_config, resolve_path, ensure_dir  # noqa: E402

sns.set_theme(style="whitegrid")


def load_features(config: dict, task: str) -> list:
    key = f"{task}_features"
    with open(resolve_path(config["paths"][key]), "r") as f:
        return json.load(f)


def prepare_task_data(df: pd.DataFrame, config: dict, task: str):
    """One-hot encode categoricals consistently, then select the Fisher-chosen features."""
    target_col = f"target_{task}"
    feature_cols_raw = load_features(config, task)

    non_feature_cols = {
        "readmitted", "target_necessity", "target_criticality",
        "race", "gender", "age", "diag_1", "diag_2", "diag_3",
        "admission_type_id", "discharge_disposition_id", "admission_source_id",
    }
    candidate_cols = [c for c in df.columns if c not in non_feature_cols]
    work_df = df[candidate_cols].copy()
    cat_cols = [c for c in work_df.columns if work_df[c].dtype == "object"
                or str(work_df[c].dtype) == "category"]
    if cat_cols:
        work_df = pd.get_dummies(work_df, columns=cat_cols, drop_first=False)

    available = [c for c in feature_cols_raw if c in work_df.columns]
    X = work_df[available]
    y = df[target_col]
    return X, y


def balance_task(X_train, y_train, method: str, random_state: int):
    if method == "tomek":
        sampler = TomekLinks()
    elif method == "smote":
        sampler = SMOTE(random_state=random_state)
    else:
        raise ValueError(f"Unknown balancing method: {method}")
    X_bal, y_bal = sampler.fit_resample(X_train, y_train)
    return X_bal, y_bal


def plot_class_distribution(counts_dict: dict, out_path: str):
    """Fig. 5 — side-by-side before/after class distribution for both tasks."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, task in zip(axes, ["necessity", "criticality"]):
        before = counts_dict[task]["before"]
        after = counts_dict[task]["after"]
        labels = ["Class 0", "Class 1"]
        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width / 2, [before.get(0, 0), before.get(1, 0)], width, label="Before")
        ax.bar(x + width / 2, [after.get(0, 0), after.get(1, 0)], width, label="After")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(f"{task.capitalize()} ({counts_dict[task]['method']})")
        ax.set_ylabel("Count")
        ax.legend()
    fig.suptitle("Fig. 5 — Class Distribution Before / After Balancing")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_balancing(config: dict):
    processed_path = resolve_path(config["paths"]["processed_data"])
    out_dir = ensure_dir(resolve_path(config["paths"]["balancing_output_dir"]))
    ensure_dir(os.path.dirname(resolve_path(config["paths"]["necessity_balanced_data"])))

    df = pd.read_csv(processed_path)
    random_state = config["random_state"]
    test_size = config["test_size"]

    summary_lines = []
    counts_for_plot = {}

    for task in ["necessity", "criticality"]:
        X, y = prepare_task_data(df, config, task)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

        before_counts = y_train.value_counts().to_dict()
        method = config["balancing"][task]
        X_bal, y_bal = balance_task(X_train, y_train, method, random_state)
        after_counts = y_bal.value_counts().to_dict()

        counts_for_plot[task] = {"before": before_counts, "after": after_counts, "method": method}

        summary_lines.append(f"[{task.upper()}] balancing method: {method}")
        summary_lines.append(f"  Train before: {before_counts}  (total={len(y_train)})")
        summary_lines.append(f"  Train after : {after_counts}  (total={len(y_bal)})")
        summary_lines.append(f"  Test (untouched): {y_test.value_counts().to_dict()} "
                              f"(total={len(y_test)})")
        summary_lines.append("")

        # Persist balanced train + untouched test sets for the modeling stage
        train_out = X_bal.copy()
        train_out[f"target_{task}"] = y_bal.values
        test_out = X_test.copy()
        test_out[f"target_{task}"] = y_test.values

        train_path = resolve_path(config["paths"][f"{task}_balanced_data"])
        test_path = train_path.replace("_balanced.csv", "_test.csv")
        train_out.to_csv(train_path, index=False)
        test_out.to_csv(test_path, index=False)

    plot_class_distribution(counts_for_plot, os.path.join(out_dir, "fig5_class_distribution.png"))

    with open(os.path.join(out_dir, "balancing_summary.txt"), "w") as f:
        f.write("\n".join(summary_lines))

    print("\n".join(summary_lines))
    print(f"Outputs saved to: {out_dir}")


if __name__ == "__main__":
    cfg = load_config()
    run_balancing(cfg)
