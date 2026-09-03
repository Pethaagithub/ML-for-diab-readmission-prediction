"""
06 - Evaluation
==================
Loads every trained model (baseline + tuned) for both tasks, scores them
on the held-out test set, and produces:

    - Table III / Table IV : Precision, Recall, F1-score, Accuracy, AUC-ROC
    - Fig. 6 / Fig. 7       : ROC-AUC curves per task
    - Fig. 8                : comparison of the top-3 performing models
                               (accuracy, precision, recall, AUC)

Outputs (written to 06_evaluation/output/):
    - table3_necessity_metrics.csv
    - table4_criticality_metrics.csv
    - fig6_roc_necessity.png
    - fig7_roc_criticality.png
    - fig8_top3_comparison.png
"""

import os
import sys
import glob
import joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve
)
from catboost import CatBoostClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_config, resolve_path, ensure_dir  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "05_model_building"))
from models import load_task_data  # noqa: E402

sns.set_theme(style="whitegrid")


def load_all_models(model_dir: str) -> dict:
    """Load every .pkl and .cbm model found in a task's model directory."""
    models = {}
    for path in glob.glob(os.path.join(model_dir, "*.pkl")):
        name = os.path.splitext(os.path.basename(path))[0]
        models[name] = joblib.load(path)
    for path in glob.glob(os.path.join(model_dir, "*.cbm")):
        name = os.path.splitext(os.path.basename(path))[0]
        cb = CatBoostClassifier()
        cb.load_model(path)
        models[name] = cb
    return models


def evaluate_models(models: dict, X_test, y_test) -> pd.DataFrame:
    rows = []
    roc_data = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred = np.round(np.asarray(y_pred).astype(float)).astype(int).ravel()

        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_data[name] = (fpr, tpr, auc)
        except Exception:
            auc = np.nan

        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": auc,
        })
    results_df = pd.DataFrame(rows).sort_values("f1_score", ascending=False).reset_index(drop=True)
    return results_df, roc_data


def plot_roc(roc_data: dict, title: str, out_path: str):
    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_top3_comparison(necessity_df: pd.DataFrame, criticality_df: pd.DataFrame, out_path: str):
    """Fig. 8 — compare top 3 models (by F1) across accuracy/precision/recall/AUC,
    combining both tasks' best model set."""
    combined = pd.concat([
        necessity_df.head(3).assign(task="Necessity"),
        criticality_df.head(3).assign(task="Criticality"),
    ])
    metrics = ["accuracy", "precision", "recall", "roc_auc"]
    combined["label"] = combined["task"] + " — " + combined["model"]
    melted = combined.melt(id_vars="label", value_vars=metrics,
                            var_name="metric", value_name="score")

    plt.figure(figsize=(11, 6))
    sns.barplot(data=melted, x="metric", y="score", hue="label")
    plt.title("Fig. 8 — Comparison of Top 3 Performing Models")
    plt.ylabel("Score")
    plt.xlabel("Metric")
    plt.ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_evaluation(config: dict):
    out_dir = ensure_dir(resolve_path(config["paths"]["evaluation_output_dir"]))

    results = {}
    for task, table_name, fig_name in [
        ("necessity", "table3_necessity_metrics.csv", "fig6_roc_necessity.png"),
        ("criticality", "table4_criticality_metrics.csv", "fig7_roc_criticality.png"),
    ]:
        model_dir = resolve_path(config["paths"][f"{task}_model_dir"])
        _, _, X_test, y_test = load_task_data(config, task)
        models = load_all_models(model_dir)

        if not models:
            print(f"No trained models found in {model_dir} — run 05_model_building first.")
            continue

        results_df, roc_data = evaluate_models(models, X_test, y_test)
        results_df.to_csv(os.path.join(out_dir, table_name), index=False)
        plot_roc(roc_data, f"ROC-AUC — {task.capitalize()} Classification",
                 os.path.join(out_dir, fig_name))
        results[task] = results_df

        print(f"\n[{task.upper()}] Evaluation results:")
        print(results_df.to_string(index=False))

    if "necessity" in results and "criticality" in results:
        plot_top3_comparison(results["necessity"], results["criticality"],
                              os.path.join(out_dir, "fig8_top3_comparison.png"))

    print(f"\nAll evaluation outputs saved to: {out_dir}")


if __name__ == "__main__":
    cfg = load_config()
    run_evaluation(cfg)
