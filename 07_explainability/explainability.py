"""
07 - Explainability (LIME)
=============================
Applies LIME (Local Interpretable Model-agnostic Explanations) to the
best-performing model — CatBoost — to explain individual predictions and
surface the key risk factors driving readmission risk, for both the
Necessity and Criticality tasks.

Outputs (written to 07_explainability/output/):
    - fig9_lime_catboost.png             : Fig. 9, LIME explanation (Necessity, CatBoost)
    - fig9_lime_catboost_criticality.png : equivalent explanation for Criticality
"""

import os
import sys
import glob

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from lime.lime_tabular import LimeTabularExplainer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_config, resolve_path, ensure_dir  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "05_model_building"))
from models import load_task_data  # noqa: E402


def find_best_catboost_model(model_dir: str) -> str:
    """Prefer the Optuna-tuned CatBoost model if present, else fall back to the baseline."""
    tuned = os.path.join(model_dir, "catboost_tuned.cbm")
    baseline = os.path.join(model_dir, "catboost.cbm")
    if os.path.exists(tuned):
        return tuned
    if os.path.exists(baseline):
        return baseline
    matches = glob.glob(os.path.join(model_dir, "catboost*.cbm"))
    return matches[0] if matches else None


def explain_task(config: dict, task: str, out_dir: str, sample_index: int = 0):
    model_dir = resolve_path(config["paths"][f"{task}_model_dir"])
    model_path = find_best_catboost_model(model_dir)
    if model_path is None:
        print(f"No CatBoost model found for '{task}' in {model_dir} — "
              f"run 05_model_building first.")
        return

    model = CatBoostClassifier()
    model.load_model(model_path)

    X_train, y_train, X_test, y_test = load_task_data(config, task)

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=["No Readmission", "Readmission"],
        mode="classification",
        discretize_continuous=True,
        random_state=config["random_state"],
    )

    instance = X_test.iloc[sample_index].values
    explanation = explainer.explain_instance(
        instance, model.predict_proba, num_features=10
    )

    fig_name = "fig9_lime_catboost.png" if task == "necessity" else "fig9_lime_catboost_criticality.png"
    fig = explanation.as_pyplot_figure()
    fig.suptitle(f"Fig. 9 — LIME Explanation ({task.capitalize()}, CatBoost)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fig_name), dpi=200)

    explanation.save_to_file(os.path.join(out_dir, fig_name.replace(".png", ".html")))
    print(f"[{task}] LIME explanation saved to: {os.path.join(out_dir, fig_name)}")


def run_explainability(config: dict):
    out_dir = ensure_dir(resolve_path(config["paths"]["explainability_output_dir"]))
    for task in ["necessity", "criticality"]:
        explain_task(config, task, out_dir)
    print(f"All explainability outputs saved to: {out_dir}")


if __name__ == "__main__":
    cfg = load_config()
    run_explainability(cfg)
