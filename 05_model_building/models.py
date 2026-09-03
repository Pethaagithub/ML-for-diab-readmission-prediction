"""
05 - Model Building
======================
Trains the baseline classifiers described in the paper for both the
Necessity and Criticality tasks:

    KNN, Logistic Regression, Naive Bayes, Decision Tree,
    Random Forest, CatBoost, and a Stacking ensemble
    (base learners: KNN, Random Forest, Gradient Boosting, CatBoost).

CatBoost and the Stacking ensemble are additionally tuned with Optuna
(see tuning.py) and re-saved with their optimized hyperparameters.

Outputs (written to 05_model_building/output/):
    - necessity_models/*.pkl / *.cbm     : trained model artifacts (Necessity)
    - criticality_models/*.pkl / *.cbm   : trained model artifacts (Criticality)
    - training_summary.txt               : which models were trained, basic timing
"""

import os
import sys
import time
import joblib

import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from catboost import CatBoostClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_config, resolve_path, ensure_dir  # noqa: E402


def build_baseline_models(random_state: int) -> dict:
    """Instantiate the baseline models with sensible default hyperparameters."""
    return {
        "knn": KNeighborsClassifier(n_neighbors=5),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "naive_bayes": GaussianNB(),
        "decision_tree": DecisionTreeClassifier(random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=random_state),
        "catboost": CatBoostClassifier(
            iterations=300, verbose=False, random_state=random_state
        ),
    }


def build_stacking_model(random_state: int) -> StackingClassifier:
    """Stacking ensemble: KNN, Random Forest, Gradient Boosting as base learners,
    CatBoost as the final (meta) estimator."""
    base_learners = [
        ("knn", KNeighborsClassifier(n_neighbors=5)),
        ("random_forest", RandomForestClassifier(n_estimators=200, random_state=random_state)),
        ("gradient_boosting", GradientBoostingClassifier(random_state=random_state)),
    ]
    final_estimator = CatBoostClassifier(iterations=200, verbose=False, random_state=random_state)
    return StackingClassifier(estimators=base_learners, final_estimator=final_estimator, cv=5)


def load_task_data(config: dict, task: str):
    train_path = resolve_path(config["paths"][f"{task}_balanced_data"])
    test_path = train_path.replace("_balanced.csv", "_test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    target_col = f"target_{task}"
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    return X_train, y_train, X_test, y_test


def train_all_models_for_task(config: dict, task: str, out_dir: str, summary: list):
    random_state = config["random_state"]
    X_train, y_train, X_test, y_test = load_task_data(config, task)

    models = build_baseline_models(random_state)
    models["stacking"] = build_stacking_model(random_state)

    for name in config["models"]:
        if name not in models:
            continue
        model = models[name]
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start

        if name == "catboost":
            model.save_model(os.path.join(out_dir, f"{name}.cbm"))
        else:
            joblib.dump(model, os.path.join(out_dir, f"{name}.pkl"))

        summary.append(f"[{task}] Trained '{name}' in {elapsed:.2f}s "
                       f"-> saved to {out_dir}")
        print(summary[-1])


def run_model_building(config: dict):
    base_out_dir = ensure_dir(resolve_path(config["paths"]["model_output_dir"]))
    summary = []

    for task in ["necessity", "criticality"]:
        task_out_dir = ensure_dir(resolve_path(config["paths"][f"{task}_model_dir"]))
        train_all_models_for_task(config, task, task_out_dir, summary)

    with open(os.path.join(base_out_dir, "training_summary.txt"), "w") as f:
        f.write("\n".join(summary))

    print(f"Training summary saved to: {os.path.join(base_out_dir, 'training_summary.txt')}")


if __name__ == "__main__":
    cfg = load_config()
    run_model_building(cfg)
