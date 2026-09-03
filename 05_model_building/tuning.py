"""
05b - Hyperparameter Optimization (Optuna)
=============================================
Tunes the two best-performing model families identified in baseline
comparisons — CatBoost and the Stacking ensemble — for both the
Necessity and Criticality tasks, using Optuna's tree-structured Parzen
estimator sampler to maximize F1-score via 5-fold cross-validation.

Outputs (written to 05_model_building/output/):
    - best_hyperparameters.json  : best params per task per model
Also overwrites the tuned models in:
    - 05_model_building/output/necessity_models/{catboost,stacking}_tuned.*
    - 05_model_building/output/criticality_models/{catboost,stacking}_tuned.*
"""

import os
import sys
import json
import joblib

import pandas as pd
import optuna
from optuna.samplers import TPESampler

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from catboost import CatBoostClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_config, resolve_path, ensure_dir  # noqa: E402
from models import load_task_data  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)


def catboost_objective(trial, X_train, y_train, random_state):
    params = {
        "iterations": trial.suggest_int("iterations", 100, 500),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "random_state": random_state,
        "verbose": False,
    }
    model = CatBoostClassifier(**params)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="f1", n_jobs=-1).mean()
    return score


def stacking_objective(trial, X_train, y_train, random_state):
    n_estimators_rf = trial.suggest_int("rf_n_estimators", 100, 400)
    max_depth_rf = trial.suggest_int("rf_max_depth", 3, 20)
    n_neighbors_knn = trial.suggest_int("knn_n_neighbors", 3, 15)
    lr_gb = trial.suggest_float("gb_learning_rate", 0.01, 0.3, log=True)
    iterations_cb = trial.suggest_int("cb_iterations", 100, 400)

    base_learners = [
        ("knn", KNeighborsClassifier(n_neighbors=n_neighbors_knn)),
        ("random_forest", RandomForestClassifier(
            n_estimators=n_estimators_rf, max_depth=max_depth_rf, random_state=random_state)),
        ("gradient_boosting", GradientBoostingClassifier(
            learning_rate=lr_gb, random_state=random_state)),
    ]
    final_estimator = CatBoostClassifier(iterations=iterations_cb, verbose=False,
                                         random_state=random_state)
    model = StackingClassifier(estimators=base_learners, final_estimator=final_estimator, cv=3)
    score = cross_val_score(model, X_train, y_train, cv=3, scoring="f1", n_jobs=-1).mean()
    return score


def tune_task(config: dict, task: str, best_params: dict):
    random_state = config["random_state"]
    n_trials = config["optuna"]["n_trials"]
    timeout = config["optuna"]["timeout_seconds"]

    X_train, y_train, X_test, y_test = load_task_data(config, task)
    task_out_dir = resolve_path(config["paths"][f"{task}_model_dir"])

    # --- Tune CatBoost ---
    study_cb = optuna.create_study(direction="maximize", sampler=TPESampler(seed=random_state))
    study_cb.optimize(
        lambda trial: catboost_objective(trial, X_train, y_train, random_state),
        n_trials=n_trials, timeout=timeout,
    )
    best_cb_params = study_cb.best_params
    best_cb_params.update({"random_state": random_state, "verbose": False})
    best_cb_model = CatBoostClassifier(**best_cb_params)
    best_cb_model.fit(X_train, y_train)
    best_cb_model.save_model(os.path.join(task_out_dir, "catboost_tuned.cbm"))

    # --- Tune Stacking ensemble ---
    study_stack = optuna.create_study(direction="maximize", sampler=TPESampler(seed=random_state))
    study_stack.optimize(
        lambda trial: stacking_objective(trial, X_train, y_train, random_state),
        n_trials=max(10, n_trials // 2), timeout=timeout,
    )
    sp = study_stack.best_params
    base_learners = [
        ("knn", KNeighborsClassifier(n_neighbors=sp["knn_n_neighbors"])),
        ("random_forest", RandomForestClassifier(
            n_estimators=sp["rf_n_estimators"], max_depth=sp["rf_max_depth"],
            random_state=random_state)),
        ("gradient_boosting", GradientBoostingClassifier(
            learning_rate=sp["gb_learning_rate"], random_state=random_state)),
    ]
    final_estimator = CatBoostClassifier(iterations=sp["cb_iterations"], verbose=False,
                                         random_state=random_state)
    best_stack_model = StackingClassifier(estimators=base_learners,
                                          final_estimator=final_estimator, cv=5)
    best_stack_model.fit(X_train, y_train)
    joblib.dump(best_stack_model, os.path.join(task_out_dir, "stacking_tuned.pkl"))

    best_params[task] = {
        "catboost": {"params": best_cb_params, "cv_f1": study_cb.best_value},
        "stacking": {"params": sp, "cv_f1": study_stack.best_value},
    }
    print(f"[{task}] Best CatBoost CV F1: {study_cb.best_value:.4f}")
    print(f"[{task}] Best Stacking CV F1: {study_stack.best_value:.4f}")


def run_tuning(config: dict):
    out_dir = ensure_dir(resolve_path(config["paths"]["model_output_dir"]))
    best_params = {}

    for task in ["necessity", "criticality"]:
        tune_task(config, task, best_params)

    with open(os.path.join(out_dir, "best_hyperparameters.json"), "w") as f:
        json.dump(best_params, f, indent=2)

    print(f"Best hyperparameters saved to: {os.path.join(out_dir, 'best_hyperparameters.json')}")


if __name__ == "__main__":
    cfg = load_config()
    run_tuning(cfg)
