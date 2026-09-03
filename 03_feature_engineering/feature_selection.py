"""
03 - Feature Engineering (Fisher's Score)
============================================
Computes Fisher's Score for every numeric/encoded feature against each of
the two binary targets (Necessity, Criticality) and selects the features
with a score greater than the configured threshold (default: 3.0).

Fisher's Score for a feature f, given two classes, is:

    F(f) = (mean_1 - mean_0)^2 / (var_1 + var_0)

i.e. the ratio of inter-class variance (how far apart the class means are)
to intra-class variance (how spread out each class is).

Outputs (written to 03_feature_engineering/output/):
    - fig3_fisher_necessity.png        : Fig. 3, Fisher's Score bar chart (Necessity)
    - fig4_fisher_criticality.png      : Fig. 4, Fisher's Score bar chart (Criticality)
    - table2_fisher_scores.csv         : Table II, raw Fisher's Score values
    - necessity_selected_features.json : list of selected feature names (Necessity)
    - criticality_selected_features.json : list of selected feature names (Criticality)
"""

import os
import sys
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_config, resolve_path, ensure_dir  # noqa: E402

sns.set_theme(style="whitegrid")

NON_FEATURE_COLUMNS = {
    "readmitted", "target_necessity", "target_criticality",
    "race", "gender", "age",  # kept as raw text; encoded versions are used instead
    "diag_1", "diag_2", "diag_3",  # raw ICD-9 codes; category versions are used instead
    "admission_type_id", "discharge_disposition_id", "admission_source_id",  # grouped versions used
}


def fisher_score(feature: pd.Series, target: pd.Series) -> float:
    """Fisher's Score for a single feature against a binary target."""
    class0 = feature[target == 0]
    class1 = feature[target == 1]
    mean_diff_sq = (class1.mean() - class0.mean()) ** 2
    denom = class1.var() + class0.var()
    if denom == 0 or np.isnan(denom):
        return 0.0
    return float(mean_diff_sq / denom)


def one_hot_categoricals(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """One-hot encode any remaining object/category columns among feature_cols."""
    cat_cols = [c for c in feature_cols if df[c].dtype == "object" or str(df[c].dtype) == "category"]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    return df


def compute_scores(df: pd.DataFrame, target_col: str) -> pd.Series:
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS
                     and c != target_col and c not in ("target_necessity", "target_criticality")]
    work_df = df[feature_cols + [target_col]].copy()
    work_df = one_hot_categoricals(work_df, feature_cols)
    numeric_feature_cols = [c for c in work_df.columns if c != target_col]

    scores = {}
    for col in numeric_feature_cols:
        scores[col] = fisher_score(work_df[col], work_df[target_col])
    return pd.Series(scores).sort_values(ascending=False)


def plot_scores(scores: pd.Series, title: str, out_path: str, top_n: int = 25):
    top_scores = scores.head(top_n)
    plt.figure(figsize=(9, max(6, 0.3 * len(top_scores))))
    sns.barplot(x=top_scores.values, y=top_scores.index, palette="crest")
    plt.axvline(x=3.0, color="red", linestyle="--", label="Threshold (Fisher's Score = 3)")
    plt.title(title)
    plt.xlabel("Fisher's Score")
    plt.ylabel("Feature")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_feature_selection(config: dict):
    processed_path = resolve_path(config["paths"]["processed_data"])
    out_dir = ensure_dir(resolve_path(config["paths"]["feature_engineering_output_dir"]))
    threshold = config["fisher_score_threshold"]

    df = pd.read_csv(processed_path)

    necessity_scores = compute_scores(df, "target_necessity")
    criticality_scores = compute_scores(df, "target_criticality")

    # Fig. 3 / Fig. 4
    plot_scores(necessity_scores, "Fig. 3 — Fisher's Score Feature Hierarchy (Necessity)",
                os.path.join(out_dir, "fig3_fisher_necessity.png"))
    plot_scores(criticality_scores, "Fig. 4 — Fisher's Score Feature Hierarchy (Criticality)",
                os.path.join(out_dir, "fig4_fisher_criticality.png"))

    # Table II
    table2 = pd.DataFrame({
        "feature": sorted(set(necessity_scores.index) | set(criticality_scores.index))
    })
    table2["fisher_score_necessity"] = table2["feature"].map(necessity_scores).fillna(0.0)
    table2["fisher_score_criticality"] = table2["feature"].map(criticality_scores).fillna(0.0)
    table2 = table2.sort_values("fisher_score_necessity", ascending=False)
    table2.to_csv(os.path.join(out_dir, "table2_fisher_scores.csv"), index=False)

    # Selected features (score > threshold)
    necessity_selected = necessity_scores[necessity_scores > threshold].index.tolist()
    criticality_selected = criticality_scores[criticality_scores > threshold].index.tolist()

    with open(resolve_path(config["paths"]["necessity_features"]), "w") as f:
        json.dump(necessity_selected, f, indent=2)
    with open(resolve_path(config["paths"]["criticality_features"]), "w") as f:
        json.dump(criticality_selected, f, indent=2)

    print(f"Necessity: {len(necessity_selected)} features selected (Fisher's Score > {threshold})")
    print(f"Criticality: {len(criticality_selected)} features selected (Fisher's Score > {threshold})")
    print(f"Outputs saved to: {out_dir}")


if __name__ == "__main__":
    cfg = load_config()
    run_feature_selection(cfg)
