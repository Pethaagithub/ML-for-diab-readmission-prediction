"""
01 - Exploratory Data Analysis
================================
Loads the raw diabetic_data.csv, prints a first look at the dataset
(head, info, describe, column list) and produces Fig. 1: the
distribution of the raw 3-class target feature ('readmitted').

Outputs (written to 01_eda/output/):
    - eda_summary.txt              : captured console output (head/info/describe/columns)
    - fig1_target_distribution.png : bar chart of the raw 'readmitted' class distribution
"""

import os
import sys
import io
import contextlib

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_config, resolve_path, ensure_dir  # noqa: E402

sns.set_theme(style="whitegrid")


def run_eda(config: dict):
    raw_path = resolve_path(config["paths"]["raw_data"])
    out_dir = ensure_dir(resolve_path(config["paths"]["eda_output_dir"]))

    df = pd.read_csv(raw_path, na_values="?")

    # ---- Capture head / info / describe / columns exactly as printed ----
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        print("First few rows of the dataset:")
        print(df.head())

        print("\nDataset information:")
        print(df.info())

        print("\nBasic statistics:")
        print(df.describe())

        print("\nColumn names:")
        print(df.columns.tolist())

    summary_text = buffer.getvalue()
    print(summary_text)  # also print to console when script is run directly

    with open(os.path.join(out_dir, "eda_summary.txt"), "w") as f:
        f.write(summary_text)

    # ---- Fig. 1: Distribution of the raw target feature ----
    if "readmitted" in df.columns:
        plt.figure(figsize=(7, 5))
        order = df["readmitted"].value_counts().index
        ax = sns.countplot(x="readmitted", data=df, order=order, palette="viridis")
        ax.set_title("Fig. 1 — Distribution of Target Feature (readmitted)")
        ax.set_xlabel("Readmission Class")
        ax.set_ylabel("Count")
        for p in ax.patches:
            ax.annotate(f"{int(p.get_height())}",
                        (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        fig_path = os.path.join(out_dir, "fig1_target_distribution.png")
        plt.savefig(fig_path, dpi=200)
        plt.close()
        print(f"Saved: {fig_path}")
    else:
        print("Warning: 'readmitted' column not found — skipping Fig. 1.")

    print(f"EDA summary saved to: {os.path.join(out_dir, 'eda_summary.txt')}")


if __name__ == "__main__":
    cfg = load_config()
    run_eda(cfg)
