"""
End-to-end pipeline orchestrator.

Runs every stage in order:
    01 EDA -> 02 Preprocessing -> 03 Feature Selection -> 04 Balancing
    -> 05 Model Building (+ Optuna tuning) -> 06 Evaluation -> 07 Explainability

Usage:
    python run_pipeline.py                  # run every stage
    python run_pipeline.py --skip-tuning     # skip the (slow) Optuna tuning step
    python run_pipeline.py --only eda,preprocessing   # run a subset of stages
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "05_model_building"))

from utils import load_config  # noqa: E402


STAGES = [
    "eda", "preprocessing", "feature_selection", "balancing",
    "model_building", "tuning", "evaluation", "explainability",
]


def main():
    parser = argparse.ArgumentParser(description="Run the diabetes readmission pipeline.")
    parser.add_argument("--skip-tuning", action="store_true",
                        help="Skip the Optuna hyperparameter tuning stage (slow).")
    parser.add_argument("--only", type=str, default=None,
                        help=f"Comma-separated subset of stages to run: {', '.join(STAGES)}")
    args = parser.parse_args()

    stages_to_run = STAGES if args.only is None else [s.strip() for s in args.only.split(",")]
    if args.skip_tuning and "tuning" in stages_to_run:
        stages_to_run.remove("tuning")

    config = load_config()

    if "eda" in stages_to_run:
        from importlib import import_module
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "01_eda"))
        import eda
        print("\n=== Stage 01: EDA ===")
        eda.run_eda(config)

    if "preprocessing" in stages_to_run:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "02_data_preprocessing"))
        import preprocessing
        print("\n=== Stage 02: Data Preprocessing ===")
        preprocessing.run_preprocessing(config)

    if "feature_selection" in stages_to_run:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "03_feature_engineering"))
        import feature_selection
        print("\n=== Stage 03: Feature Engineering ===")
        feature_selection.run_feature_selection(config)

    if "balancing" in stages_to_run:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "04_data_balancing"))
        import balancing
        print("\n=== Stage 04: Data Balancing ===")
        balancing.run_balancing(config)

    if "model_building" in stages_to_run:
        import models
        print("\n=== Stage 05a: Model Building ===")
        models.run_model_building(config)

    if "tuning" in stages_to_run:
        import tuning
        print("\n=== Stage 05b: Hyperparameter Tuning (Optuna) ===")
        tuning.run_tuning(config)

    if "evaluation" in stages_to_run:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "06_evaluation"))
        import evaluate
        print("\n=== Stage 06: Evaluation ===")
        evaluate.run_evaluation(config)

    if "explainability" in stages_to_run:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "07_explainability"))
        import explainability
        print("\n=== Stage 07: Explainability (LIME) ===")
        explainability.run_explainability(config)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
