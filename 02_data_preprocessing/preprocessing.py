"""
02 - Data Preprocessing
=========================
Cleans the raw dataset and prepares it for feature selection:

  - Drops irrelevant / high-missingness columns (encounter_id, patient_nbr,
    weight, medical_specialty, max_glu_serum, A1Cresult, payer_code).
  - Nominally encodes 'race' and 'gender'; ordinally encodes 'age' buckets.
  - Groups admission_type_id / discharge_disposition_id / admission_source_id
    into broader categories.
  - Maps diag_1/diag_2/diag_3 ICD-9 codes into 17 general chapter categories.
  - Binarizes medication columns (No -> 0, {Up, Down, Steady} -> 1) and drops
    medications that are constant across the dataset (examide, citoglipton).
  - Builds the two binary target columns described in Table I:
        target_necessity   : 1 if readmitted in {'<30', '>30'}, else 0
        target_criticality : 1 if readmitted == '<30', else 0

Outputs (written to 02_data_preprocessing/output/):
    - preprocessing_log.txt          : shapes, dropped columns, encoding summary
    - table1_target_grouping.md      : markdown rendering of Table I
    - processed_data_sample.csv      : head() sample only (not the full dataset)
Also writes the full processed dataset to data/processed/processed_data.csv
(gitignored — regenerate by running this script).
"""

import os
import sys
import json

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_config, resolve_path, ensure_dir  # noqa: E402


# ---------------------------------------------------------------------------
# ICD-9 chapter mapping: (start, end, label). 'V' and 'E' codes handled separately.
# ---------------------------------------------------------------------------
ICD9_CHAPTERS = [
    (1, 139, "Infectious and Parasitic Diseases"),
    (140, 239, "Neoplasms"),
    (240, 279, "Endocrine, Nutritional, Metabolic, Immunity"),
    (280, 289, "Blood and Blood-forming Organs"),
    (290, 319, "Mental Disorders"),
    (320, 389, "Nervous System and Sense Organs"),
    (390, 459, "Circulatory System"),
    (460, 519, "Respiratory System"),
    (520, 579, "Digestive System"),
    (580, 629, "Genitourinary System"),
    (630, 679, "Pregnancy, Childbirth, Puerperium"),
    (680, 709, "Skin and Subcutaneous Tissue"),
    (710, 739, "Musculoskeletal and Connective Tissue"),
    (740, 759, "Congenital Anomalies"),
    (760, 779, "Perinatal Conditions"),
    (780, 799, "Symptoms, Signs, Ill-defined Conditions"),
    (800, 999, "Injury and Poisoning"),
]


def map_icd9_to_chapter(code):
    """Map a single ICD-9 diagnosis code to one of 17 general chapters."""
    if pd.isna(code):
        return "Missing"
    code = str(code)
    if code.startswith("V"):
        return "Supplemental (V-codes)"
    if code.startswith("E"):
        return "External Causes (E-codes)"
    try:
        numeric = float(code)
    except ValueError:
        return "Missing"
    for start, end, label in ICD9_CHAPTERS:
        if start <= numeric <= end:
            return label
    return "Other"


# ---------------------------------------------------------------------------
# Broader groupings for admission/discharge/source ids (standard groupings
# used widely with this UCI dataset).
# ---------------------------------------------------------------------------
ADMISSION_TYPE_MAP = {
    1: "Emergency", 2: "Urgent", 3: "Elective", 4: "Newborn",
    5: "Unknown", 6: "Unknown", 7: "Trauma Center", 8: "Unknown",
}

DISCHARGE_DISPOSITION_MAP = {
    1: "Discharged to Home",
    6: "Discharged to Home", 8: "Discharged to Home",
    3: "Discharged to Facility", 4: "Discharged to Facility",
    5: "Discharged to Facility", 22: "Discharged to Facility",
    23: "Discharged to Facility", 24: "Discharged to Facility",
    2: "Transferred", 9: "Transferred", 10: "Transferred",
    12: "Transferred", 15: "Transferred", 16: "Transferred", 17: "Transferred",
    11: "Expired", 19: "Expired", 20: "Expired", 21: "Expired",
    7: "Left AMA",
    13: "Hospice", 14: "Hospice",
    18: "Unknown", 25: "Unknown", 26: "Unknown",
}

ADMISSION_SOURCE_MAP = {
    1: "Referral", 2: "Referral", 3: "Referral",
    4: "Transfer", 5: "Transfer", 6: "Transfer", 10: "Transfer",
    18: "Transfer", 19: "Transfer", 22: "Transfer", 25: "Transfer", 26: "Transfer",
    7: "Emergency Room",
    8: "Court/Law Enforcement",
    9: "Unknown", 15: "Unknown", 17: "Unknown", 20: "Unknown", 21: "Unknown",
    11: "Newborn", 12: "Newborn", 13: "Newborn", 14: "Newborn",
    23: "Newborn", 24: "Newborn",
}

AGE_ORDINAL_MAP = {
    "[0-10)": 0, "[10-20)": 1, "[20-30)": 2, "[30-40)": 3, "[40-50)": 4,
    "[50-60)": 5, "[60-70)": 6, "[70-80)": 7, "[80-90)": 8, "[90-100)": 9,
}

MEDICATION_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]


def preprocess(df: pd.DataFrame, config: dict, log: list) -> pd.DataFrame:
    df = df.copy()
    log.append(f"Initial shape: {df.shape}")

    # --- Drop irrelevant / high-missingness columns ---
    cols_to_drop = [c for c in config["columns_to_drop"] if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    log.append(f"Dropped columns (irrelevant / >40% missing): {cols_to_drop}")

    # --- Drop medications that are constant ('No' for every row) ---
    const_meds = [c for c in config["constant_medication_columns"] if c in df.columns]
    df = df.drop(columns=const_meds)
    log.append(f"Dropped constant medication columns: {const_meds}")

    # --- Nominal encoding: race, gender ---
    for col in ["race", "gender"]:
        if col in df.columns:
            df[col] = df[col].astype("category")
            df[col + "_code"] = df[col].cat.codes
            log.append(f"Nominally encoded '{col}' -> '{col}_code' "
                       f"({len(df[col].cat.categories)} categories)")

    # --- Ordinal encoding: age ---
    if "age" in df.columns:
        df["age_ordinal"] = df["age"].map(AGE_ORDINAL_MAP)
        log.append("Ordinally encoded 'age' into 10 decade buckets (0-9)")

    # --- Group admission_type_id / discharge_disposition_id / admission_source_id ---
    if "admission_type_id" in df.columns:
        df["admission_type_group"] = df["admission_type_id"].map(ADMISSION_TYPE_MAP).fillna("Unknown")
        log.append("Grouped 'admission_type_id' into broader categories")
    if "discharge_disposition_id" in df.columns:
        df["discharge_disposition_group"] = df["discharge_disposition_id"].map(
            DISCHARGE_DISPOSITION_MAP).fillna("Unknown")
        log.append("Grouped 'discharge_disposition_id' into broader categories")
    if "admission_source_id" in df.columns:
        df["admission_source_group"] = df["admission_source_id"].map(
            ADMISSION_SOURCE_MAP).fillna("Unknown")
        log.append("Grouped 'admission_source_id' into broader categories")

    # --- ICD-9 mapping for diagnosis columns ---
    for diag_col in ["diag_1", "diag_2", "diag_3"]:
        if diag_col in df.columns:
            df[diag_col + "_category"] = df[diag_col].apply(map_icd9_to_chapter)
            log.append(f"Mapped '{diag_col}' to 17 ICD-9 chapter categories -> "
                       f"'{diag_col}_category'")

    # --- Binarize medication columns ---
    present_meds = [c for c in MEDICATION_COLUMNS if c in df.columns]
    for med in present_meds:
        df[med] = df[med].apply(lambda v: 0 if v == "No" else 1)
    log.append(f"Binarized {len(present_meds)} medication columns "
               f"(No -> 0, {{Up, Down, Steady}} -> 1)")

    # --- Handle remaining missing values / outliers (simple, documented strategy) ---
    n_missing_before = df.isna().sum().sum()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    for c in categorical_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mode().iloc[0])
    n_missing_after = df.isna().sum().sum()
    log.append(f"Missing values filled: {n_missing_before} -> {n_missing_after} "
               f"(numeric: median imputation, categorical: mode imputation)")

    # --- Build binary targets (Table I) ---
    if "readmitted" in df.columns:
        necessity_positive = set(config["targets"]["necessity"]["positive_classes"])
        criticality_positive = set(config["targets"]["criticality"]["positive_classes"])
        df["target_necessity"] = df["readmitted"].apply(lambda v: 1 if v in necessity_positive else 0)
        df["target_criticality"] = df["readmitted"].apply(lambda v: 1 if v in criticality_positive else 0)
        log.append("Built binary targets: 'target_necessity' (<30,>30 -> 1) and "
                   "'target_criticality' (<30 -> 1)")

    log.append(f"Final shape: {df.shape}")
    return df


def write_table1(out_dir: str):
    content = (
        "# Table I — Grouping of Target Classes\n\n"
        "| Objective    | Class 1 (positive)                | Class 0 (negative)      |\n"
        "|--------------|------------------------------------|--------------------------|\n"
        "| Necessity    | `<30` + `>30` (needs readmission)   | `No` (no readmission)   |\n"
        "| Criticality  | `<30` (critical readmission)        | `No` (no readmission)   |\n"
    )
    with open(os.path.join(out_dir, "table1_target_grouping.md"), "w") as f:
        f.write(content)


def run_preprocessing(config: dict):
    raw_path = resolve_path(config["paths"]["raw_data"])
    processed_path = resolve_path(config["paths"]["processed_data"])
    sample_path = resolve_path(config["paths"]["processed_sample"])
    out_dir = ensure_dir(resolve_path(config["paths"]["preprocessing_output_dir"]))
    ensure_dir(os.path.dirname(processed_path))

    df = pd.read_csv(raw_path, na_values="?")

    log = []
    processed_df = preprocess(df, config, log)

    ensure_dir(os.path.dirname(processed_path))
    processed_df.to_csv(processed_path, index=False)
    processed_df.head(10).to_csv(sample_path, index=False)

    with open(os.path.join(out_dir, "preprocessing_log.txt"), "w") as f:
        f.write("\n".join(log))

    write_table1(out_dir)

    print("\n".join(log))
    print(f"\nProcessed dataset saved to: {processed_path}")
    print(f"Sample (head) saved to: {sample_path}")
    print(f"Log saved to: {os.path.join(out_dir, 'preprocessing_log.txt')}")


if __name__ == "__main__":
    cfg = load_config()
    run_preprocessing(cfg)
