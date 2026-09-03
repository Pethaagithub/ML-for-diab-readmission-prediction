# Diabetes Hospital Readmission Prediction

Predicting hospital readmission for diabetic patients using clinical data
from 130 US hospitals, framed as two complementary classification tasks:

- **Necessity Case Readmission** — will the patient be readmitted at all
  (`<30` or `>30` days)? Useful for hospital resource management.
- **Criticality Case Readmission** — will the patient be readmitted
  *within 30 days*? Useful for early identification of patients at risk
  of severe complications.

The pipeline covers preprocessing, Fisher's Score feature selection,
class balancing (TOMEK / SMOTE), model training with Optuna hyperparameter
optimization (best model: **CatBoost**), and LIME explainability.

**Headline results:** 0.80 recall, 0.78 F1-score, 80% accuracy.

---

## Project Structure

Each numbered folder corresponds to one stage of the pipeline and is
self-contained: it has its own script and an `output/` folder holding
everything that stage produces (logs, tables, figures).

```
diabetes-readmission-prediction/
├── config/config.yaml          # paths, hyperparameter spaces, thresholds
├── utils.py                     # shared config/path helpers
├── run_pipeline.py               # runs every stage end-to-end
│
├── 01_eda/                       # exploratory data analysis
├── 02_data_preprocessing/        # cleaning, encoding, ICD-9 mapping, target binarization
├── 03_feature_engineering/       # Fisher's Score feature selection
├── 04_data_balancing/            # TOMEK (Necessity) / SMOTE (Criticality)
├── 05_model_building/            # baseline models + Optuna-tuned CatBoost & Stacking
├── 06_evaluation/                # metrics tables, ROC-AUC, top-model comparison
├── 07_explainability/            # LIME explanations for CatBoost
│
└── fig2_architecture.png         # end-to-end architecture diagram
```

Each `output/` folder is the ground truth for that stage's results — the
tables and figures below are pulled from there.

---

## Dataset

Source: [UC Irvine ML Repository — Diabetes 130-US Hospitals for Years
1999–2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)

Clinical data gathered between 1999 and 2008 from 130 US institutions:
patient demographics, comorbidities, test results, prescription
information, and hospitalization details — 50+ features across
100,000+ inpatient encounters.

> Download `diabetic_data.csv` from the link above and place it at
> `data/raw/diabetic_data.csv` before running the pipeline. The raw
> file is not committed to this repo (see `.gitignore`).

<details>
<summary><strong>Sample output — df.head(), df.info(), df.describe(), df.columns</strong></summary>

```
[PASTE YOUR EDA OUTPUT HERE — this is the exact text you already generated
with print(df.head()), print(df.info()), print(df.describe()), and
print(df.columns.tolist()). The same text should also be saved as
01_eda/output/eda_summary.txt]
```

</details>

**Fig. 1 — Distribution of the raw target feature (`readmitted`)**

![Fig 1](01_eda/output/fig1_target_distribution.png)

The raw target is heavily imbalanced across its three original classes
(`No`, `>30`, `<30`), which motivates the binarization strategy below and
the balancing step later in the pipeline.

---

## Methodology

**Fig. 2 — Proposed architecture**

![Fig 2](fig2_architecture.png)

### 1. Data Preprocessing (`02_data_preprocessing/`)

- Dropped irrelevant / high-missingness columns: `encounter_id`,
  `patient_nbr`, `weight`, `medical_specialty`, `max_glu_serum`,
  `A1Cresult`, `payer_code`.
- Nominal encoding for `race`, `gender`; ordinal encoding for `age`
  buckets (`[0-10)` → 0 … `[90-100)` → 9).
- Grouped `admission_type_id`, `discharge_disposition_id`, and
  `admission_source_id` into broader categories.
- Mapped `diag_1`/`diag_2`/`diag_3` ICD-9 codes into 17 general chapter
  categories.
- Binarized medication columns (`No` → 0, `{Up, Down, Steady}` → 1);
  dropped constant-valued medications (`examide`, `citoglipton`).
- Simplified the 3-class `readmitted` target into two binary targets:

**Table I — Grouping of Target Classes**

| Objective    | Class 1 (positive)                | Class 0 (negative)      |
|--------------|------------------------------------|--------------------------|
| Necessity    | `<30` + `>30` (needs readmission)   | `No` (no readmission)   |
| Criticality  | `<30` (critical readmission)        | `No` (no readmission)   |

Full preprocessing log: [`02_data_preprocessing/output/preprocessing_log.txt`](02_data_preprocessing/output/preprocessing_log.txt)

### 2. Feature Selection (`03_feature_engineering/`)

Fisher's Score — the ratio of inter-class variance to intra-class
variance — was computed per feature per task. Features scoring above
**3.0** were retained.

| Task | Figure |
|---|---|
| Necessity | ![Fig 3](03_feature_engineering/output/fig3_fisher_necessity.png) |
| Criticality | ![Fig 4](03_feature_engineering/output/fig4_fisher_criticality.png) |

**Table II — Fisher's Scores**

```
[PASTE YOUR TABLE II VALUES HERE, or reference the generated file:
03_feature_engineering/output/table2_fisher_scores.csv]
```

### 3. Data Balancing (`04_data_balancing/`)

80/20 stratified train/test split, balancing applied to the training set only:

- **Necessity** → TOMEK links (removes borderline/overlapping instances)
- **Criticality** → SMOTE (synthetic oversampling of the minority class)

**Fig. 5 — Class distribution before/after balancing**

![Fig 5](04_data_balancing/output/fig5_class_distribution.png)

### 4. Model Building (`05_model_building/`)

Baseline models: KNN, Logistic Regression, Naïve Bayes, Decision Tree,
Random Forest, CatBoost, and a Stacking ensemble (KNN + Random Forest +
Gradient Boosting as base learners, CatBoost as meta-learner).

CatBoost and the Stacking ensemble were further tuned with **Optuna**
(TPE sampler, 5-fold CV, F1-score objective). Best hyperparameters:
[`05_model_building/output/best_hyperparameters.json`](05_model_building/output/best_hyperparameters.json)

### 5. Evaluation & Explainability

Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC.

---

## Results

**Table III — Evaluation Metrics: Necessity Classification**

```
[PASTE YOUR TABLE III VALUES HERE, or reference the generated file:
06_evaluation/output/table3_necessity_metrics.csv]
```

**Table IV — Evaluation Metrics: Criticality Classification**

```
[PASTE YOUR TABLE IV VALUES HERE, or reference the generated file:
06_evaluation/output/table4_criticality_metrics.csv]
```

| Necessity ROC-AUC | Criticality ROC-AUC |
|---|---|
| ![Fig 6](06_evaluation/output/fig6_roc_necessity.png) | ![Fig 7](06_evaluation/output/fig7_roc_criticality.png) |

**Fig. 8 — Comparison of top 3 performing models (accuracy, precision, recall, AUC)**

![Fig 8](06_evaluation/output/fig8_top3_comparison.png)

The best-performing model overall was **CatBoost** (with Optuna tuning),
achieving a recall of 0.80, an F1-score of 0.78, and 80% accuracy.

### Explainability

**Fig. 9 — LIME explanation for the best performing model (CatBoost)**

![Fig 9](07_explainability/output/fig9_lime_catboost.png)

LIME (Local Interpretable Model-agnostic Explanations) was used to
interpret individual predictions and surface the key clinical risk
factors driving readmission risk, supporting transparency and trust in
the model's decisions for healthcare stakeholders.

---

## Running the Pipeline

```bash
# 1. Set up environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Download the dataset and place it at data/raw/diabetic_data.csv
#    https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

# 3a. Run the full pipeline
python run_pipeline.py

# 3b. Or run stages individually
python 01_eda/eda.py
python 02_data_preprocessing/preprocessing.py
python 03_feature_engineering/feature_selection.py
python 04_data_balancing/balancing.py
python 05_model_building/models.py
python 05_model_building/tuning.py       # optional, slower (Optuna search)
python 06_evaluation/evaluate.py
python 07_explainability/explainability.py

# 3c. Or skip the slow tuning stage / run a subset
python run_pipeline.py --skip-tuning
python run_pipeline.py --only eda,preprocessing,feature_selection
```

---

## Tech Stack

Python · pandas · scikit-learn · imbalanced-learn (SMOTE/TOMEK) ·
CatBoost · Optuna · LIME · matplotlib / seaborn

---

## Citation

If you use this work, please cite the associated paper:

```
[Add your paper citation / BibTeX here once available]
```

## License

[Add a license, e.g. MIT, if you intend this repo to be reused]
