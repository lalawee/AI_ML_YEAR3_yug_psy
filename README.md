# Machine Learning Pipeline — Titanic Survival Prediction

A model-agnostic machine learning pipeline for predicting Titanic passenger survival. Supports multiple scikit-learn classifiers, config-driven hyperparameter search (grid or random), and automated report generation in Excel and PDF formats.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Pipeline](#running-the-pipeline)
- [Outputs](#outputs)
- [Adding a New Model](#adding-a-new-model)
- [Requirements](#requirements)
- [Contact](#contact)

---

## Overview

The pipeline predicts whether a Titanic passenger survived based on features such as age, sex, ticket class, fare, and embarkation point. It is designed to be fully config-driven — no source code changes are needed to switch models, tune hyperparameters, or adjust search strategy.

Key capabilities:

- **Unified preprocessing** via `preprocessing.py` — handles missing values, categorical encoding, and feature selection in one place
- **Model registry** (`model_config/model_registry.json`) — add any scikit-learn classifier without touching Python code
- **Hyperparameter search** — `GridSearchCV` or `RandomizedSearchCV` driven entirely by the experiment config
- **Structured outputs** — predictions (Excel), metrics (JSON), classification reports (TXT), and PDF summaries, all routed to versioned subfolders
- **Labeled and unlabeled test CSV support** — prediction-only mode when no `Survived` column is present

---

## Project Structure

```
.
├── pipeline.bat                          # Windows pipeline entry point
├── pipeline.sh                           # Unix/macOS pipeline entry point
├── train_model.py                        # Training script (search + fit + metrics)
├── predict_model.py                      # Prediction script (labeled or unlabeled CSVs)
├── generate_pdf.py                       # PDF report generator (reads metrics JSON)
├── preprocessing.py                      # Shared preprocessing module
│
├── raw_csv/
│   ├── MS_1_Scenario_train.csv           # Training dataset
│   ├── MS_1_Scenario_test.csv            # Validation dataset (labeled)
│   └── MS_1_Scenario_test_answer.csv     # Answer key (optional reference)
│
├── test CSV/                             # Drop unlabeled CSVs here for batch testing
│
├── model_config/
│   ├── model_registry.json               # Global model registry (class paths, defaults, seed)
│   ├── random_forest/
│   │   ├── config_grid_search_random_forest_exp_000001.json
│   │   └── config_random_search_random_forest_exp_000001.json
│   └── logistic_regression/
│       ├── config_grid_search_logistic_regression_exp_000001.json
│       └── config_random_search_logistic_regression_exp_000001.json
│
├── models/                               # Saved .pkl files (created at runtime)
│   └── {model_name}/
│       └── {strategy}_exp_{exp_number}_{model_name}.pkl
│
├── metrics/                              # Metrics JSON files (created at runtime)
│   └── {model_name}/
│       ├── metrics_{strategy}_{model_name}_exp_{exp}_train.json
│       └── metrics_{strategy}_{model_name}_exp_{exp}_test.json
│
├── output/                               # All report outputs (created at runtime)
│   ├── predicted_data/
│   │   └── {dataset_name}_prediction_output.xlsx
│   └── {model_name}/
│       ├── {strategy}_exp_{exp_number}_train/
│       │   ├── {dataset_name}_classification_report.txt
│       │   └── {dataset_name}_report.pdf
│       └── {strategy}_exp_{exp_number}_test/
│           ├── {dataset_name}_classification_report.txt
│           └── {dataset_name}_report.pdf
│
├── requirements.txt
├── CLAUDE.md                             # Architecture decisions and refactor log
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd AI_MachineLearning_Project_SIT
```

### 2. Create and activate a conda environment

```bash
conda create -n sit_ml python=3.9
conda activate sit_ml
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Important:** `numpy` must be installed before `pandas` to avoid binary incompatibility errors. The `requirements.txt` lists `numpy` first and pins both to compatible versions — do not change these pins without testing.

---

## Configuration

### Model Registry — `model_config/model_registry.json`

Defines which scikit-learn models the pipeline knows about. Each entry specifies the fully-qualified class path, the seed parameter name, and default hyperparameters.

```json
{
  "default_seed": 42,
  "models": {
    "random_forest": {
      "class": "sklearn.ensemble.RandomForestClassifier",
      "seed_param": "random_state",
      "default_params": {
        "n_estimators": 100,
        "max_depth": null
      }
    }
  }
}
```

### Experiment Config — `model_config/{model_name}/config_{strategy}_{model_name}_exp_{MMPPBB}.json`

Each experiment is a separate JSON file. It references the model, sets fixed (non-searched) parameters, defines the search space, and controls the search strategy.

```json
{
  "model_type": "random_forest",
  "search_strategy": "grid_search",
  "scoring": "accuracy",
  "cv": 5,
  "n_iter": 10,
  "fixed_params": {
    "min_samples_leaf": 4
  },
  "search_space": {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 8, null]
  }
}
```

**Parameter resolution order:** registry `default_params` → experiment `fixed_params` → best params from search.

---

## Running the Pipeline

### Windows

```bat
pipeline.bat
```

### Unix / macOS

```bash
bash pipeline.sh
```

### Startup prompts

The pipeline will ask for three inputs before launching:

```
Model name (e.g. random_forest): random_forest
Search strategy (e.g. random_search, grid_search): grid_search
Experiment number (e.g. 000001): 000001
```

It then resolves all file paths and displays a summary before showing the menu.

### CLI Menu

```
Pipeline CLI Menu (model: random_forest | strategy: grid_search | exp: 000001):
1. Train the Model
2. Test on Available Validation Dataset
3. Test on New/Random Dataset
4. Exit
```

**Option 1 — Train the Model**
Runs hyperparameter search on `raw_csv/MS_1_Scenario_train.csv`, fits the best estimator, saves the model `.pkl`, writes a `_train.json` metrics file, and generates a training PDF report.

**Option 2 — Test on Available Validation Dataset**
Runs the trained model against `raw_csv/MS_1_Scenario_test.csv` (labeled), writes a `_test.json` metrics file, and generates a test PDF report and color-coded Excel file.

**Option 3 — Test on New/Random Dataset**
Iterates over every CSV in `test CSV/`. Works with both labeled CSVs (full metrics) and unlabeled CSVs (prediction-only output — no metrics, no crash).

---

## Outputs

| Output | Location |
|---|---|
| Trained model | `models/{model_name}/{strategy}_exp_{exp}_{model_name}.pkl` |
| Metrics (train) | `metrics/{model_name}/metrics_{strategy}_{model_name}_exp_{exp}_train.json` |
| Metrics (test) | `metrics/{model_name}/metrics_{strategy}_{model_name}_exp_{exp}_test.json` |
| Prediction Excel | `output/predicted_data/{dataset_name}_prediction_output.xlsx` |
| Classification report | `output/{model_name}/{strategy}_exp_{exp}_{split}/{dataset_name}_classification_report.txt` |
| PDF report | `output/{model_name}/{strategy}_exp_{exp}_{split}/{dataset_name}_report.pdf` |

All output directories are created automatically by the pipeline — no manual setup needed.

---

## Adding a New Model

No Python code changes are required. To add, for example, a Gradient Boosting classifier:

**Step 1** — Add an entry to `model_config/model_registry.json`:

```json
"gradient_boosting": {
  "class": "sklearn.ensemble.GradientBoostingClassifier",
  "seed_param": "random_state",
  "default_params": {
    "n_estimators": 100,
    "learning_rate": 0.1
  }
}
```

**Step 2** — Create an experiment config at `model_config/gradient_boosting/config_random_search_gradient_boosting_exp_000001.json` with the relevant search space.

**Step 3** — Run the pipeline and enter `gradient_boosting` / `random_search` / `000001` at the prompts.

---

## Requirements

```
numpy==1.23.5
pandas==1.5.3
scikit-learn==1.2.2
joblib==1.2.0
openpyxl==3.0.10
fpdf==1.7.2
```

Install with:

```bash
pip install -r requirements.txt
```

> `argparse` and `json` are part of the Python standard library — no separate install needed.

---

## Contact

For questions or feedback: [lordmuze@gmail.com](mailto:lordmuze@gmail.com)
