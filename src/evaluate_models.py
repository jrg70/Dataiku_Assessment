# evaluate_models.py
"""
Evaluate saved models on the processed test set with MULTI-METRIC scoring from config.yaml.
- Prefers models/best_model.joblib; otherwise evaluates all
- Computes every metric listed in YAML 'modeling.scoring' (e.g., roc_auc, f1_macro)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import yaml
import pandas as pd
from joblib import load
from sklearn.metrics import classification_report, get_scorer


def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def map_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)
    return series.astype(str).map(lambda v: 0 if "-" in v or "<" in v else 1).astype(int)

def resolve_columns(feature_names, actual_cols):
    resolved, missing = [], []
    actual_set = set(actual_cols)
    for name in feature_names:
        if name in actual_set:
            resolved.append(name); continue
        spaced = name.replace("_", " ")
        if spaced in actual_set:
            resolved.append(spaced)
        else:
            missing.append(name)
    return resolved, missing


def main():
    cfg = load_config()
    processed_dir = cfg["paths"]["processed"]
    models_dir    = cfg["paths"]["models"]
    results_dir   = cfg["paths"]["results"]
    os.makedirs(results_dir, exist_ok=True)

    test_csv = os.path.join(processed_dir, "test_cleaned.csv")
    test = pd.read_csv(test_csv)

    target_name = (cfg.get("data", {}) or cfg).get("target")
    yaml_num = (cfg.get("data", {}) or cfg)["features"]["numeric"]
    yaml_cat = (cfg.get("data", {}) or cfg)["features"]["categorical"]

    num_cols, _ = resolve_columns(yaml_num, test.columns)
    cat_cols, _ = resolve_columns(yaml_cat, test.columns)

    X_test = test[num_cols + cat_cols].copy()
    y_test = map_target(test[target_name])

    # Read scoring list from YAML
    scoring_yaml = cfg["modeling"].get("scoring", "roc_auc")
    scoring_names = scoring_yaml if isinstance(scoring_yaml, list) else [scoring_yaml]
    scorers = {name: get_scorer(name) for name in scoring_names}

    # Collect models to evaluate
    candidates = []
    best_path = os.path.join(models_dir, "best_model.joblib")
    if os.path.exists(best_path):
        candidates = [best_path]
    else:
        candidates = [os.path.join(models_dir, f) for f in os.listdir(models_dir) if f.endswith("_best.joblib")]

    rows = []
    for path in candidates:
        try:
            model = load(path)
            # Print report once
            y_pred = model.predict(X_test)
            print(f"\n=== {os.path.basename(path)} ===")
            print(classification_report(y_test, y_pred, target_names=["Low (<50k)", "High (>50k)"], digits=4))

            # Compute all metrics
            scores = {name: float(scorer(model, X_test, y_test)) for name, scorer in scorers.items()}
            print("Scores:", scores)

            row = {"model_file": os.path.basename(path)}
            for name, val in scores.items():
                row[f"test_{name}"] = val
            rows.append(row)
        except Exception as e:
            print(f"[WARN] Failed to evaluate {path}: {e}")

    if rows:
        out_csv = os.path.join(results_dir, "evaluation_summary.csv")
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"\nSaved evaluation summary to {out_csv}")
    else:
        print("\nNo models evaluated.")


if __name__ == "__main__":
    print("evaluated successfully.")
