# train.py
"""
- Reads config.yaml
- Load train_cleaned.csv and test_cleaned.csv (as saved by run.py)
- Simple preprocessing: impute (most_frequent), one-hot (drop='first'), scale numerics
- Trains & tunes 4 models: LogisticRegression, DecisionTree, RandomForest, GradientBoosting
- CV optimizes the FIRST metric in YAML `modeling.scoring`
- Test reports ALL metrics from YAML `modeling.scoring`
- Saves:
    models/<model_name>_best.joblib
    models/best_model.joblib
    results/training_summary.csv
"""

import os
import warnings
warnings.filterwarnings("ignore")

import yaml
import numpy as np
import pandas as pd
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, get_scorer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from xgboost import XGBClassifier

# ---------- small helpers ----------
def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def map_target(series: pd.Series) -> pd.Series:
    """Map '-50k'/ '<=50K' -> 0 and '>50k' -> 1 to obtain binary label."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)
    return series.astype(str).map(lambda v: 0 if "-" in v or "<" in v else 1).astype(int)

def resolve_columns(feature_names, actual_cols):
    """
    YAML uses snake_case; CSV might have spaces.
    """
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


# ---------- main ----------
def main():
    cfg = load_config()

    # Paths
    processed_dir = cfg["paths"]["processed"]
    models_dir    = cfg["paths"]["models"]
    results_dir   = cfg["paths"]["results"]
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Files (as saved by your run.py)
    train_csv = os.path.join(processed_dir, "train_cleaned.csv")
    test_csv  = os.path.join(processed_dir, "test_cleaned.csv")

    # Load data
    train = pd.read_csv(train_csv)
    test  = pd.read_csv(test_csv)

    # Target & features from YAML (works whether you nest under "data" or not)
    conf_data = cfg.get("data", cfg)
    target_name = conf_data["target"] if "target" in conf_data else cfg["target"]
    yaml_num = conf_data["features"]["numeric"]
    yaml_cat = conf_data["features"]["categorical"]

    # Resolve column names against the CSV
    num_cols, miss_num = resolve_columns(yaml_num, train.columns)
    cat_cols, miss_cat = resolve_columns(yaml_cat, train.columns)
    if miss_num or miss_cat:
        print("Skipping missing columns from YAML as they were left out from preprocessing:")
        if miss_num: print("  numeric:", miss_num)
        if miss_cat: print("  categorical:", miss_cat)

    # Split
    X_train = train[num_cols + cat_cols].copy()
    y_train = map_target(train[target_name])
    X_test  = test[num_cols + cat_cols].copy()
    y_test  = map_target(test[target_name])

    # Preprocessing
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", sparse_output=False, min_frequency=10, handle_unknown="infrequent_if_exist")),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop"
    )

    # Modeling/setup
    modeling = cfg["modeling"]
    cv_folds = modeling.get("cv_folds", 5)
    scoring_cfg = modeling.get("scoring", "roc_auc")

    scoring_list = scoring_cfg if isinstance(scoring_cfg, list) else [scoring_cfg]
    refit_metric = scoring_list[0]
    seed = cfg.get("random_seed", 42)

    # Define models and parameter grids from YAML
    grids = {}

    lr = LogisticRegression(class_weight="balanced", random_state=seed, max_iter=2000)
    lr_grid = {f"model__{k}": v for k, v in modeling["models"].get("logreg", {}).items()}
    grids["logistic_regression"] = GridSearchCV(
        Pipeline([("pre", pre), ("model", lr)]),
        param_grid=lr_grid or {"model__C": [1.0]},
        cv=cv_folds, scoring=refit_metric, n_jobs=-1, verbose=1
    )

    dt = DecisionTreeClassifier(criterion="entropy", class_weight="balanced", random_state=seed)
    dt_grid = {f"model__{k}": v for k, v in modeling["models"].get("decision_tree", {}).items()}
    grids["decision_tree"] = GridSearchCV(
        Pipeline([("pre", pre), ("model", dt)]),
        param_grid=dt_grid or {"model__max_depth": [None]},
        cv=cv_folds, scoring=refit_metric, n_jobs=-1, verbose=1
    )

    rf = RandomForestClassifier(class_weight="balanced", random_state=seed)
    rf_grid = {f"model__{k}": v for k, v in modeling["models"].get("random_forest", {}).items()}
    grids["random_forest"] = GridSearchCV(
        Pipeline([("pre", pre), ("model", rf)]),
        param_grid=rf_grid or {"model__n_estimators": [300]},
        cv=cv_folds, scoring=refit_metric, n_jobs=-1, verbose=1
    )

    gb = GradientBoostingClassifier(random_state=seed)
    gb_grid = {f"model__{k}": v for k, v in modeling["models"].get("gradient_boosting", {}).items()}
    grids["gradient_boosting"] = GridSearchCV(
        Pipeline([("pre", pre), ("model", gb)]),
        param_grid=gb_grid or {"model__n_estimators": [100]},
        cv=cv_folds, scoring=refit_metric, n_jobs=-1, verbose=1
    )

    xgb = XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",          
        random_state=seed,
        n_jobs=-1,
        eval_metric="logloss"
    )

    xgb_grid_raw = modeling["models"].get("xgboost", {})
    def _wrap(v): return v if isinstance(v, (list, tuple, np.ndarray)) else [v]
    xgb_grid = {f"model__{k}": _wrap(v) for k, v in (xgb_grid_raw or {}).items()}
    if not xgb_grid:
        xgb_grid = {"model__n_estimators": [200]} 
    grids["xgboost"] = GridSearchCV(
        Pipeline([("pre", pre), ("model", xgb)]),
        param_grid=xgb_grid,
        cv=cv_folds, scoring=refit_metric, n_jobs=-1, verbose=1
    )

    # Train, evaluate, save
    summary_rows = []
    best_name, best_score, best_est = None, -np.inf, None

    for name, gscv in grids.items():
        print(f"\n=== Tuning {name} ===")
        gscv.fit(X_train, y_train)
        print(f"Best params ({refit_metric}): {gscv.best_params_}")

        # Test-set metrics (compute ALL metrics listed in YAML)
        test_scores = {}
        for metric in scoring_list:
            scorer = get_scorer(metric)
            test_scores[metric] = float(scorer(gscv.best_estimator_, X_test, y_test))

        # report
        y_pred = gscv.best_estimator_.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=["Low (<50k)", "High (>50k)"], digits=4))
        print("Test scores:", test_scores)

        # Save best estimator for model
        out_path = os.path.join(models_dir, f"{name}_best.joblib")
        dump(gscv.best_estimator_, out_path)
        print(f"Saved: {out_path}")

        # Record summary row
        row = {"model": name}
        for m, val in test_scores.items():
            row[f"test_{m}"] = val
        summary_rows.append(row)

        # Pick overall best by the FIRST metric (refit_metric check in yaml)
        if test_scores.get(refit_metric, -np.inf) > best_score:
            best_score = test_scores[refit_metric]
            best_name  = name
            best_est   = gscv.best_estimator_

    # Save overall best
    if best_est is not None:
        dump(best_est, os.path.join(models_dir, "best_model.joblib"))
        print(f"\nOverall best (by TEST {refit_metric}): {best_name} = {best_score:.4f}")

    # Save summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(results_dir, "training_summary.csv")
    summary_df.sort_values(f"test_{refit_metric}", ascending=False).to_csv(summary_csv, index=False)
    print(f"Summary saved to {summary_csv}")


if __name__ == "__main__":
    main()
