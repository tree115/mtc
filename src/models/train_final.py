"""
train_final.py
FINAL TRAINING USING OPTUNA BEST HYPERPARAMETERS
100% compatible with original train.py & predict.py
"""

import json
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from pathlib import Path

from config import TRAIN_FEATURES, MODELS_DIR
from model import (
    train_lgbm_cv_improved,
    train_lgbm_ensemble,
    select_features_by_importance
)

# =====================================================
# LOAD OPTUNA BEST PARAMS
# =====================================================
OPTUNA_JSON = Path("outputs/optuna_best_params.json")
if not OPTUNA_JSON.exists():
    raise FileNotFoundError(" optuna_best_params.json not found")

with open(OPTUNA_JSON, "r") as f:
    optuna_result = json.load(f)

BEST_PARAMS = optuna_result["best_params"]

print("\n USING OPTUNA BEST PARAMS")
for k, v in BEST_PARAMS.items():
    print(f"  {k}: {v}")

# =====================================================
# PATCH LGBMClassifier INIT (SAFE – NO SIDE EFFECT)
# =====================================================
from lightgbm import LGBMClassifier

_old_init = LGBMClassifier.__init__

def _patched_init(self, **kwargs):
    base_params = {
        "boosting_type": "gbdt",
        "n_estimators": 3000,
        "objective": "binary",
        "metric": "auc",
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
        "force_row_wise": True,
    }
    base_params.update(BEST_PARAMS)
    base_params.update(kwargs)
    _old_init(self, **base_params)

LGBMClassifier.__init__ = _patched_init


# =====================================================
# MAIN TRAINING (COPY OF train.py)
# =====================================================
def main():
    print("=" * 80)
    print(" TDE MALLORN FINAL TRAINING (OPTUNA)")
    print("=" * 80)

    # -------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------
    if not TRAIN_FEATURES.exists():
        raise FileNotFoundError(f" File not found: {TRAIN_FEATURES}")

    print(" Loading train features...")
    df = pd.read_csv(TRAIN_FEATURES)

    # -------------------------------------------------
    # DATA CLEANING & PREPARATION
    # -------------------------------------------------
    df["target"] = df["target"].fillna(0).astype(int)
    y = df["target"]

    print(f" Total samples : {len(df)}")
    print(f" TDE count     : {y.sum()}")
    print(f" TDE ratio     : {y.mean():.4f}")

    drop_cols = [
        "object_id", "target", "split", "SpecType",
        "English Translation", "Z_err"
    ]

    X_full = df.drop(columns=drop_cols, errors="ignore")

    # categorical → factorize (GIỐNG TRAIN CŨ)
    cat_cols = X_full.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        print(f" Converting categorical: {cat_cols.tolist()}")
        for col in cat_cols:
            X_full[col] = pd.factorize(X_full[col])[0]

    # fill NaN
    if X_full.isnull().any().any():
        print("NaNs detected – filling with column median")
        X_full = X_full.fillna(X_full.median())

    print(f" Initial features : {X_full.shape[1]}")

    # -------------------------------------------------
    # STRATEGY 1: ALL FEATURES
    # -------------------------------------------------
    print("\n" + "="*60)
    print("STRATEGY 1: TRAIN WITH ALL FEATURES")
    print("="*60)

    result_all = train_lgbm_cv_improved(
        X=X_full,
        y=y,
        n_splits=5,
        random_state=42,
        use_early_stopping=True,
        save_feature_importance=True
    )

    # -------------------------------------------------
    # STRATEGY 2: FEATURE SELECTION
    # -------------------------------------------------
    print("\n" + "="*60)
    print("STRATEGY 2: TOP FEATURES ONLY")
    print("="*60)

    selected_features, importance_df = select_features_by_importance(
        X=X_full,
        y=y,
        top_k=min(100, X_full.shape[1]),
        random_state=42
    )

    X_selected = X_full[selected_features]

    result_selected = train_lgbm_cv_improved(
        X=X_selected,
        y=y,
        n_splits=5,
        random_state=42,
        use_early_stopping=True,
        save_feature_importance=False
    )

    # -------------------------------------------------
    # STRATEGY 3: ENSEMBLE (OPTUNA PARAMS)
    # -------------------------------------------------
    print("\n" + "="*60)
    print("STRATEGY 3: ENSEMBLE MODELS")
    print("="*60)

    result_ensemble = train_lgbm_ensemble(
        X=X_selected,
        y=y,
        n_models=3,
        n_splits=5
    )

    # -------------------------------------------------
    # COMPARE STRATEGIES
    # -------------------------------------------------
    print("\n" + "="*80)
    print(" STRATEGY COMPARISON")
    print("="*80)

    strategies = {
        "All Features": result_all,
        "Selected Features": result_selected,
        "Ensemble": result_ensemble
    }

    best_f1 = 0
    best_strategy = None
    best_result = None

    for name, result in strategies.items():
        print(f"\n{name}:")
        print(f"  OOF AUC: {result['oof_auc']:.4f}")
        print(f"  OOF F1:  {result['oof_f1']:.4f}")
        print(f"  Threshold: {result['best_threshold']:.3f}")

        if result["oof_f1"] > best_f1:
            best_f1 = result["oof_f1"]
            best_strategy = name
            best_result = result

    print(f"\n BEST STRATEGY: {best_strategy} (F1: {best_f1:.4f})")

    # -------------------------------------------------
    # SAVE MODEL (SAME FORMAT AS OLD)
    # -------------------------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_path = MODELS_DIR / f"tde_lgbm_optuna_final_{ts}.pkl"

    if best_strategy == "All Features":
        features_used = X_full.columns.tolist()
    else:
        features_used = selected_features

    joblib.dump(
        {
            "models": best_result["models"],
            "features": features_used,
            "threshold": best_result["best_threshold"],
            "oof_f1": best_result["oof_f1"],
            "oof_auc": best_result["oof_auc"],
            "strategy": best_strategy,
            "n_features": len(features_used),
            "created_at": ts,
            "optuna_f1": optuna_result["best_f1"],
            "optuna_params": BEST_PARAMS,
        },
        model_path,
    )

    print(f"\n FINAL MODEL SAVED → {model_path}")
    print(f" FINAL OOF F1 → {best_result['oof_f1']:.4f}")

    return best_result


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    try:
        main()
    finally:
        # restore LGBM init
        LGBMClassifier.__init__ = _old_init
