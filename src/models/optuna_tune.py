# optuna_tune.py
"""
Optuna hyperparameter tuning (train pipeline identical to train.py)

- Uses train_lgbm_cv_improved from model.py (exact same training pipeline)
- Builds a base param dict consistent with model.train_lgbm_cv_improved defaults
- Monkey-patches lightgbm.sklearn.LGBMClassifier CLASS (NOT __init__)
  so models created inside train_lgbm_cv_improved receive:
    base_params + trial params + caller kwargs
- Fully compatible with scikit-learn / LightGBM >= 4.x
- Restores original class after each trial
"""

import json
from pathlib import Path
import optuna
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from model import train_lgbm_cv_improved
from config import TRAIN_FEATURES

# -------------------------
# CONFIG (match train.py)
# -------------------------
N_SPLITS = 5
RANDOM_STATE = 42
N_TRIALS = 80          # recommend 80 on Colab
TIMEOUT = 5 * 60 * 60  # 5 hours
SEED = 42

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# LOAD DATA (IDENTICAL TO train.py)
# -------------------------
print("📥 Loading data (same as train.py)...")
if not TRAIN_FEATURES.exists():
    raise FileNotFoundError(f"TRAIN_FEATURES not found: {TRAIN_FEATURES}")

df = pd.read_csv(TRAIN_FEATURES)

df["target"] = df["target"].fillna(0).astype(int)
y = df["target"]

drop_cols = [
    "object_id", "target", "split", "SpecType",
    "English Translation", "Z_err"
]
X = df.drop(columns=drop_cols, errors="ignore")

cat_cols = X.select_dtypes(include=["object", "category"]).columns
for col in cat_cols:
    X[col] = pd.factorize(X[col])[0]

if X.isnull().any().any():
    X = X.fillna(X.median())

print(f"🧠 Features: {X.shape[1]}")
print(f"🎯 Positives: {y.sum()} / {len(y)}")

n_pos = y.sum()
n_neg = len(y) - n_pos
imbalance_ratio = n_neg / max(1, n_pos)
print(f"📊 Class balance: {n_pos} TDE vs {n_neg} Non-TDE (ratio: {imbalance_ratio:.1f}x)")

# -------------------------
# BASE PARAMS (IDENTICAL TO model.py)
# -------------------------
def get_base_params(random_state: int, imbalance_ratio: float):
    return {
        "boosting_type": "gbdt",
        "n_estimators": 3000,
        "learning_rate": 0.01,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.3,
        "reg_lambda": 0.7,
        "min_split_gain": 0.01,
        "scale_pos_weight": imbalance_ratio * 0.9,
        "objective": "binary",
        "metric": "auc",
        "random_state": random_state,
        "n_jobs": -1,
        "verbosity": -1,
        "force_row_wise": True,
    }

# -------------------------
# OPTUNA SEARCH SPACE
# -------------------------
def build_search_space(trial: optuna.Trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 800, 3000),
        "scale_pos_weight_mult": trial.suggest_float("scale_pos_weight_mult", 0.5, 1.5),
    }

# -------------------------
# OPTUNA OBJECTIVE (FIXED)
# -------------------------
def objective(trial: optuna.Trial):
    trial_params = build_search_space(trial)

    if "scale_pos_weight_mult" in trial_params:
        mult = trial_params.pop("scale_pos_weight_mult")
        trial_params["scale_pos_weight"] = imbalance_ratio * mult

    import lightgbm.sklearn
    _OldLGBM = lightgbm.sklearn.LGBMClassifier

    class PatchedLGBMClassifier(_OldLGBM):
        def __init__(self, **kwargs):
            base = get_base_params(RANDOM_STATE, imbalance_ratio)
            merged = {}
            merged.update(base)
            merged.update(trial_params)
            merged.update(kwargs)
            super().__init__(**merged)

    lightgbm.sklearn.LGBMClassifier = PatchedLGBMClassifier

    try:
        res = train_lgbm_cv_improved(
            X=X,
            y=y,
            n_splits=N_SPLITS,
            random_state=RANDOM_STATE,
            use_early_stopping=True,
            save_feature_importance=True,
        )
        f1 = res.get("oof_f1", 0.0)
        if f1 is None or np.isnan(f1):
            f1 = 0.0

    except Exception as e:
        print("❌ Trial failed:", repr(e))
        f1 = 0.0

    finally:
        lightgbm.sklearn.LGBMClassifier = _OldLGBM

    return float(f1)

# -------------------------
# RUN OPTUNA
# -------------------------
if __name__ == "__main__":
    np.random.seed(SEED)

    sampler = optuna.samplers.TPESampler(
        seed=SEED, multivariate=True, group=True
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="lgbm_f1_baseline_model",
    )

    print("🚀 Starting Optuna tuning (SAFE, sklearn-compatible)...")
    study.optimize(
        objective,
        n_trials=N_TRIALS,
        timeout=TIMEOUT,
        show_progress_bar=True,
    )

    best_trial_params = dict(study.best_params)
    if "scale_pos_weight_mult" in best_trial_params:
        mult = best_trial_params.pop("scale_pos_weight_mult")
        best_trial_params["scale_pos_weight"] = imbalance_ratio * mult

    full_best = get_base_params(RANDOM_STATE, imbalance_ratio)
    full_best.update(best_trial_params)

    out = {
        "best_f1": study.best_value,
        "optuna_best_params": best_trial_params,
        "best_params_full": full_best,
        "n_trials": len(study.trials),
    }

    out_path = OUT_DIR / "optuna_best_params.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n🏆 BEST RESULT")
    print("F1:", study.best_value)
    print("\nMerged best params:")
    for k, v in full_best.items():
        print(f"  {k}: {v}")

    print(f"\n💾 Saved → {out_path}")
