# train.py - UPDATED VERSION
"""
train.py - OPTIMIZED FOR MAX F1
"""

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


def main():
    print("=" * 80)
    print(" TDE MALLORN – ULTIMATE F1 OPTIMIZATION TRAINING")
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
    # Fill NaN target với 0 (Non-TDE)
    df["target"] = df["target"].fillna(0).astype(int)
    y = df["target"]
    
    print(f" Total samples : {len(df)}")
    print(f" TDE count     : {y.sum()}")
    print(f" TDE ratio     : {y.mean():.4f}")
    
    # -------------------------------------------------
    # FEATURE SELECTION STRATEGY
    # -------------------------------------------------
    # 1. Remove non-predictive columns
    drop_cols = [
        "object_id", "target", "split", "SpecType", 
        "English Translation", "Z_err"
    ]
    
    X_full = df.drop(columns=drop_cols, errors="ignore")
    
    # 2. Handle categorical columns
    cat_cols = X_full.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        print(f" Converting categorical: {cat_cols.tolist()}")
        for col in cat_cols:
            X_full[col] = pd.factorize(X_full[col])[0]
    
    # 3. Handle NaN
    if X_full.isnull().any().any():
        print(" NaNs detected – filling with column median")
        X_full = X_full.fillna(X_full.median())
    
    print(f" Initial features : {X_full.shape[1]}")
    
    # -------------------------------------------------
    # STRATEGY 1: TRAIN WITH ALL FEATURES
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
    # STRATEGY 2: FEATURE SELECTION + RETRAIN
    # -------------------------------------------------
    print("\n" + "="*60)
    print("STRATEGY 2: TOP FEATURES ONLY")
    print("="*60)
    
    # Select top features based on importance
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
    # STRATEGY 3: ENSEMBLE
    # -------------------------------------------------
    print("\n" + "="*60)
    print("STRATEGY 3: ENSEMBLE MODELS")
    print("="*60)
    
    # Use selected features for ensemble (better performance)
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
    
    best_strategy = None
    best_f1 = 0
    
    for name, result in strategies.items():
        print(f"\n{name}:")
        print(f"  OOF AUC: {result['oof_auc']:.4f}")
        print(f"  OOF F1:  {result['oof_f1']:.4f}")
        print(f"  Threshold: {result['best_threshold']:.3f}")
        
        if result['oof_f1'] > best_f1:
            best_f1 = result['oof_f1']
            best_strategy = name
            best_result = result
    
    print(f"\n BEST STRATEGY: {best_strategy} (F1: {best_f1:.4f})")
    
    # -------------------------------------------------
    # SAVE BEST MODEL
    # -------------------------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODELS_DIR / f"tde_lgbm_best_{ts}.pkl"
    
    # Chọn features tương ứng
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
            "n_pos": best_result.get("n_pos", y.sum()),
            "n_neg": best_result.get("n_neg", len(y) - y.sum()),
        },
        model_path,
    )
    
    print(f"\n Best model saved → {model_path}")
    
    # -------------------------------------------------
    # FEATURE IMPORTANCE ANALYSIS
    # -------------------------------------------------
    print("\n" + "="*80)
    print(" FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    if "feature_importance" in result_all and result_all["feature_importance"] is not None:
        importance_df = result_all["feature_importance"]
        
        # Save importance
        importance_path = MODELS_DIR / f"feature_importance_{ts}.csv"
        importance_df.to_csv(importance_path, index=False)
        print(f" Feature importance saved → {importance_path}")
        
        # Show most important features
        print("\n Top 20 Most Important Features:")
        top_features = importance_df.nlargest(20, 'importance_mean')
        for idx, row in top_features.iterrows():
            importance_bar = "█" * int(row['importance_mean'] / top_features['importance_mean'].max() * 50)
            print(f"  {row['feature']:30s} |{importance_bar:<50}| {row['importance_mean']:.1f}")
    
    print("\n" + "="*80)
    print(f" TRAINING COMPLETED - BEST F1: {best_f1:.4f}")
    print("="*80)
    
    return best_result


if __name__ == "__main__":
    main()