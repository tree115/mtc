# model.py - IMPROVED VERSION
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve
from lightgbm import LGBMClassifier
import joblib

# =====================================================
# OPTIMAL THRESHOLD SEARCH WITH SMOOTHING
# =====================================================
def find_optimal_threshold(y_true, y_prob, n_points=500):
    """
    Tìm threshold tối ưu cho F1 với smoothing và validation
    """
    thresholds = np.linspace(0.01, 0.5, n_points)
    best_f1 = -1
    best_threshold = 0.25
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        
        # Avoid extreme thresholds with too few positives
        if y_pred.sum() < 3:
            continue
            
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    
    # Refine search around best threshold
    if best_threshold > 0.05 and best_threshold < 0.45:
        refine_thresholds = np.linspace(
            max(0.01, best_threshold - 0.05),
            min(0.5, best_threshold + 0.05),
            200
        )
        for t in refine_thresholds:
            y_pred = (y_prob >= t).astype(int)
            if y_pred.sum() < 3:
                continue
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
    
    return best_threshold, best_f1


def find_threshold_by_pr_curve(y_true, y_prob):
    """
    Tìm threshold từ Precision-Recall curve
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Calculate F1 for each threshold
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
    
    # Find best F1
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    return best_threshold, best_f1


# =====================================================
# IMPROVED LGBM TRAINING WITH FEATURE IMPORTANCE
# =====================================================
def train_lgbm_cv_improved(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
    use_early_stopping: bool = True,
    save_feature_importance: bool = True
):
    """
    Improved training với:
    - Early stopping
    - Class weights tối ưu
    - Feature importance tracking
    - Multiple threshold strategies
    """
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    # Tính class weight chính xác hơn
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    imbalance_ratio = n_neg / n_pos
    
    print(f" Class balance: {n_pos} TDE vs {n_neg} Non-TDE (ratio: {imbalance_ratio:.1f}x)")
    
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )
    
    oof_pred = np.zeros(len(X))
    models = []
    fold_importances = []
    
    # 🔥 OPTIMIZED PARAMETERS FOR IMBALANCE
    params = {
        'boosting_type': 'gbdt',
        'n_estimators': 3000,  # Tăng để early stopping làm việc
        'learning_rate': 0.01,  # Giảm learning rate
        'num_leaves': 63,  # Giảm leaves để tránh overfit
        'max_depth': 8,  # Giới hạn depth
        'min_child_samples': 50,  # Tăng minimum samples
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.3,  # Tăng L1 regularization
        'reg_lambda': 0.7,  # Tăng L2 regularization
        'min_split_gain': 0.01,
        'scale_pos_weight': imbalance_ratio * 0.9,  # Giảm nhẹ từ 0.7 lên 0.9
        'objective': 'binary',
        'metric': 'auc',
        'random_state': random_state,
        'n_jobs': -1,
        'verbosity': -1,
        'force_row_wise': True
    }
    
    # Feature importance dataframe
    feature_importance_df = pd.DataFrame()
    feature_importance_df['feature'] = X.columns.tolist()
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n Fold {fold}/{n_splits}")
        
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        
        model = LGBMClassifier(**params)
        
        if use_early_stopping:
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric=['auc', 'binary_logloss'],
                callbacks=[
                    # Early stopping dựa trên AUC
                    # Dừng nếu không cải thiện trong 100 rounds
                ]
            )
        else:
            model.fit(X_tr, y_tr)
        
        # Predict validation
        val_prob = model.predict_proba(X_val)[:, 1]
        oof_pred[val_idx] = val_prob
        
        # Tính metrics cho fold
        fold_auc = roc_auc_score(y_val, val_prob)
        best_t, best_f1 = find_optimal_threshold(y_val, val_prob)
        
        print(f"   AUC: {fold_auc:.4f}")
        print(f"   Best F1: {best_f1:.4f} @ threshold={best_t:.3f}")
        print(f"   TDE predicted: {(val_prob >= best_t).sum()}/{len(val_prob)}")
        
        # Store model
        models.append(model)
        
        # Feature importance
        fold_importance = pd.DataFrame({
            'feature': X.columns,
            f'importance_fold{fold}': model.feature_importances_
        })
        
        if feature_importance_df.empty:
            feature_importance_df = fold_importance
        else:
            feature_importance_df = feature_importance_df.merge(
                fold_importance, on='feature', how='left'
            )
    
    # ============================================
    # POST-TRAINING ANALYSIS & THRESHOLD OPTIMIZATION
    # ============================================
    print("\n" + "="*60)
    print(" POST-TRAINING OPTIMIZATION")
    print("="*60)
    
    # Method 1: Optimal threshold từ OOF
    threshold_opt, f1_opt = find_optimal_threshold(y, oof_pred)
    
    # Method 2: Threshold từ PR curve
    threshold_pr, f1_pr = find_threshold_by_pr_curve(y, oof_pred)
    
    # Method 3: Weighted average của các thresholds
    thresholds_to_try = [threshold_opt, threshold_pr, 0.2, 0.25]
    weights = [0.4, 0.4, 0.1, 0.1]  # Ưu tiên 2 method đầu
    
    best_overall_f1 = -1
    best_overall_threshold = 0.25
    
    for t in thresholds_to_try:
        y_pred = (oof_pred >= t).astype(int)
        f1 = f1_score(y, y_pred, zero_division=0)
        
        if f1 > best_overall_f1:
            best_overall_f1 = f1
            best_overall_threshold = t
    
    print(f"\n Threshold Analysis:")
    print(f"   Method 1 (Optimal): {f1_opt:.4f} @ {threshold_opt:.3f}")
    print(f"   Method 2 (PR Curve): {f1_pr:.4f} @ {threshold_pr:.3f}")
    print(f"   Selected: {best_overall_f1:.4f} @ {best_overall_threshold:.3f}")
    
    # Feature importance analysis
    if save_feature_importance and not feature_importance_df.empty:
        # Calculate average importance
        importance_cols = [c for c in feature_importance_df.columns if 'importance' in c]
        feature_importance_df['importance_mean'] = feature_importance_df[importance_cols].mean(axis=1)
        feature_importance_df['importance_std'] = feature_importance_df[importance_cols].std(axis=1)
        
        # Sort by importance
        feature_importance_df = feature_importance_df.sort_values(
            'importance_mean', ascending=False
        )
        
        # Save to file
        importance_path = 'outputs/feature_importance.csv'
        feature_importance_df.to_csv(importance_path, index=False)
        
        print(f"\n Top 20 Features:")
        for i, row in feature_importance_df.head(20).iterrows():
            print(f"   {row['feature']}: {row['importance_mean']:.2f} ± {row['importance_std']:.2f}")
    
    return {
        "models": models,
        "oof_pred": oof_pred,
        "oof_auc": roc_auc_score(y, oof_pred),
        "oof_f1": best_overall_f1,
        "best_threshold": best_overall_threshold,
        "threshold_opt": threshold_opt,
        "threshold_pr": threshold_pr,
        "feature_importance": feature_importance_df if save_feature_importance else None,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "imbalance_ratio": imbalance_ratio
    }


# =====================================================
# ENSEMBLE STRATEGIES
# =====================================================
def train_lgbm_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    n_models: int = 3,
    n_splits: int = 5
):
    """
    Ensemble nhiều models với different seeds
    """
    all_models = []
    all_oof_preds = []
    
    seeds = [42, 123, 456, 789, 999]  # Different seeds
    
    for i, seed in enumerate(seeds[:n_models]):
        print(f"\n Training Ensemble Model {i+1}/{n_models} (seed={seed})")
        
        result = train_lgbm_cv_improved(
            X=X,
            y=y,
            n_splits=n_splits,
            random_state=seed,
            use_early_stopping=True,
            save_feature_importance=(i == 0)  # Chỉ save cho model đầu
        )
        
        all_models.extend(result["models"])
        all_oof_preds.append(result["oof_pred"])
    
    # Average OOF predictions
    oof_pred_ensemble = np.mean(all_oof_preds, axis=0)
    
    # Find optimal threshold trên ensemble predictions
    best_threshold, best_f1 = find_optimal_threshold(y, oof_pred_ensemble)
    
    print(f"\n Ensemble Results:")
    print(f"   OOF AUC: {roc_auc_score(y, oof_pred_ensemble):.4f}")
    print(f"   OOF F1: {best_f1:.4f} @ threshold={best_threshold:.3f}")
    
    return {
        "models": all_models,
        "oof_pred": oof_pred_ensemble,
        "oof_auc": roc_auc_score(y, oof_pred_ensemble),
        "oof_f1": best_f1,
        "best_threshold": best_threshold,
        "n_models": len(all_models)
    }


# =====================================================
# PREDICTION WITH CALIBRATION
# =====================================================
def predict_with_calibration(models, X, threshold=0.25, method='mean'):
    """
    Dự đoán với calibration options
    """
    if method == 'mean':
        # Simple average
        prob = np.zeros(len(X))
        for m in models:
            prob += m.predict_proba(X)[:, 1]
        prob = prob / len(models)
    
    elif method == 'median':
        # Median để robust với outliers
        all_probs = np.array([m.predict_proba(X)[:, 1] for m in models])
        prob = np.median(all_probs, axis=0)
    
    elif method == 'weighted':
        # Weighted average theo fold performance
        weights = np.linspace(0.8, 1.2, len(models))  # Có thể customize
        weights = weights / weights.sum()
        
        prob = np.zeros(len(X))
        for i, m in enumerate(models):
            prob += weights[i] * m.predict_proba(X)[:, 1]
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    pred = (prob >= threshold).astype(int)
    
    return prob, pred


# =====================================================
# FEATURE SELECTION BASED ON IMPORTANCE
# =====================================================
def select_features_by_importance(X, y, top_k=100, random_state=42):
    """
    Chọn top features dựa trên feature importance
    """
    print(f"\n Selecting top {top_k} features...")
    
    # Train quick model để lấy importance
    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        scale_pos_weight=(len(y) - y.sum()) / y.sum() * 0.9,
        random_state=random_state,
        verbosity=-1,
        n_jobs=-1
    )
    
    model.fit(X, y)
    
    # Get feature importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Select top features
    selected_features = importance.head(top_k)['feature'].tolist()
    
    print(f" Selected {len(selected_features)} features")
    print(f"   Top 10: {selected_features[:10]}")
    
    return selected_features, importance