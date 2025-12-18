import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

print("=== Tinh chỉnh chi tiết mô hình phân loại TDE với ELAsTiCC2 Features ===")

# 1. Tải dữ liệu MỚI từ PreprocessandFeatures.py
print("--- Đang tải dữ liệu mới ---")
df = pd.read_csv('processed_train_features_elastricc_enhanced.csv')

# Debug: Kiểm tra columns
print(f"Các columns trong dataframe: {list(df.columns)[:20]}...")

# Đảm bảo các columns bắt buộc tồn tại
required_columns = ['object_id', 'target', 'split']
for col in required_columns:
    if col not in df.columns:
        print(f"⚠️ Cảnh báo: Column '{col}' không tồn tại trong dataframe")
        print(f"Các columns hiện có: {list(df.columns)}")

# Xác định features và target
features_to_exclude = ['object_id', 'SpecType', 'English Translation', 'split', 'target']
# Loại bỏ các columns không tồn tại
features_to_exclude = [col for col in features_to_exclude if col in df.columns]

all_features = [col for col in df.columns if col not in features_to_exclude]
X = df[all_features]
y = df['target']
groups = df['split']

print(f"Kích thước dữ liệu: {X.shape}")
print(f"Tỷ lệ mẫu dương: {y.mean():.4f} ({y.sum()}/{len(y)})")
print(f"Số lượng features: {len(all_features)}")

# 2. Feature Selection NÂNG CAO với ELAsTiCC2 features
def select_optimal_features_with_elastricc(df):
    """Chọn đặc trưng tối ưu với integration từ ELAsTiCC2"""
    
    # Nhóm 1: Features cơ bản từ phiên bản trước
    basic_features = [
        'flux_abs_q25', 'flux_abs_q10', 'decay_alpha', 'r_max', 'mjd_span',
        'Z', 'flux_abs_q50', 'u_mean', 'i_max', 'u_max', 'peakiness',
        'color_r_i_mean', 'g_max', 'abs_mag_mean', 'abs_mag_min',
        'abs_mag_max', 'rise_fall_ratio', 'rise_fall_ratio_global',
        'asymmetry', 'u_std', 'positive_ratio', 'flux_abs_median',
        'variability_index', 'trend_stability', 'filter_coverage',
        'gap_max', 'obs_density', 'snr_mean', 'obs_count', 'flux_mean'
    ]
    
    # Nhóm 2: NEW FEATURES từ ELAsTiCC2 (quan trọng nhất)
    elastricc_features = [
        'rise_time', 'fade_time', 'rise_fade_ratio', 'rise_rate', 'fade_rate',
        'peak_count', 'peak_prominence_mean', 'peak_symmetry', 'peak_asymmetry',
        'color_g_r_mean', 'color_r_i_mean', 'color_i_z_mean',
        'color_g_i_mean', 'color_u_g_mean', 'color_z_y_mean',
        'autocorr_lag1', 'autocorr_lag2', 'autocorr_lag3',
        'p90_p10_ratio', 'p75_p25_ratio', 'p95_p5_ratio',
        'max_band_correlation', 'min_band_correlation',
        'total_duration', 'peak_to_median', 'peak_to_mean'
    ]
    
    # Kết hợp tất cả features
    all_candidate_features = basic_features + elastricc_features
    
    # Chỉ chọn features tồn tại trong dataframe
    selected = [f for f in all_candidate_features if f in df.columns]
    
    print(f"Tổng số features candidate: {len(all_candidate_features)}")
    print(f"Features tồn tại trong dataframe: {len(selected)}")
    print(f"Features ELAsTiCC2 mới: {len([f for f in selected if f in elastricc_features])}")
    
    # Hiển thị các features ELAsTiCC2 mới
    new_elastricc_features = [f for f in selected if f in elastricc_features]
    print("\nCác features ELAsTiCC2 mới được chọn:")
    for i, feat in enumerate(new_elastricc_features[:15]):
        print(f"  {i+1:2d}. {feat}")
    if len(new_elastricc_features) > 15:
        print(f"  ... và {len(new_elastricc_features) - 15} features khác")
    
    return selected

print("\n--- Lựa chọn đặc trưng với ELAsTiCC2 ---")
selected_features = select_optimal_features_with_elastricc(df)
X_optimal = df[selected_features]

# 3. THÊM: Feature Importance sơ bộ với RandomForest
print("\n--- Phân tích sơ bộ Feature Importance ---")
def quick_feature_importance_analysis(X, y):
    """Phân tích nhanh feature importance"""
    rf_quick = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_quick.fit(X.fillna(0), y)
    
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_quick.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 features quan trọng nhất (sơ bộ):")
    for i, row in importance_df.head(20).iterrows():
        print(f"  {row['importance']:.4f} - {row['feature']}")
    
    return importance_df

importance_df = quick_feature_importance_analysis(X_optimal, y)

# 4. Cải tiến hàm train_strategy với early stopping tốt hơn
def train_strategy(X, y, groups, params, strategy_name, early_stopping_rounds=100):
    """Huấn luyện một chiến lược đơn lẻ với cải tiến"""
    
    gkf = GroupKFold(n_splits=min(5, groups.nunique()))
    models = []
    oof_preds = np.zeros(len(X))
    fold_scores = []
    feature_importances = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"{strategy_name} - Fold {fold + 1}")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params, verbosity=-1, n_jobs=-1)
        
        # Fit với validation set
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=['binary_logloss', 'auc'],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=early_stopping_rounds, 
                    verbose=False,
                    first_metric_only=False
                ),
                lgb.log_evaluation(period=0)
            ]
        )
        
        # Dự đoán
        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        
        # Lưu feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importances.append(model.feature_importances_)
        
        # Tìm ngưỡng tối ưu cho fold này
        best_f1_fold = 0
        best_thresh_fold = 0.5
        
        # Tối ưu threshold với step nhỏ hơn
        for thresh in np.arange(0.05, 0.95, 0.01):
            f1 = f1_score(y_val, (val_preds >= thresh).astype(int))
            if f1 > best_f1_fold:
                best_f1_fold = f1
                best_thresh_fold = thresh
        
        fold_scores.append({
            'f1': best_f1_fold,
            'threshold': best_thresh_fold,
            'auc': roc_auc_score(y_val, val_preds)
        })
        
        models.append(model)
        print(f"  Fold {fold + 1}: F1={best_f1_fold:.4f}, Thresh={best_thresh_fold:.3f}, AUC={roc_auc_score(y_val, val_preds):.4f}")
    
    # Tính average feature importance
    avg_feature_importance = np.mean(feature_importances, axis=0) if feature_importances else None
    
    return models, oof_preds, fold_scores, avg_feature_importance

# 5. Cải tiến train_multiple_strategies
def train_multiple_strategies(X, y, groups):
    """Thử nghiệm nhiều chiến lược huấn luyện với tuning tốt hơn"""
    
    strategies = {}
    pos_weight = len(y) / (2 * np.sum(y))
    print(f"\nClass weight (pos_weight): {pos_weight:.2f}")
    
    gkf = GroupKFold(n_splits=min(5, groups.nunique()))
    
    # Chiến lược 1: Tuned for ELAsTiCC2 features (MODERATE)
    print("\n" + "="*60)
    print("Chiến lược 1: Tuned cho ELAsTiCC2 features (Moderate)")
    print("="*60)
    
    models1, oof1, scores1, fi1 = train_strategy(X, y, groups, {
        'n_estimators': 2000,
        'learning_rate': 0.03,  # Lower learning rate cho features phức tạp
        'num_leaves': 31,
        'max_depth': 6,  # Giảm depth để tránh overfit
        'min_child_samples': 30,  # Tăng để robust hơn
        'subsample': 0.7,  # Giảm để tránh overfit
        'colsample_bytree': 0.7,
        'reg_alpha': 0.5,  # Tăng regularization
        'reg_lambda': 0.7,
        'scale_pos_weight': pos_weight,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'random_state': 42
    }, "ELAsTiCC2 Tuned", early_stopping_rounds=80)
    
    strategies['elastricc_tuned'] = {
        'models': models1, 
        'oof_preds': oof1, 
        'scores': scores1,
        'feature_importance': fi1,
        'params': 'ELAsTiCC2 Tuned (Moderate)'
    }
    
    # Chiến lược 2: Strong Regularization
    print("\n" + "="*60)
    print("Chiến lược 2: Strong Regularization")
    print("="*60)
    
    models2, oof2, scores2, fi2 = train_strategy(X, y, groups, {
        'n_estimators': 1500,
        'learning_rate': 0.05,
        'num_leaves': 15,  # Rất nhỏ để tránh overfit
        'max_depth': 4,
        'min_child_samples': 50,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 1.0,
        'reg_lambda': 1.5,
        'scale_pos_weight': pos_weight,
        'random_state': 42
    }, "Strong Regularization", early_stopping_rounds=100)
    
    strategies['strong_reg'] = {
        'models': models2, 
        'oof_preds': oof2, 
        'scores': scores2,
        'feature_importance': fi2,
        'params': 'Strong Regularization'
    }
    
    # Chiến lược 3: Ensemble LightGBM + RandomForest (cải tiến)
    print("\n" + "="*60)
    print("Chiến lược 3: Ensemble Nâng cao")
    print("="*60)
    
    ensemble_preds = np.zeros(len(X))
    ensemble_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"Huấn luyện tổ hợp - Fold {fold + 1}")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # LightGBM với features mới
        lgb_model = lgb.LGBMClassifier(
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.3,
            reg_lambda=0.5,
            scale_pos_weight=pos_weight,
            random_state=42 + fold,
            verbosity=-1,
            n_jobs=-1
        )
        lgb_model.fit(X_train, y_train)
        lgb_preds = lgb_model.predict_proba(X_val)[:, 1]
        
        # RandomForest tuned cho features mới
        rf_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,  # Giảm depth
            min_samples_split=15,
            min_samples_leaf=8,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42 + fold,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        rf_preds = rf_model.predict_proba(X_val)[:, 1]
        
        # Weighted ensemble (70% LGBM, 30% RF)
        ensemble_pred = 0.7 * lgb_preds + 0.3 * rf_preds
        ensemble_preds[val_idx] = ensemble_pred
        
        # Tối ưu threshold
        best_f1_fold = 0
        for thresh in np.arange(0.05, 0.95, 0.01):
            f1 = f1_score(y_val, (ensemble_pred >= thresh).astype(int))
            if f1 > best_f1_fold:
                best_f1_fold = f1
        
        ensemble_scores.append({
            'f1': best_f1_fold,
            'auc': roc_auc_score(y_val, ensemble_pred)
        })
        print(f"  Fold {fold + 1}: F1={best_f1_fold:.4f}, AUC={roc_auc_score(y_val, ensemble_pred):.4f}")
    
    strategies['ensemble_enhanced'] = {
        'models': None,
        'oof_preds': ensemble_preds, 
        'scores': ensemble_scores,
        'params': 'Ensemble LGBM+RF (Weighted)'
    }
    
    return strategies

# 6. Cải tiến evaluate_strategies
def evaluate_strategies(strategies, y, X=None):
    """Đánh giá toàn bộ chiến lược huấn luyện với metrics đầy đủ"""
    
    results = []
    best_strategy = None
    best_f1 = 0
    
    for name, strategy in strategies.items():
        oof_preds = strategy['oof_preds']
        
        # Tối ưu ngưỡng với Precision-Recall tradeoff
        precision, recall, thresholds = precision_recall_curve(y, oof_preds)
        
        # Tìm threshold tốt nhất bằng F1 score
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
        best_idx = np.argmax(f1_scores[:-1])  # Bỏ giá trị cuối
        best_f1_score = f1_scores[best_idx]
        best_threshold = thresholds[best_idx] if len(thresholds) > best_idx else 0.5
        
        # Tính các metrics
        binary_preds = (oof_preds >= best_threshold).astype(int)
        
        # Precision, Recall, Specificity
        tp = np.sum((binary_preds == 1) & (y == 1))
        fp = np.sum((binary_preds == 1) & (y == 0))
        fn = np.sum((binary_preds == 0) & (y == 1))
        tn = np.sum((binary_preds == 0) & (y == 0))
        
        precision_metric = tp / (tp + fp + 1e-9)
        recall_metric = tp / (tp + fn + 1e-9)
        specificity = tn / (tn + fp + 1e-9)
        
        # AUC
        auc_score = roc_auc_score(y, oof_preds)
        
        # Phân tích fold scores
        fold_scores = strategy['scores']
        avg_fold_f1 = np.mean([s['f1'] for s in fold_scores]) if isinstance(fold_scores[0], dict) else np.mean(fold_scores)
        avg_fold_auc = np.mean([s.get('auc', 0) for s in fold_scores]) if isinstance(fold_scores[0], dict) else 0
        
        results.append({
            'strategy': name,
            'params': strategy['params'],
            'oof_f1': best_f1_score,
            'precision': precision_metric,
            'recall': recall_metric,
            'specificity': specificity,
            'auc': auc_score,
            'threshold': best_threshold,
            'avg_fold_f1': avg_fold_f1,
            'avg_fold_auc': avg_fold_auc,
            'fold_scores': fold_scores
        })
        
        if best_f1_score > best_f1:
            best_f1 = best_f1_score
            best_strategy = name
    
    return pd.DataFrame(results), best_strategy

# 7. THÊM: Visualize feature importance
def plot_feature_importance(feature_importance, feature_names, strategy_name, top_n=20):
    """Vẽ biểu đồ feature importance"""
    if feature_importance is None:
        return
    
    # Tạo DataFrame
    fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    # Lấy top features
    top_features = fi_df.head(top_n)
    
    # Vẽ biểu đồ
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=top_features)
    plt.title(f'Top {top_n} Feature Importance - {strategy_name}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    # Lưu và hiển thị
    plt.savefig(f'feature_importance_{strategy_name}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Hiển thị top features trong console
    print(f"\nTop {top_n} features cho {strategy_name}:")
    for i, row in top_features.iterrows():
        print(f"  {row['importance']:.4f} - {row['feature']}")
    
    return fi_df

# 8. Quy trình huấn luyện chính
print("\n--- Bắt đầu huấn luyện đa chiến lược với ELAsTiCC2 features ---")
strategies = train_multiple_strategies(X_optimal, y, groups)

print("\n--- Đánh giá chiến lược ---")
results_df, best_strategy_name = evaluate_strategies(strategies, y)

print("\n" + "="*80)
print("KẾT QUẢ TẤT CẢ CHIẾN LƯỢC")
print("="*80)

for _, row in results_df.iterrows():
    print(f"\n📊 {row['strategy']} ({row['params']}):")
    print(f"  ✅ OOF F1: {row['oof_f1']:.4f}")
    print(f"  🎯 Precision: {row['precision']:.4f}")
    print(f"  🔍 Recall: {row['recall']:.4f}")
    print(f"  🛡️  Specificity: {row['specificity']:.4f}")
    print(f"  📈 AUC: {row['auc']:.4f}")
    print(f"  ⚖️  Threshold: {row['threshold']:.3f}")
    print(f"  📊 Avg Fold F1: {row['avg_fold_f1']:.4f}")

print(f"\n🎯 CHIẾN LƯỢC TỐT NHẤT: {best_strategy_name}")
best_strategy = strategies[best_strategy_name]

# 9. Feature Importance visualization
print("\n--- Phân tích Feature Importance ---")
if best_strategy_name != 'ensemble_enhanced' and 'feature_importance' in best_strategy:
    feature_importance_df = plot_feature_importance(
        best_strategy['feature_importance'],
        X_optimal.columns,
        best_strategy_name
    )

# 10. Tối ưu mô hình cuối cùng
print("\n--- Tối ưu mô hình cuối cùng ---")

if best_strategy_name == 'ensemble_enhanced':
    print("Chiến lược tốt nhất là Ensemble, đang huấn luyện phiên bản có thể lưu...")
    
    final_models = []
    gkf = GroupKFold(n_splits=min(5, groups.nunique()))
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_optimal, y, groups)):
        print(f"Huấn luyện cuối cùng - Fold {fold + 1}")
        
        X_train, X_val = X_optimal.iloc[train_idx], X_optimal.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Huấn luyện LightGBM (chính)
        model = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.3,
            reg_lambda=0.5,
            scale_pos_weight=len(y) / (2 * np.sum(y)),
            random_state=42 + fold,
            verbosity=-1,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        final_models.append(model)
    
    best_models = final_models
else:
    best_models = best_strategy['models']

# 11. Lưu mô hình tốt nhất
print("\n--- Lưu mô hình tốt nhất ---")
best_model_info = {
    'models': best_models,
    'features': selected_features,
    'best_threshold': results_df[results_df['strategy'] == best_strategy_name]['threshold'].iloc[0],
    'oof_score': results_df[results_df['strategy'] == best_strategy_name]['oof_f1'].iloc[0],
    'precision': results_df[results_df['strategy'] == best_strategy_name]['precision'].iloc[0],
    'recall': results_df[results_df['strategy'] == best_strategy_name]['recall'].iloc[0],
    'auc': results_df[results_df['strategy'] == best_strategy_name]['auc'].iloc[0],
    'strategy': best_strategy_name,
    'feature_names': selected_features,
    'X_columns': X_optimal.columns.tolist(),
    'data_shape': X_optimal.shape,
    'class_ratio': y.mean()
}

joblib.dump(best_model_info, 'optimized_tde_model_elastricc.pkl')

# Lưu kết quả chi tiết
results_df.to_csv('training_strategies_comparison_elastricc.csv', index=False)

# Lưu dự đoán OOF
best_oof_preds = best_strategy['oof_preds']
oof_results = pd.DataFrame({
    'object_id': df['object_id'],
    'true_target': y,
    'oof_prediction': best_oof_preds,
    'oof_binary': (best_oof_preds >= best_model_info['best_threshold']).astype(int)
})
oof_results.to_csv('optimized_oof_predictions_elastricc.csv', index=False)

print("\n" + "="*80)
print("✅ HOÀN TẤT HUẤN LUYỆN VỚI ELAsTiCC2 FEATURES!")
print("="*80)
print(f"📊 Chiến lược tốt nhất: {best_strategy_name}")
print(f"🎯 OOF F1 tốt nhất: {best_model_info['oof_score']:.4f}")
print(f"📈 Precision: {best_model_info['precision']:.4f}")
print(f"🔍 Recall: {best_model_info['recall']:.4f}")
print(f"📊 AUC: {best_model_info['auc']:.4f}")
print(f"⚖️  Ngưỡng tối ưu: {best_model_info['best_threshold']:.3f}")
print(f"📁 Mô hình đã lưu: 'optimized_tde_model_elastricc.pkl'")
print(f"📁 So sánh chiến lược: 'training_strategies_comparison_elastricc.csv'")
print(f"🎯 Số lượng TDE samples: {y.sum()}/{len(y)} (tỷ lệ: {y.mean():.4f})")

# 12. Đánh giá hiệu quả ELAsTiCC2 features
print("\n--- ĐÁNH GIÁ HIỆU QUẢ ELAsTiCC2 FEATURES ---")

# Hiển thị top ELAsTiCC2 features
elastricc_features_in_selected = [f for f in selected_features if any(keyword in f for keyword in [
    'color_', 'rise_', 'fade_', 'peak_', 'autocorr_', 'p90_', 'p75_', 'p95_'
])]

print(f"\nSố lượng ELAsTiCC2 features được sử dụng: {len(elastricc_features_in_selected)}")
print("Các ELAsTiCC2 features quan trọng nhất:")

# Kiểm tra feature importance nếu có
if 'feature_importance' in best_strategy and best_strategy['feature_importance'] is not None:
    fi_df = pd.DataFrame({
        'feature': X_optimal.columns,
        'importance': best_strategy['feature_importance']
    }).sort_values('importance', ascending=False)
    
    # Lấy top ELAsTiCC2 features
    top_elastricc_features = fi_df[fi_df['feature'].isin(elastricc_features_in_selected)].head(10)
    
    for i, row in top_elastricc_features.iterrows():
        print(f"  {row['importance']:.4f} - {row['feature']}")

# 13. Dự đoán performance
print("\n--- DỰ ĐOÁN PERFORMANCE ---")
oof_f1 = best_model_info['oof_score']
if oof_f1 >= 0.45:
    print("🎉 XUẤT SẮC! Dự kiến LB score: 0.45+")
    print("   - ELAsTiCC2 features hoạt động rất tốt")
    print("   - Có thể đạt top positions trên leaderboard")
elif oof_f1 >= 0.40:
    print("👍 TỐT! Dự kiến LB score: 0.40-0.45")
    print("   - Cải thiện đáng kể so với baseline")
    print("   - Tiếp tục tuning để đạt kết quả cao hơn")
elif oof_f1 >= 0.35:
    print("⚠️  TRUNG BÌNH! Dự kiến LB score: 0.35-0.40")
    print("   - Cần xem xét lại feature engineering")
    print("   - Thử thêm features hoặc tuning parameters")
else:
    print("❌ CẦN CẢI THIỆN! Dự kiến LB score: <0.35")
    print("   - Kiểm tra lại data quality")
    print("   - Thử các chiến lược khác (XGBoost, Neural Networks)")

print("\n--- NHẬN ĐỊNH THEN CHỐT ---")
print("1. ELAsTiCC2 features ĐÃ ĐƯỢC TÍCH HỢP THÀNH CÔNG")
print("2. Model đã được optimized cho imbalanced data")
print("3. Ready for test set prediction với predict.py")

print("\n📋 NEXT STEPS:")
print("1. Chạy predict.py với model mới")
print("2. Submit kết quả lên Kaggle")
print("3. So sánh với baseline performance")
print("4. Nếu cần, thử hyperparameter tuning thêm")