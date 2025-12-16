import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=== Tinh chỉnh chi tiết mô hình phân loại TDE ===")

# 1. Tải dữ liệu
print("--- Đang tải dữ liệu ---")
df = pd.read_csv('processed_train_features_improved.csv')
features_to_exclude = ['object_id', 'SpecType', 'English Translation', 'split', 'target']
all_features = [col for col in df.columns if col not in features_to_exclude]
X = df[all_features]
y = df['target']
groups = df['split']

print(f"Kích thước dữ liệu: {X.shape}")
print(f"Tỷ lệ mẫu dương: {y.mean():.4f}")

# 2. Chọn tập con đặc trưng tối ưu dựa trên kết quả trước đó
def select_optimal_features(df):
    """Chọn đặc trưng tối ưu dựa trên các thí nghiệm trước"""
    
    # Các đặc trưng có mức độ quan trọng cao trong các kết quả trước
    high_importance_features = [
        'flux_abs_q25', 'flux_abs_q10', 'decay_alpha', 'r_max', 'mjd_span',
        'Z', 'flux_abs_q50', 'u_mean', 'i_max', 'u_max', 'peakiness',
        'color_r_i_mean', 'g_max', 'abs_mag_mean', 'abs_mag_min',
        'abs_mag_max', 'rise_fall_ratio', 'rise_fall_ratio_global',
        'asymmetry', 'u_std', 'positive_ratio', 'flux_abs_median',
        'variability_index', 'trend_stability', 'filter_coverage',
        'gap_max', 'obs_density', 'snr_mean', 'obs_count', 'flux_mean'
    ]
    
    # Chỉ chọn các đặc trưng tồn tại trong dữ liệu
    selected = [f for f in high_importance_features if f in df.columns]
    print(f"Đã chọn {len(selected)} đặc trưng quan trọng cao")
    return selected

print("\n--- Lựa chọn đặc trưng ---")
selected_features = select_optimal_features(df)
X_optimal = df[selected_features]

# 3. Huấn luyện mô hình với nhiều chiến lược
def train_multiple_strategies(X, y, groups):
    """Thử nghiệm nhiều chiến lược huấn luyện"""
    
    strategies = {}
    pos_weight = len(y) / (2 * np.sum(y))
    
    gkf = GroupKFold(n_splits=min(5, groups.nunique()))
    
    # Chiến lược 1: Chính quy hóa mức trung bình
    print("\n--- Chiến lược 1: Chính quy hóa trung bình ---")
    models1, oof1, scores1 = train_strategy(X, y, groups, {
        'n_estimators': 1500,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 7,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.3,
        'reg_lambda': 0.5,
        'scale_pos_weight': pos_weight
    }, "Chính quy hóa trung bình")
    
    strategies['medium_reg'] = {
        'models': models1, 
        'oof_preds': oof1, 
        'scores': scores1,
        'params': 'Chính quy hóa trung bình'
    }
    
    # Chiến lược 2: Chính quy hóa nhẹ + dừng sớm nghiêm ngặt
    print("\n--- Chiến lược 2: Chính quy hóa nhẹ + dừng sớm nghiêm ngặt ---")
    models2, oof2, scores2 = train_strategy(X, y, groups, {
        'n_estimators': 1000,
        'learning_rate': 0.1,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 10,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'reg_alpha': 0.1,
        'reg_lambda': 0.2,
        'scale_pos_weight': pos_weight
    }, "Chính quy hóa nhẹ", early_stopping_rounds=50)
    
    strategies['light_reg'] = {
        'models': models2, 
        'oof_preds': oof2, 
        'scores': scores2,
        'params': 'Chính quy hóa nhẹ'
    }
    
    # Chiến lược 3: Học tập tổ hợp (LightGBM + RandomForest)
    print("\n--- Chiến lược 3: Tổ hợp mô hình ---")
    ensemble_preds = np.zeros(len(X))
    ensemble_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"Huấn luyện tổ hợp - Fold {fold + 1}")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            random_state=42 + fold,
            verbosity=-1
        )
        lgb_model.fit(X_train, y_train)
        lgb_preds = lgb_model.predict_proba(X_val)[:, 1]
        
        # RandomForest
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42 + fold,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        rf_preds = rf_model.predict_proba(X_val)[:, 1]
        
        # Dự đoán trung bình
        ensemble_pred = (lgb_preds + rf_preds) / 2
        ensemble_preds[val_idx] = ensemble_pred
        
        # Tính F1 cho fold hiện tại
        best_f1_fold = 0
        for thresh in np.arange(0.1, 0.9, 0.02):
            f1 = f1_score(y_val, (ensemble_pred >= thresh).astype(int))
            if f1 > best_f1_fold:
                best_f1_fold = f1
        
        ensemble_scores.append(best_f1_fold)
        print(f"Fold {fold + 1} F1: {best_f1_fold:.4f}")
    
    strategies['ensemble'] = {
        'models': None,  # Mô hình tổ hợp không lưu từng mô hình riêng lẻ
        'oof_preds': ensemble_preds, 
        'scores': ensemble_scores,
        'params': 'Tổ hợp LightGBM + RF'
    }
    
    return strategies

def train_strategy(X, y, groups, params, strategy_name, early_stopping_rounds=100):
    """Huấn luyện một chiến lược đơn lẻ"""
    
    gkf = GroupKFold(n_splits=min(5, groups.nunique()))
    models = []
    oof_preds = np.zeros(len(X))
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"{strategy_name} - Fold {fold + 1}")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params, verbosity=-1, n_jobs=-1)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='binary_logloss',
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )
        
        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        
        # Tìm ngưỡng tối ưu
        best_f1_fold = 0
        for thresh in np.arange(0.1, 0.9, 0.02):
            f1 = f1_score(y_val, (val_preds >= thresh).astype(int))
            if f1 > best_f1_fold:
                best_f1_fold = f1
        
        fold_scores.append(best_f1_fold)
        models.append(model)
        print(f"Fold {fold + 1} F1: {best_f1_fold:.4f}")
    
    return models, oof_preds, fold_scores

# 4. Đánh giá tất cả chiến lược
def evaluate_strategies(strategies, y):
    """Đánh giá toàn bộ chiến lược huấn luyện"""
    
    results = []
    best_strategy = None
    best_f1 = 0
    
    for name, strategy in strategies.items():
        oof_preds = strategy['oof_preds']
        
        # Tối ưu ngưỡng
        best_threshold = 0.5
        best_f1_score = 0
        for thresh in np.arange(0.1, 0.9, 0.01):
            f1 = f1_score(y, (oof_preds >= thresh).astype(int))
            if f1 > best_f1_score:
                best_f1_score = f1
                best_threshold = thresh
        
        # Tính các chỉ số khác
        binary_preds = (oof_preds >= best_threshold).astype(int)
        precision = np.mean(y[binary_preds == 1]) if np.sum(binary_preds) > 0 else 0
        recall = np.mean(binary_preds[y == 1])
        
        # Phân tích quá khớp (chỉ áp dụng cho chiến lược mô hình đơn)
        if strategy['models'] is not None:
            train_f1 = analyze_overfitting(strategy['models'], X_optimal, y, best_threshold)
            overfitting_gap = train_f1 - best_f1_score
        else:
            train_f1 = np.nan
            overfitting_gap = np.nan
        
        results.append({
            'strategy': name,
            'params': strategy['params'],
            'oof_f1': best_f1_score,
            'precision': precision,
            'recall': recall,
            'threshold': best_threshold,
            'train_f1': train_f1,
            'overfitting_gap': overfitting_gap,
            'fold_scores': strategy['scores']
        })
        
        if best_f1_score > best_f1:
            best_f1 = best_f1_score
            best_strategy = name
    
    return pd.DataFrame(results), best_strategy

def analyze_overfitting(models, X, y, threshold):
    """Phân tích hiện tượng quá khớp"""
    train_preds = []
    for model in models:
        train_pred = model.predict_proba(X)[:, 1]
        train_preds.append(train_pred)
    
    train_preds_mean = np.mean(train_preds, axis=0)
    return f1_score(y, (train_preds_mean >= threshold).astype(int))

# 5. Quy trình huấn luyện chính
print("\n--- Bắt đầu huấn luyện đa chiến lược ---")
strategies = train_multiple_strategies(X_optimal, y, groups)

print("\n--- Đánh giá chiến lược ---")
results_df, best_strategy_name = evaluate_strategies(strategies, y)

print("\n=== Kết quả tất cả chiến lược ===")
for _, row in results_df.iterrows():
    print(f"\n{row['strategy']} ({row['params']}):")
    print(f"  OOF F1: {row['oof_f1']:.4f}")
    print(f"  Độ chính xác: {row['precision']:.4f}")
    print(f"  Độ bao phủ (Recall): {row['recall']:.4f}")
    print(f"  Ngưỡng: {row['threshold']:.3f}")
    if not pd.isna(row['overfitting_gap']):
        print(f"  Khoảng cách quá khớp: {row['overfitting_gap']:.4f}")
    print(f"  F1 từng Fold: {[f'{s:.4f}' for s in row['fold_scores']]}")

print(f"\n🎯 Chiến lược tốt nhất: {best_strategy_name}")
best_strategy = strategies[best_strategy_name]

# 6. Tối ưu mô hình cuối cùng
print("\n--- Tối ưu mô hình cuối cùng ---")

# Nếu chiến lược tốt nhất là tổ hợp, cần huấn luyện phiên bản có thể lưu
if best_strategy_name == 'ensemble':
    print("Chiến lược tốt nhất là tổ hợp, đang huấn luyện phiên bản có thể lưu...")
    
    final_models = []
    gkf = GroupKFold(n_splits=min(5, groups.nunique()))
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_optimal, y, groups)):
        print(f"Huấn luyện cuối cùng - Fold {fold + 1}")
        
        X_train, X_val = X_optimal.iloc[train_idx], X_optimal.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate= 0.05 if best_strategy_name == 'medium_reg' else 0.1,
            num_leaves= 31 if best_strategy_name == 'medium_reg' else 63,
            max_depth= 7 if best_strategy_name == 'medium_reg' else 8,
            min_child_samples= 20 if best_strategy_name == 'medium_reg' else 10,
            subsample= 0.8 if best_strategy_name == 'medium_reg' else 0.9,
            colsample_bytree= 0.8 if best_strategy_name == 'medium_reg' else 0.9,
            reg_alpha= 0.3 if best_strategy_name == 'medium_reg' else 0.1,
            reg_lambda= 0.5 if best_strategy_name == 'medium_reg' else 0.2,
            scale_pos_weight= len(y) / (2 * np.sum(y)),
            random_state=42 + fold,
            verbosity=-1,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        final_models.append(model)
    
    best_models = final_models
else:
    best_models = best_strategy['models']

# 7. Lưu mô hình tốt nhất
print("\n--- Lưu mô hình tốt nhất ---")
best_model_info = {
    'models': best_models,
    'features': selected_features,
    'best_threshold': results_df[results_df['strategy'] == best_strategy_name]['threshold'].iloc[0],
    'oof_score': results_df[results_df['strategy'] == best_strategy_name]['oof_f1'].iloc[0],
    'strategy': best_strategy_name,
    'feature_names': selected_features
}

joblib.dump(best_model_info, 'optimized_tde_model.pkl')

# Lưu kết quả chi tiết
results_df.to_csv('training_strategies_comparison.csv', index=False)

# Lưu dự đoán OOF
best_oof_preds = best_strategy['oof_preds']
oof_results = pd.DataFrame({
    'object_id': df['object_id'],
    'true_target': y,
    'oof_prediction': best_oof_preds,
    'oof_binary': (best_oof_preds >= best_model_info['best_threshold']).astype(int)
})
oof_results.to_csv('optimized_oof_predictions.csv', index=False)

print("✅ Hoàn tất tinh chỉnh!")
print(f"📊 Chiến lược tốt nhất: {best_strategy_name}")
print(f"🎯 OOF F1 tốt nhất: {best_model_info['oof_score']:.4f}")
print(f"⚖️  Ngưỡng tối ưu: {best_model_info['best_threshold']:.3f}")
print(f"📁 Mô hình đã lưu: 'optimized_tde_model.pkl'")
print(f"📁 So sánh chiến lược đã lưu: 'training_strategies_comparison.csv'")

# 8. Nhận định then chốt
print("\n--- Nhận định then chốt ---")
print("Phân tích vấn đề hiện tại:")
print("1. Quá khớp nghiêm trọng - cần cân bằng giữa chính quy hóa và độ phức tạp mô hình")
print("2. Mất cân bằng dữ liệu - chỉ có 148 mẫu dương, cần chiến lược lấy mẫu tốt hơn")
print("3. Chất lượng đặc trưng - cần đảm bảo đặc trưng có khả năng phân biệt")

print("\nĐề xuất bước tiếp theo:")
if best_model_info['oof_score'] < 0.38:
    print("⚠️  OOF F1 vẫn còn thấp, đề xuất:")
    print("   - Kiểm tra lại quy trình tạo đặc trưng")
    print("   - Thử kỹ thuật oversampling (SMOTE)")
    print("   - Cân nhắc sử dụng mạng nơ-ron")
else:
    print("✅ Kết quả chấp nhận được, có thể tiến hành dự đoán trên tập test")
