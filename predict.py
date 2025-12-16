import pandas as pd
import numpy as np
import extinction
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
import joblib
import os
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.stats import theilslopes
import warnings
warnings.filterwarnings('ignore')

print("=== Dự đoán tập test TDE ===")

# Cấu hình tham số
BASE_PATH = './data/'
EFF_WAVELENGTHS = {'u': 3641, 'g': 4704, 'r': 6155, 'i': 7504, 'z': 8695, 'y': 10056}
R_V = 3.1
FILTERS = ['u', 'g', 'r', 'i', 'z', 'y']

# 1. Tải mô hình đã huấn luyện
print("--- Đang tải mô hình ---")
model_info = joblib.load('optimized_tde_model.pkl')
selected_features = model_info['features']
best_threshold = model_info['best_threshold']
models = model_info['models']

print(f"Loại mô hình: {model_info['strategy']}")
print(f"Số lượng đặc trưng: {len(selected_features)}")
print(f"Ngưỡng tối ưu: {best_threshold:.3f}")
print(f"Số mô hình: {len(models)}")

# 2. Tải metadata của tập test
print("\n--- Đang tải dữ liệu tập test ---")
test_log = pd.read_csv(f'{BASE_PATH}test_log.csv')
print(f"Số lượng đối tượng trong tập test: {len(test_log)}")

# 3. Hàm feature engineering cho tập test (giữ nhất quán với tập train)
def tde_decay_model(t, alpha, t0, A):
    return A * (t - t0 + 1e-6) ** (-alpha)

def fit_decay_alpha(group):
    group = group.sort_values('mjd')
    t = group['mjd'].values
    f = group['flux_corrected'].values
    if len(f) < 3: return np.nan
    peak_idx = np.argmax(f)
    if peak_idx == len(f) - 1: return np.nan
    t_decay = t[peak_idx:] - t[peak_idx]
    f_decay = f[peak_idx:]
    try:
        popt, _ = curve_fit(
            tde_decay_model, t_decay, f_decay,
            p0=[1.5, 0, f[peak_idx]],
            bounds=([0.5, -10, 0], [3.0, 10, np.inf]),
            maxfev=500
        )
        alpha = popt[0]
        return alpha if 0.5 <= alpha <= 3.0 else np.nan
    except:
        return np.nan

def is_single_peak(group):
    f = group['flux_corrected'].values
    if len(f) < 3: return 0
    peaks = (f[1:-1] > f[:-2]) & (f[1:-1] > f[2:])
    return int(np.sum(peaks) == 1)

def rise_fall_ratio(group):
    group = group.sort_values('mjd')
    t = group['mjd'].values
    f = group['flux_corrected'].values
    if len(f) < 3: return np.nan
    peak_idx = np.argmax(f)
    if peak_idx == 0 or peak_idx == len(f) - 1: return np.nan
    rise_time = t[peak_idx] - t[0]
    fall_time = t[-1] - t[peak_idx]
    return rise_time / (fall_time + 1e-6)

def extract_improved_features(df):
    """Trích xuất các đặc trưng cải tiến và ổn định (giống tập train)"""
    features_list = []
    
    for object_id in df['object_id'].unique():
        obj_data = df[df['object_id'] == object_id].sort_values('mjd')
        
        features = {}
        flux = obj_data['flux_corrected'].values
        mjd = obj_data['mjd'].values
        
        # 1. Xử lý flux âm
        flux_positive = flux.clip(min=1e-9)
        flux_abs = np.abs(flux)
        
        # Đặc trưng theo phân vị
        for q in [10, 25, 50, 75, 90]:
            features[f'flux_q{q}'] = np.percentile(flux_positive, q)
            features[f'flux_abs_q{q}'] = np.percentile(flux_abs, q)
        
        features['positive_ratio'] = (flux > 0).mean()
        features['flux_abs_median'] = np.median(flux_abs)
        
        # 2. Đặc trưng khoảng cách thời gian quan sát
        if len(mjd) > 1:
            mjd_diff = np.diff(mjd)
            features['gap_mean'] = np.mean(mjd_diff)
            features['gap_std'] = np.std(mjd_diff)
            features['gap_max'] = np.max(mjd_diff)
            features['gap_median'] = np.median(mjd_diff)
            features['obs_density'] = len(obj_data) / (mjd[-1] - mjd[0] + 1e-9)
        
        # 3. Đặc trưng theo dải sóng
        unique_filters = obj_data['Filter'].nunique()
        features['filter_coverage'] = unique_filters / 6.0
        
        # Tỷ lệ số quan sát theo từng dải
        for band in ['u', 'g', 'r', 'i', 'z', 'y']:
            band_count = (obj_data['Filter'] == band).sum()
            features[f'{band}_obs_ratio'] = band_count / len(obj_data)
        
        # 4. Đặc trưng tín hiệu trên nhiễu
        snr = flux_positive / (obj_data['flux_err'].values + 1e-9)
        features['snr_median'] = np.median(snr)
        features['snr_q10'] = np.percentile(snr, 10)
        features['snr_q90'] = np.percentile(snr, 90)
        
        # 5. Độ ổn định xu hướng (Theil-Sen)
        if len(flux) >= 5:
            try:
                x = np.arange(len(flux))
                slope, _, _, _ = theilslopes(flux, x)
                features['trend_slope'] = slope
                # Độ ổn định xu hướng: độ lớn tương đối của phần dư
                predicted = slope * x + np.median(flux)
                residuals = flux - predicted
                features['trend_stability'] = 1.0 / (np.std(residuals) / (np.std(flux) + 1e-9) + 1e-9)
            except:
                features['trend_slope'] = 0
                features['trend_stability'] = 0
        else:
            features['trend_slope'] = 0
            features['trend_stability'] = 0
        
        # 6. Đặc trưng hình thái chuỗi thời gian tổng quát (không phụ thuộc nhãn)
        if len(flux) >= 3:
            # Chỉ số biến thiên
            features['variability_index'] = np.std(flux) / (np.mean(flux_positive) + 1e-9)
            
            # Độ sắc đỉnh
            features['peakiness'] = np.max(flux_positive) / (np.median(flux_positive) + 1e-9)
            
            # Độ bất đối xứng
            peak_idx = np.argmax(flux)
            if 0 < peak_idx < len(flux) - 1:
                before_peak = flux[:peak_idx]
                after_peak = flux[peak_idx+1:]
                std_before = np.std(before_peak) if len(before_peak) > 1 else 0
                std_after = np.std(after_peak) if len(after_peak) > 1 else 0
                features['asymmetry'] = std_after / (std_before + 1e-9)
            else:
                features['asymmetry'] = 0
            
            # Tỷ lệ tăng/giảm toàn cục
            if len(flux) > 5:
                first_half = flux[:len(flux)//2]
                second_half = flux[len(flux)//2:]
                features['rise_fall_ratio_global'] = np.mean(first_half) / (np.mean(second_half) + 1e-9)
        
        features['object_id'] = object_id
        features_list.append(features)
    
    return pd.DataFrame(features_list)

# 4. Xử lý light curve của tập test
def process_test_lightcurves():
    """Xử lý light curve tập test và trích xuất đặc trưng"""
    
    print("\n--- Đang xử lý light curve tập test ---")
    
    # Kiểm tra cấu trúc file tập test
    test_lc_paths = []
    
    # Kiểm tra file test đơn
    single_test_path = f'{BASE_PATH}test_lightcurves/test_full_lightcurves.csv'
    if os.path.exists(single_test_path):
        test_lc_paths = [single_test_path]
        print("Đã tìm thấy file test đơn")
    else:
        # Kiểm tra các file test chia nhỏ
        for i in range(1, 21):
            split_path = f'{BASE_PATH}split_{i:02d}/test_full_lightcurves.csv'
            if os.path.exists(split_path):
                test_lc_paths.append(split_path)
        print(f"Tìm thấy {len(test_lc_paths)} file test phân mảnh")
    
    if not test_lc_paths:
        raise FileNotFoundError("Không tìm thấy file light curve của tập test")
    
    all_test_features = []
    
    for lc_path in tqdm(test_lc_paths, desc="Xử lý các phân mảnh test"):
        # Tải dữ liệu light curve
        lc = pd.read_csv(lc_path)
        lc.rename(columns={'Time (MJD)': 'mjd', 'Flux': 'flux', 'Flux_err': 'flux_err'}, inplace=True)
        
        # Chỉ xử lý object_id tồn tại trong test_log
        relevant_objects = test_log['object_id'].unique()
        lc = lc[lc['object_id'].isin(relevant_objects)]
        
        if lc.empty:
            continue
        
        processed_dfs = []
        
        # Hiệu chỉnh flux cho từng object
        for object_id in lc['object_id'].unique():
            object_lc = lc[lc['object_id'] == object_id].copy()
            if object_lc.empty:
                continue
                
            # Lấy giá trị EBV của object
            object_ebv = test_log[test_log['object_id'] == object_id]['EBV'].iloc[0]
            A_v = R_V * object_ebv
            
            flux_corrected_list = []
            for _, row in object_lc.iterrows():
                A_lambda = extinction.fitzpatrick99(
                    np.array([EFF_WAVELENGTHS[row['Filter']]]), A_v, R_V
                )[0]
                flux_corrected_list.append(row['flux'] * 10**(0.4 * A_lambda))
            
            object_lc['flux_corrected'] = flux_corrected_list
            processed_dfs.append(object_lc)
        
        if not processed_dfs:
            continue
            
        df = pd.concat(processed_dfs)
        
        # Tính khoảng cách và độ sáng tuyệt đối (dùng Z của tập test, có thể có sai số)
        df['Z'] = df['object_id'].map(test_log.set_index('object_id')['Z'])
        z_values = df['Z'].fillna(0).values
        z_values[z_values <= 0] = 1e-6
        
        # Tính khoảng cách
        dist_pc = cosmo.luminosity_distance(z_values).to(u.pc).value
        df['distance_pc'] = dist_pc
        
        # Tính độ sáng tuyệt đối
        df['flux_positive'] = df['flux_corrected'].clip(lower=1e-9)
        df['apparent_mag'] = -2.5 * np.log10(df['flux_positive'])
        df['absolute_mag'] = df['apparent_mag'] - 5 * (np.log10(df['distance_pc']) - 1)
        
        # Tính toán cơ bản
        df['snr'] = df['flux_corrected'] / df['flux_err']
        df = df.sort_values(['object_id', 'mjd'])
        
        # Feature engineering
        grouped = df.groupby('object_id')
        
        # Đặc trưng cơ bản
        agg_features = grouped.agg(
            flux_mean=('flux_corrected', 'mean'),
            flux_std=('flux_corrected', 'std'),
            flux_max=('flux_corrected', 'max'),
            flux_min=('flux_corrected', 'min'),
            flux_skew=('flux_corrected', 'skew'),
            mjd_span=('mjd', lambda x: x.max() - x.min()),
            snr_mean=('snr', 'mean'),
            snr_max=('snr', 'max'),
            snr_std=('snr', 'std'),
            obs_count=('mjd', 'size')
        )
        
        # Đặc trưng độ sáng tuyệt đối
        abs_mag_agg = grouped.agg(
            abs_mag_min=('absolute_mag', 'min'),
            abs_mag_max=('absolute_mag', 'max'),
            abs_mag_mean=('absolute_mag', 'mean'),
            abs_mag_std=('absolute_mag', 'std'),
            abs_mag_span=('absolute_mag', lambda x: x.max() - x.min())
        )
        agg_features = agg_features.join(abs_mag_agg, how='left')
        
        # Độ dốc có trọng số theo thời gian
        def weighted_slope(group):
            group = group.sort_values('mjd')
            delta_mjd = np.diff(group['mjd'])
            delta_flux = np.diff(group['flux_corrected'])
            slopes = delta_flux / (delta_mjd + 1e-9)
            return np.mean(slopes) if len(slopes) > 0 else 0
        
        agg_features['weighted_slope_mean'] = grouped.apply(weighted_slope)
        
        # Đặc trưng vật lý
        agg_features['decay_alpha'] = grouped.apply(fit_decay_alpha)
        agg_features['is_single_peak'] = grouped.apply(is_single_peak)
        agg_features['rise_fall_ratio'] = grouped.apply(rise_fall_ratio)
        
        # Đặc trưng theo dải
        pivot_mean = df.pivot_table(index='object_id', columns='Filter', values='flux_corrected', aggfunc='mean').add_suffix('_mean')
        pivot_std = df.pivot_table(index='object_id', columns='Filter', values='flux_corrected', aggfunc='std').add_suffix('_std')
        pivot_max = df.pivot_table(index='object_id', columns='Filter', values='flux_corrected', aggfunc='max').add_suffix('_max')
        agg_features = agg_features.join([pivot_mean, pivot_std, pivot_max], how='left')
        
        # Đặc trưng màu sắc
        for f in FILTERS:
            if f'{f}_mean' not in agg_features.columns: 
                agg_features[f'{f}_mean'] = np.nan
        agg_features['color_g_r_mean'] = agg_features['g_mean'] - agg_features['r_mean']
        agg_features['color_r_i_mean'] = agg_features['r_mean'] - agg_features['i_mean']
        agg_features['color_i_z_mean'] = agg_features['i_mean'] - agg_features['z_mean']
        
        # Thêm các đặc trưng cải tiến
        improved_features = extract_improved_features(df)
        agg_features = agg_features.merge(improved_features, on='object_id', how='left')
        
        all_test_features.append(agg_features)
    
    if not all_test_features:
        raise ValueError("Không tạo được đặc trưng cho tập test")
    
    # Gộp tất cả đặc trưng
    test_feature_df = pd.concat(all_test_features)
    test_feature_df.reset_index(inplace=True)
    
    return test_feature_df

# 5. Sinh đặc trưng cho tập test
test_features_df = process_test_lightcurves()

# 6. Gộp metadata và đặc trưng
print("\n--- Gộp dữ liệu tập test ---")
test_final = test_log.merge(test_features_df, on='object_id', how='left')

# Kiểm tra đặc trưng bị thiếu
missing_features = set(selected_features) - set(test_final.columns)
if missing_features:
    print(f"Cảnh báo: thiếu {len(missing_features)} đặc trưng, sẽ điền 0")
    for feature in missing_features:
        test_final[feature] = 0

# 7. Chuẩn bị dữ liệu dự đoán
X_test = test_final[selected_features].copy()

# Xử lý giá trị thiếu (theo chiến lược của tập train)
numeric_cols = X_test.select_dtypes(include=[np.number]).columns
X_test[numeric_cols] = X_test[numeric_cols].fillna(0)

std_cols = [c for c in X_test.columns if 'std' in c]
X_test[std_cols] = X_test[std_cols].fillna(0)

print(f"Kích thước đặc trưng tập test: {X_test.shape}")

# 8. Tiến hành dự đoán
print("\n--- Đang dự đoán ---")
test_predictions = []

for i, model in enumerate(models):
    pred = model.predict_proba(X_test)[:, 1]
    test_predictions.append(pred)
    print(f"Mô hình {i+1} đã dự đoán xong")

# Trung bình dự đoán
test_preds_mean = np.mean(test_predictions, axis=0)

# Áp dụng ngưỡng tối ưu
test_binary = (test_preds_mean >= best_threshold).astype(int)

print(f"Dự đoán tập test hoàn tất:")
print(f"Xác suất dự đoán trung bình: {test_preds_mean.mean():.4f} ± {test_preds_mean.std():.4f}")
print(f"Tỷ lệ mẫu dương được dự đoán: {test_binary.mean():.4f}")
print(f"Phân bố xác suất dự đoán:")
for percentile in [10, 25, 50, 75, 90]:
    value = np.percentile(test_preds_mean, percentile)
    print(f"  Phân vị {percentile}%: {value:.4f}")

# 9. Tạo file submission
print("\n--- Tạo file submission ---")
submission = pd.DataFrame({
    'object_id': test_final['object_id'],
    'predicted': test_binary
})

# Đảm bảo mọi object trong tập test đều có dự đoán
missing_objects = set(test_log['object_id']) - set(submission['object_id'])
if missing_objects:
    print(f"Cảnh báo: {len(missing_objects)} object chưa có dự đoán, gán 0")
    missing_df = pd.DataFrame({
        'object_id': list(missing_objects),
        'predicted': 0
    })
    submission = pd.concat([submission, missing_df], ignore_index=True)

# Sắp xếp theo object_id
submission = submission.sort_values('object_id')

# Lưu file submission
submission_file = 'submission.csv'
submission.to_csv(submission_file, index=False)

print(f"✅ Đã tạo file submission: '{submission_file}'")
print(f"Kích thước file submission: {submission.shape}")
print(f"Số mẫu dương được dự đoán: {submission['predicted'].sum()}")
print(f"Tỷ lệ mẫu dương: {submission['predicted'].mean():.4f}")

# 10. Lưu kết quả dự đoán chi tiết
detailed_predictions = pd.DataFrame({
    'object_id': test_final['object_id'],
    'prediction_prob': test_preds_mean,
    'predicted': test_binary
})
detailed_predictions.to_csv('test_detailed_predictions.csv', index=False)
print("Đã lưu kết quả dự đoán chi tiết: 'test_detailed_predictions.csv'")

# 11. Phân tích dự đoán
print("\n--- Phân tích dự đoán ---")
print(f"Ngưỡng sử dụng: {best_threshold}")
print(f"Khoảng xác suất dự đoán: [{test_preds_mean.min():.4f}, {test_preds_mean.max():.4f}]")
print(f"Mẫu dương độ tin cậy cao (p > 0.7): {(test_preds_mean > 0.7).sum()}")
print(f"Mẫu âm độ tin cậy cao (p < 0.3): {(test_preds_mean < 0.3).sum()}")

# Kiểm tra độ nhất quán của độ quan trọng đặc trưng
if hasattr(models[0], 'feature_importances_'):
    feature_importance = np.mean([model.feature_importances_ for model in models], axis=0)
    importance_df = pd.DataFrame({
        'feature': selected_features,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 đặc trưng quan trọng nhất trên tập test:")
    print(importance_df.head(10))

print("\n🎉 Hoàn tất dự đoán tập test!")
print("📤 Hãy nộp file 'submission.csv' lên Kaggle")
print(f"📊 Điểm LB kỳ vọng xấp xỉ: {model_info['oof_score']:.4f} (dựa trên OOF F1)")
print("💡 Nếu điểm LB không tốt, hãy kiểm tra sự khác biệt phân phối giữa train và test")
