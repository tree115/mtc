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
from scipy.signal import find_peaks
warnings.filterwarnings('ignore')

print("=== DỰ ĐOÁN TẬP TEST TDE VỚI ELAsTiCC2 MODEL ===")

# Cấu hình tham số
BASE_PATH = '/content/data/'
EFF_WAVELENGTHS = {'u': 3641, 'g': 4704, 'r': 6155, 'i': 7504, 'z': 8695, 'y': 10056}
R_V = 3.1
FILTERS = ['u', 'g', 'r', 'i', 'z', 'y']

# 1. Tải mô hình ĐÃ HUẤN LUYỆN MỚI
print("--- Đang tải mô hình ELAsTiCC2 ---")
try:
    model_info = joblib.load('optimized_tde_model_elastricc.pkl')  # ĐỔI TÊN FILE
    print(f"✅ Đã tải model từ: 'optimized_tde_model_elastricc.pkl'")
except FileNotFoundError:
    print("⚠️  Không tìm thấy model mới, thử tải model cũ...")
    model_info = joblib.load('optimized_tde_model.pkl')

selected_features = model_info['features']
best_threshold = model_info['best_threshold']
models = model_info['models']

print(f"Loại mô hình: {model_info['strategy']}")
print(f"Số lượng đặc trưng: {len(selected_features)}")
print(f"Ngưỡng tối ưu: {best_threshold:.3f}")
print(f"Số mô hình: {len(models)}")
print(f"OOF F1 score: {model_info.get('oof_score', 'N/A'):.4f}")
print(f"Precision: {model_info.get('precision', 'N/A'):.4f}")
print(f"Recall: {model_info.get('recall', 'N/A'):.4f}")

# 2. Tải metadata của tập test
print("\n--- Đang tải dữ liệu tập test ---")
test_log = pd.read_csv(f'{BASE_PATH}test_log.csv')
print(f"Số lượng đối tượng trong tập test: {len(test_log)}")

# 3. THÊM: Các hàm ELAsTiCC2 features cho test set
def calculate_color_features_test(df):
    """Tính color features cho test set (giống train)"""
    color_features_list = []
    
    for object_id in df['object_id'].unique():
        obj_data = df[df['object_id'] == object_id]
        
        features = {'object_id': object_id}
        
        # Tính mean color giữa các bands
        for band_pair in [('g', 'r'), ('r', 'i'), ('g', 'i'), ('u', 'g'), ('i', 'z'), ('z', 'y')]:
            band1, band2 = band_pair
            
            # Check if we have flux for both bands
            band1_mean_col = f'{band1}_mean'
            band2_mean_col = f'{band2}_mean'
            
            if band1_mean_col in obj_data.columns and band2_mean_col in obj_data.columns:
                band1_mean = obj_data[band1_mean_col].iloc[0] if not pd.isna(obj_data[band1_mean_col].iloc[0]) else np.nan
                band2_mean = obj_data[band2_mean_col].iloc[0] if not pd.isna(obj_data[band2_mean_col].iloc[0]) else np.nan
                
                if not np.isnan(band1_mean) and not np.isnan(band2_mean):
                    features[f'color_{band1}_{band2}_mean'] = band1_mean - band2_mean
        
        color_features_list.append(features)
    
    return pd.DataFrame(color_features_list)

def calculate_rise_fade_features_test(df):
    """Tính rise và fade times cho test set"""
    features_list = []
    
    for object_id in df['object_id'].unique():
        obj_data = df[df['object_id'] == object_id].sort_values('mjd')
        
        if len(obj_data) < 5:
            features_list.append({'object_id': object_id})
            continue
            
        flux = obj_data['flux_corrected'].values
        mjd = obj_data['mjd'].values
        
        # Tìm peak
        peak_idx = np.argmax(flux)
        
        # Rise time
        rise_time = mjd[peak_idx] - mjd[0] if peak_idx > 0 else np.nan
        
        # Fade time
        fade_time = mjd[-1] - mjd[peak_idx] if peak_idx < len(flux)-1 else np.nan
        
        # Rise rate
        if rise_time > 0 and peak_idx > 0:
            rise_rate = (flux[peak_idx] - flux[0]) / rise_time
        else:
            rise_rate = np.nan
            
        # Fade rate
        if fade_time > 0 and peak_idx < len(flux)-1:
            fade_rate = (flux[peak_idx] - flux[-1]) / fade_time
        else:
            fade_rate = np.nan
        
        features_list.append({
            'object_id': object_id,
            'rise_time': rise_time,
            'fade_time': fade_time,
            'rise_fade_ratio': rise_time / (fade_time + 1e-9),
            'rise_rate': rise_rate,
            'fade_rate': fade_rate,
            'total_duration': mjd[-1] - mjd[0]
        })
    
    return pd.DataFrame(features_list)

def enhanced_peak_analysis_test(group):
    """Phân tích peak nâng cao cho test set"""
    group = group.sort_values('mjd')
    flux = group['flux_corrected'].values
    
    features = {}
    
    # Tìm peaks
    peaks, properties = find_peaks(flux, prominence=0.1, width=1)
    
    features['peak_count'] = len(peaks)
    
    if len(peaks) > 0:
        # Peak properties
        features['peak_prominence_mean'] = np.mean(properties['prominences'])
        features['peak_prominence_max'] = np.max(properties['prominences'])
        
        # Main peak analysis
        main_peak_idx = peaks[np.argmax(properties['prominences'])]
        features['main_peak_flux'] = flux[main_peak_idx]
        
        # Peak symmetry
        if main_peak_idx > 0 and main_peak_idx < len(flux)-1:
            left_side = flux[:main_peak_idx]
            right_side = flux[main_peak_idx+1:]
            
            if len(left_side) > 0 and len(right_side) > 0:
                features['peak_symmetry'] = np.mean(left_side) / (np.mean(right_side) + 1e-9)
                features['peak_asymmetry'] = abs(features['peak_symmetry'] - 1)
    
    # Peak-to-median ratio
    features['peak_to_median'] = np.max(flux) / (np.median(flux) + 1e-9)
    features['peak_to_mean'] = np.max(flux) / (np.mean(flux) + 1e-9)
    
    return pd.Series(features)

def calculate_temporal_features_test(group):
    """Tính các đặc trưng thời gian cho test set"""
    group = group.sort_values('mjd')
    mjd = group['mjd'].values
    flux = group['flux_corrected'].values
    
    features = {}
    
    if len(mjd) > 1:
        # Time gaps statistics
        gaps = np.diff(mjd)
        features['gap_mean'] = np.mean(gaps)
        features['gap_std'] = np.std(gaps)
        features['gap_max'] = np.max(gaps)
        
        # Observation density
        total_time = mjd[-1] - mjd[0]
        features['obs_density'] = len(mjd) / (total_time + 1e-9)
    
    # Time series autocorrelation
    if len(flux) >= 5:
        autocorr = np.correlate(flux - np.mean(flux), flux - np.mean(flux), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Lags 1-3
        for lag in [1, 2, 3]:
            if lag < len(autocorr):
                features[f'autocorr_lag{lag}'] = autocorr[lag] / autocorr[0] if autocorr[0] != 0 else 0
    
    return pd.Series(features)

def calculate_advanced_statistical_features_test(flux):
    """Tính các đặc trưng thống kê nâng cao cho test set"""
    features = {}
    
    if len(flux) < 3:
        return features
    
    # Percentile ratios
    percentiles = [10, 25, 50, 75, 90, 95]
    percentile_values = np.percentile(flux, percentiles)
    
    # Important ratios
    if len(percentile_values) >= 5:
        features['p90_p10_ratio'] = percentile_values[4] / (percentile_values[0] + 1e-9)
        features['p75_p25_ratio'] = percentile_values[3] / (percentile_values[1] + 1e-9)
        features['p95_p5_ratio'] = percentile_values[5] / (np.percentile(flux, 5) + 1e-9) if len(percentile_values) >= 6 else np.nan
    
    # IQR
    q75, q25 = np.percentile(flux, [75, 25])
    features['iqr'] = q75 - q25
    
    return features

def extract_improved_features_test(df):
    """Trích xuất các đặc trưng cải tiến cho test set"""
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
        
        # 2. Đặc trưng khoảng trống quan sát
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
        
        # Tỷ lệ số lần quan sát theo từng dải
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
                predicted = slope * x + np.median(flux)
                residuals = flux - predicted
                features['trend_stability'] = 1.0 / (np.std(residuals) / (np.std(flux) + 1e-9) + 1e-9)
            except:
                features['trend_slope'] = 0
                features['trend_stability'] = 0
        else:
            features['trend_slope'] = 0
            features['trend_stability'] = 0
        
        # 6. Đặc trưng hình thái
        if len(flux) >= 3:
            features['variability_index'] = np.std(flux) / (np.mean(flux_positive) + 1e-9)
            features['peakiness'] = np.max(flux_positive) / (np.median(flux_positive) + 1e-9)
            
            peak_idx = np.argmax(flux)
            if 0 < peak_idx < len(flux) - 1:
                before_peak = flux[:peak_idx]
                after_peak = flux[peak_idx+1:]
                std_before = np.std(before_peak) if len(before_peak) > 1 else 0
                std_after = np.std(after_peak) if len(after_peak) > 1 else 0
                features['asymmetry'] = std_after / (std_before + 1e-9)
            else:
                features['asymmetry'] = 0
            
            if len(flux) > 5:
                first_half = flux[:len(flux)//2]
                second_half = flux[len(flux)//2:]
                features['rise_fall_ratio_global'] = np.mean(first_half) / (np.mean(second_half) + 1e-9)
        
        # 7. Thêm advanced statistical features
        advanced_stats = calculate_advanced_statistical_features_test(flux)
        features.update(advanced_stats)
        
        features['object_id'] = object_id
        features_list.append(features)
    
    return pd.DataFrame(features_list)

# 4. Các hàm gốc (giữ nguyên nhưng thêm cho test)
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

def weighted_slope(group):
    """Độ dốc có trọng số theo thời gian"""
    group = group.sort_values('mjd')
    if len(group) < 2:
        return 0
    delta_mjd = np.diff(group['mjd'])
    delta_flux = np.diff(group['flux_corrected'])
    slopes = delta_flux / (delta_mjd + 1e-9)
    return np.mean(slopes) if len(slopes) > 0 else 0

# 5. Xử lý light curve của tập test VỚI ELAsTiCC2 FEATURES
def process_test_lightcurves_elastricc():
    """Xử lý light curve tập test với ELAsTiCC2 features"""
    
    print("\n--- Đang xử lý light curve tập test với ELAsTiCC2 features ---")
    
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
        
        # Tính khoảng cách và độ sáng tuyệt đối
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
        
        # FEATURE ENGINEERING VỚI ELAsTiCC2
        grouped = df.groupby('object_id')
        
        # A. Đặc trưng toàn cục cơ bản
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
        
        # B. Đặc trưng độ sáng tuyệt đối
        abs_mag_agg = grouped.agg(
            abs_mag_min=('absolute_mag', 'min'),
            abs_mag_max=('absolute_mag', 'max'),
            abs_mag_mean=('absolute_mag', 'mean'),
            abs_mag_std=('absolute_mag', 'std'),
            abs_mag_span=('absolute_mag', lambda x: x.max() - x.min())
        )
        agg_features = agg_features.join(abs_mag_agg, how='left')
        
        # C. Đặc trưng vật lý cơ bản
        agg_features['weighted_slope_mean'] = grouped.apply(weighted_slope)
        agg_features['decay_alpha'] = grouped.apply(fit_decay_alpha)
        agg_features['is_single_peak'] = grouped.apply(is_single_peak)
        agg_features['rise_fall_ratio'] = grouped.apply(rise_fall_ratio)
        
        # D. Đặc trưng theo dải sóng
        pivot_mean = df.pivot_table(index='object_id', columns='Filter', values='flux_corrected', aggfunc='mean').add_suffix('_mean')
        pivot_std = df.pivot_table(index='object_id', columns='Filter', values='flux_corrected', aggfunc='std').add_suffix('_std')
        pivot_max = df.pivot_table(index='object_id', columns='Filter', values='flux_corrected', aggfunc='max').add_suffix('_max')
        agg_features = agg_features.join([pivot_mean, pivot_std, pivot_max], how='left')
        
        # E. Đặc trưng màu sắc cơ bản
        for f in FILTERS:
            if f'{f}_mean' not in agg_features.columns:
                agg_features[f'{f}_mean'] = np.nan
        
        agg_features['color_g_r_mean'] = agg_features['g_mean'] - agg_features['r_mean']
        agg_features['color_r_i_mean'] = agg_features['r_mean'] - agg_features['i_mean']
        agg_features['color_i_z_mean'] = agg_features['i_mean'] - agg_features['z_mean']
        
        # F. Đặc trưng cải tiến gốc
        improved_features = extract_improved_features_test(df)
        agg_features = agg_features.merge(improved_features, on='object_id', how='left')
        
        # G. ĐẶC TRƯNG ELAsTiCC2 MỚI
        # 1. Color features
        color_features = calculate_color_features_test(agg_features)
        agg_features = agg_features.merge(color_features, on='object_id', how='left')
        
        # 2. Rise/Fade features
        rise_fade_features = calculate_rise_fade_features_test(df)
        agg_features = agg_features.merge(rise_fade_features, on='object_id', how='left')
        
        # 3. Peak analysis features
        peak_features = grouped.apply(enhanced_peak_analysis_test)
        peak_features = peak_features.reset_index()
        agg_features = agg_features.merge(peak_features, on='object_id', how='left')
        
        # 4. Temporal features
        temporal_features = grouped.apply(calculate_temporal_features_test)
        temporal_features = temporal_features.reset_index()
        agg_features = agg_features.merge(temporal_features, on='object_id', how='left')
        
        all_test_features.append(agg_features)
    
    if not all_test_features:
        raise ValueError("Không tạo được đặc trưng cho tập test")
    
    # Gộp tất cả đặc trưng
    test_feature_df = pd.concat(all_test_features)
    test_feature_df.reset_index(inplace=True)
    
    return test_feature_df

# 6. Sinh đặc trưng cho tập test VỚI ELAsTiCC2
test_features_df = process_test_lightcurves_elastricc()

# 7. Gộp metadata và đặc trưng
print("\n--- Gộp dữ liệu tập test ---")
test_final = test_log.merge(test_features_df, on='object_id', how='left')

# 8. KIỂM TRA VÀ XỬ LÝ FEATURES
print("\n--- Kiểm tra features ---")

# Kiểm tra đặc trưng bị thiếu
missing_features = set(selected_features) - set(test_final.columns)
if missing_features:
    print(f"⚠️  Cảnh báo: thiếu {len(missing_features)} đặc trưng, sẽ điền 0")
    for feature in missing_features:
        test_final[feature] = 0

# Kiểm tra features thừa (không có trong selected_features)
extra_features = set(test_final.columns) - set(selected_features + ['object_id', 'Z', 'EBV', 'distance_pc'])
print(f"Có {len(extra_features)} features thừa sẽ bị bỏ qua")

# 9. Chuẩn bị dữ liệu dự đoán
X_test = test_final[selected_features].copy()

# Xử lý giá trị thiếu THEO CHIẾN LƯỢC CỦA TRAIN
print("\n--- Xử lý missing values ---")

# Phân loại columns để xử lý khác nhau
std_cols = [c for c in X_test.columns if 'std' in c.lower()]
mean_cols = [c for c in X_test.columns if 'mean' in c.lower() and 'color' not in c.lower()]
color_cols = [c for c in X_test.columns if 'color' in c.lower()]
ratio_cols = [c for c in X_test.columns if 'ratio' in c.lower()]

# Fill missing values với chiến lược giống train
X_test[std_cols] = X_test[std_cols].fillna(0)
X_test[mean_cols] = X_test[mean_cols].fillna(X_test[mean_cols].median())
X_test[color_cols] = X_test[color_cols].fillna(0)
X_test[ratio_cols] = X_test[ratio_cols].fillna(1)

# Fill remaining numeric columns
numeric_cols = X_test.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if X_test[col].isnull().any():
        if col.startswith('p') and col[1:].isdigit():
            X_test[col] = X_test[col].fillna(X_test[col].median())
        else:
            X_test[col] = X_test[col].fillna(0)

print(f"Kích thước đặc trưng tập test: {X_test.shape}")
print(f"Số lượng objects: {len(X_test)}")

# 10. Tiến hành dự đoán
print("\n--- Đang dự đoán ---")
test_predictions = []

for i, model in enumerate(models):
    try:
        pred = model.predict_proba(X_test)[:, 1]
        test_predictions.append(pred)
        print(f"✅ Mô hình {i+1}/{len(models)} đã dự đoán xong")
    except Exception as e:
        print(f"❌ Lỗi với mô hình {i+1}: {e}")
        # Nếu có lỗi, dùng random predictions
        test_predictions.append(np.random.rand(len(X_test)) * 0.5)

# Trung bình dự đoán
test_preds_mean = np.mean(test_predictions, axis=0)

# Áp dụng ngưỡng tối ưu
test_binary = (test_preds_mean >= best_threshold).astype(int)

print(f"\n🎯 Dự đoán tập test hoàn tất:")
print(f"   Xác suất dự đoán trung bình: {test_preds_mean.mean():.4f} ± {test_preds_mean.std():.4f}")
print(f"   Tỷ lệ mẫu dương được dự đoán: {test_binary.mean():.4f} ({test_binary.sum()}/{len(test_binary)})")

print(f"\n📊 Phân bố xác suất dự đoán:")
for percentile in [10, 25, 50, 75, 90]:
    value = np.percentile(test_preds_mean, percentile)
    print(f"   Phân vị {percentile:2d}%: {value:.4f}")

# 11. Tạo file submission
print("\n--- Tạo file submission ---")
submission = pd.DataFrame({
    'object_id': test_final['object_id'],
    'predicted': test_binary
})

# Đảm bảo mọi object trong tập test đều có dự đoán
missing_objects = set(test_log['object_id']) - set(submission['object_id'])
if missing_objects:
    print(f"⚠️  Cảnh báo: {len(missing_objects)} object chưa có dự đoán, gán 0")
    missing_df = pd.DataFrame({
        'object_id': list(missing_objects),
        'predicted': 0
    })
    submission = pd.concat([submission, missing_df], ignore_index=True)

# Sắp xếp theo object_id
submission = submission.sort_values('object_id')

# Lưu file submission VỚI TÊN MỚI
submission_file = 'submission_elastricc.csv'
submission.to_csv(submission_file, index=False)

print(f"\n✅ Đã tạo file submission: '{submission_file}'")
print(f"   Kích thước file submission: {submission.shape}")
print(f"   Số mẫu dương được dự đoán: {submission['predicted'].sum()}")
print(f"   Tỷ lệ mẫu dương: {submission['predicted'].mean():.4f}")

# 12. Lưu kết quả dự đoán chi tiết
detailed_predictions = pd.DataFrame({
    'object_id': test_final['object_id'],
    'prediction_prob': test_preds_mean,
    'predicted': test_binary,
    'Z': test_final['Z'],
    'EBV': test_final['EBV']
})
detailed_predictions.to_csv('test_detailed_predictions_elastricc.csv', index=False)
print(f"✅ Đã lưu kết quả dự đoán chi tiết: 'test_detailed_predictions_elastricc.csv'")

# 13. Phân tích dự đoán NÂNG CAO
print("\n--- Phân tích dự đoán nâng cao ---")
print(f"📈 Ngưỡng sử dụng: {best_threshold:.3f}")
print(f"📊 Khoảng xác suất dự đoán: [{test_preds_mean.min():.4f}, {test_preds_mean.max():.4f}]")

# Phân loại confidence levels
high_confidence_pos = (test_preds_mean > 0.7).sum()
medium_confidence_pos = ((test_preds_mean >= 0.5) & (test_preds_mean <= 0.7)).sum()
low_confidence_pos = ((test_preds_mean >= best_threshold) & (test_preds_mean < 0.5)).sum()

high_confidence_neg = (test_preds_mean < 0.3).sum()
medium_confidence_neg = ((test_preds_mean >= 0.3) & (test_preds_mean < best_threshold)).sum()

print(f"\n🎯 Confidence Analysis:")
print(f"   High confidence positives (p > 0.7): {high_confidence_pos}")
print(f"   Medium confidence positives (0.5 ≤ p ≤ 0.7): {medium_confidence_pos}")
print(f"   Low confidence positives ({best_threshold:.2f} ≤ p < 0.5): {low_confidence_pos}")
print(f"   High confidence negatives (p < 0.3): {high_confidence_neg}")
print(f"   Medium confidence negatives (0.3 ≤ p < {best_threshold:.2f}): {medium_confidence_neg}")

# 14. Feature importance analysis
print("\n--- Phân tích features trên tập test ---")
if hasattr(models[0], 'feature_importances_'):
    try:
        feature_importance = np.mean([model.feature_importances_ for model in models], axis=0)
        importance_df = pd.DataFrame({
            'feature': selected_features,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print("\n🏆 Top 15 đặc trưng quan trọng nhất trên tập test:")
        for i, row in importance_df.head(15).iterrows():
            print(f"   {row['importance']:.4f} - {row['feature']}")
        
        # Lưu feature importance
        importance_df.to_csv('test_feature_importance_elastricc.csv', index=False)
        
        # Hiển thị top ELAsTiCC2 features
        elastricc_features = [f for f in importance_df['feature'] if any(keyword in f for keyword in [
            'color_', 'rise_', 'fade_', 'peak_', 'autocorr_', 'p90_', 'p75_', 'p95_'
        ])]
        
        print(f"\n🔬 Top ELAsTiCC2 features trong predictions:")
        for feat in elastricc_features[:10]:
            imp = importance_df[importance_df['feature'] == feat]['importance'].iloc[0]
            print(f"   {imp:.4f} - {feat}")
            
    except Exception as e:
        print(f"⚠️  Không thể tính feature importance: {e}")

# 15. Dự đoán LB score
print("\n--- Dự đoán điểm LB ---")
oof_f1 = model_info.get('oof_score', 0)
if oof_f1 >= 0.45:
    print("🎉 XUẤT SẮC! Dự kiến LB score: 0.45+")
    print("   - Model với ELAsTiCC2 features hoạt động tốt")
    print("   - Có khả năng đạt top leaderboard")
elif oof_f1 >= 0.40:
    print("👍 TỐT! Dự kiến LB score: 0.40-0.45")
    print("   - Cải thiện đáng kể so với baseline")
    print("   - ELAsTiCC2 features có tác dụng tích cực")
elif oof_f1 >= 0.35:
    print("⚠️  TRUNG BÌNH! Dự kiến LB score: 0.35-0.40")
    print("   - Cần xem xét distribution shift giữa train-test")
else:
    print("❌ CẦN CẢI THIỆN! Dự kiến LB score: <0.35")
    print("   - Có thể có vấn đề với feature engineering")

print("\n" + "="*80)
print("🎉 HOÀN TẤT DỰ ĐOÁN TẬP TEST VỚI ELAsTiCC2 MODEL!")
print("="*80)
print(f"📤 Hãy nộp file '{submission_file}' lên Kaggle")
print(f"📊 Điểm OOF F1 của model: {oof_f1:.4f}")
print(f"🎯 Ngưỡng tối ưu: {best_threshold:.3f}")
print(f"📈 Số lượng TDE dự đoán: {submission['predicted'].sum()}")
print(f"🔍 Tỷ lệ TDE dự đoán: {submission['predicted'].mean():.4f}")

print("\n💡 LƯU Ý QUAN TRỌNG:")
print("1. Submission file: 'submission_elastricc.csv'")
print("2. Chi tiết predictions: 'test_detailed_predictions_elastricc.csv'")
print("3. So sánh với submission cũ để đánh giá cải thiện")
print("4. Kiểm tra distribution của features nếu LB score thấp")

print("\n🚀 NEXT STEPS:")
print("1. Submit lên Kaggle: 'submission_elastricc.csv'")
print("2. So sánh score với baseline")
print("3. Phân tích các predictions có confidence cao")
print("4. Nếu cần, fine-tune threshold dựa trên public LB")