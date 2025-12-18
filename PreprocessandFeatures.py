import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import extinction
from scipy.optimize import curve_fit
from scipy.stats import theilslopes
import os
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u

# Cấu hình
BASE_PATH = './data/'
NUM_SPLITS = 20
EFF_WAVELENGTHS = {'u': 3641, 'g': 4704, 'r': 6155, 'i': 7504, 'z': 8695, 'y': 10056}
R_V = 3.1
FILTERS = ['u', 'g', 'r', 'i', 'z', 'y']

print("Đang tải file metadata chính train_log.csv...")
train_log_full = pd.read_csv(f'{BASE_PATH}train_log.csv')

# ---------------------------------------------------------------------------------
# Tính trước khoảng cách cho tất cả các thiên thể
# ---------------------------------------------------------------------------------
print("\n--- Bắt đầu tính các đặc trưng vật lý thiên văn (khoảng cách & độ sáng tuyệt đối) ---")
log_meta = train_log_full[['object_id', 'Z']].copy()

# Xử lý trường hợp redshift bằng 0 hoặc âm
z = log_meta['Z'].values
z[z <= 0] = 1e-6

print("Đang tính khoảng cách quang độ...")
dist_pc = cosmo.luminosity_distance(z).to(u.pc).value # TDE là sự kiện cực sáng, cần so độ sáng tuyệt đối
dist_map = dict(zip(log_meta['object_id'], dist_pc))
print("Hoàn tất tính khoảng cách!")

# ---------------------------------------------------------------------------------
# Các hàm trích xuất đặc trưng
# ---------------------------------------------------------------------------------
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
    """Trích xuất các đặc trưng cải tiến và ổn định"""
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
        
        # 5. Độ ổn định xu hướng cải tiến (dùng Theil-Sen)
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
            
            # Tỷ lệ tăng/giảm
            if len(flux) > 5:
                first_half = flux[:len(flux)//2]
                second_half = flux[len(flux)//2:]
                features['rise_fall_ratio_global'] = np.mean(first_half) / (np.mean(second_half) + 1e-9)
        
        features['object_id'] = object_id
        features_list.append(features)
    
    return pd.DataFrame(features_list)

# ---------------------------------------------------------------------------------
# Quy trình xử lý chính
# ---------------------------------------------------------------------------------
all_features_list = []

print(f"\n--- Bắt đầu xử lý {NUM_SPLITS} phân mảnh dữ liệu huấn luyện ---")
start_time = time.time()

for i in tqdm(range(1, NUM_SPLITS + 1), desc="Tiến độ tổng thể"):
    split_id_str = str(i).zfill(2)
    
    lc = pd.read_csv(f'{BASE_PATH}split_{split_id_str}/train_full_lightcurves.csv')
    lc.rename(columns={'Time (MJD)': 'mjd', 'Flux': 'flux', 'Flux_err': 'flux_err'}, inplace=True)

    log = train_log_full[train_log_full['split'] == f'split_{split_id_str}']
    if log.empty or lc.empty: continue

    # 1. Hiệu chỉnh flux
    processed_dfs = []
    for object_id in log['object_id'].unique():
        object_lc = lc[lc['object_id'] == object_id].copy()
        if object_lc.empty: continue
        object_log = log[log['object_id'] == object_id].iloc[0]
        ebv = object_log['EBV']
        A_v = R_V * ebv
        flux_corrected_list = []
        for index, row in object_lc.iterrows(): #Hiệu chỉnh extinction
            A_lambda = extinction.fitzpatrick99(np.array([EFF_WAVELENGTHS[row['Filter']]]), A_v, R_V)[0]
            flux_corrected_list.append(row['flux'] * 10**(0.4 * A_lambda))
        object_lc['flux_corrected'] = flux_corrected_list
        processed_dfs.append(object_lc)
    if not processed_dfs: continue
    df = pd.concat(processed_dfs)

    # 2. Khoảng cách và độ sáng tuyệt đối
    df['distance_pc'] = df['object_id'].map(dist_map)
    df['flux_positive'] = df['flux_corrected'].clip(lower=1e-9)
    df['apparent_mag'] = -2.5 * np.log10(df['flux_positive'])
    df['absolute_mag'] = df['apparent_mag'] - 5 * (np.log10(df['distance_pc']) - 1)

    # 3. Tính toán cơ bản & sắp xếp
    df['snr'] = df['flux_corrected'] / df['flux_err']
    df = df.sort_values(['object_id', 'mjd'])
    
    # 4. Feature engineering
    grouped = df.groupby('object_id')
    
    # a. Đặc trưng toàn cục
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

    # Đặc trưng dựa trên độ sáng tuyệt đối
    abs_mag_agg = grouped.agg(
        abs_mag_min=('absolute_mag', 'min'),
        abs_mag_max=('absolute_mag', 'max'),
        abs_mag_mean=('absolute_mag', 'mean'),
        abs_mag_std=('absolute_mag', 'std'),
        abs_mag_span=('absolute_mag', lambda x: x.max() - x.min())
    )
    agg_features = agg_features.join(abs_mag_agg, how='left')

    # b. Đặc trưng độ dốc có trọng số theo thời gian
    def weighted_slope(group):
        group = group.sort_values('mjd')
        delta_mjd = np.diff(group['mjd'])
        delta_flux = np.diff(group['flux_corrected'])
        slopes = delta_flux / (delta_mjd + 1e-9)
        return np.mean(slopes) if len(slopes) > 0 else 0
    agg_features['weighted_slope_mean'] = grouped.apply(weighted_slope)

    # c. Đặc trưng vật lý
    agg_features['decay_alpha'] = grouped.apply(fit_decay_alpha)
    agg_features['is_single_peak'] = grouped.apply(is_single_peak)
    agg_features['rise_fall_ratio'] = grouped.apply(rise_fall_ratio)

    # d. Đặc trưng theo dải sóng
    pivot_mean = df.pivot_table(index='object_id', columns='Filter', values='flux_corrected', aggfunc='mean').add_suffix('_mean')
    pivot_std = df.pivot_table(index='object_id', columns='Filter', values='flux_corrected', aggfunc='std').add_suffix('_std')
    pivot_max = df.pivot_table(index='object_id', columns='Filter', values='flux_corrected', aggfunc='max').add_suffix('_max')
    agg_features = agg_features.join([pivot_mean, pivot_std, pivot_max], how='left')

    # e. Đặc trưng màu sắc
    for f in FILTERS:
        if f'{f}_mean' not in agg_features.columns: 
            agg_features[f'{f}_mean'] = np.nan
    agg_features['color_g_r_mean'] = agg_features['g_mean'] - agg_features['r_mean']
    agg_features['color_r_i_mean'] = agg_features['r_mean'] - agg_features['i_mean']
    agg_features['color_i_z_mean'] = agg_features['i_mean'] - agg_features['z_mean']

    # f. Thêm các đặc trưng cải tiến
    improved_features = extract_improved_features(df)
    agg_features = agg_features.merge(improved_features, on='object_id', how='left')

    all_features_list.append(agg_features)

# 5. Tổng hợp cuối cùng
print("\n--- Tất cả các phân mảnh đã xử lý xong, bắt đầu tổng hợp ---")
full_feature_df = pd.concat(all_features_list)
full_feature_df.reset_index(inplace=True)

# 6. Gộp với metadata
final_df = pd.merge(train_log_full, full_feature_df, on='object_id', how='right')

# 7. Điền giá trị thiếu
std_cols = [c for c in final_df.columns if 'std' in c]
final_df[std_cols] = final_df[std_cols].fillna(0)
numeric_cols = final_df.select_dtypes(include=[np.number]).columns
final_df[numeric_cols] = final_df[numeric_cols].fillna(final_df[numeric_cols].median())

# 8. Lưu kết quả
OUTPUT_FILE = 'processed_train_features_improved.csv'
final_df.to_csv(OUTPUT_FILE, index=False)

print("\n--- ✅ Hoàn tất toàn bộ quá trình xử lý dữ liệu! ---")
print(f"Tổng thời gian: {(time.time() - start_time)/60:.2f} phút")
print(f"Kích thước dữ liệu đặc trưng huấn luyện cuối cùng: {final_df.shape}")
print(f"File đã được lưu tại: '{OUTPUT_FILE}'")
