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
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# Cấu hình
BASE_PATH = '/content/data/'
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
# CÁC HÀM TRÍCH XUẤT ĐẶC TRƯNG MỚI TỪ ELAsTiCC2
# ---------------------------------------------------------------------------------

def calculate_color_features(df):
    """
    Tính các đặc trưng màu sắc từ ELAsTiCC2
    """
    color_features_list = []
    
    for object_id in df['object_id'].unique():
        obj_data = df[df['object_id'] == object_id]
        
        features = {'object_id': object_id}
        
        # Tính mean color giữa các bands
        for band_pair in [('g', 'r'), ('r', 'i'), ('g', 'i'), ('u', 'g'), ('i', 'z'), ('z', 'y')]:
            band1, band2 = band_pair
            
            # Check if we have flux for both bands
            band1_cols = [col for col in obj_data.columns if f'{band1}_' in col]
            band2_cols = [col for col in obj_data.columns if f'{band2}_' in col]
            
            if band1_cols and band2_cols:
                # Use mean flux for color calculation
                band1_mean = obj_data[f'{band1}_mean'].iloc[0] if f'{band1}_mean' in obj_data.columns else np.nan
                band2_mean = obj_data[f'{band2}_mean'].iloc[0] if f'{band2}_mean' in obj_data.columns else np.nan
                
                if not np.isnan(band1_mean) and not np.isnan(band2_mean):
                    features[f'color_{band1}_{band2}_mean'] = band1_mean - band2_mean
        
        # Color ratios (important for TDE identification)
        if 'color_g_r_mean' in features and 'color_r_i_mean' in features:
            if features['color_r_i_mean'] != 0:
                features['color_ratio_gr_ri'] = features['color_g_r_mean'] / features['color_r_i_mean']
        
        color_features_list.append(features)
    
    return pd.DataFrame(color_features_list)

def calculate_rise_fade_features(df):
    """
    Tính rise và fade times dựa trên ý tưởng ELAsTiCC2 (đơn giản hóa)
    """
    features_list = []
    
    for object_id in df['object_id'].unique():
        obj_data = df[df['object_id'] == object_id].sort_values('mjd')
        
        if len(obj_data) < 5:
            features_list.append({'object_id': object_id, 'rise_time': np.nan, 'fade_time': np.nan})
            continue
            
        flux = obj_data['flux_corrected'].values
        mjd = obj_data['mjd'].values
        
        # Tìm peak đơn giản
        peak_idx = np.argmax(flux)
        
        # Rise time: từ start đến peak (ngày)
        rise_time = mjd[peak_idx] - mjd[0] if peak_idx > 0 else np.nan
        
        # Fade time: từ peak đến end (ngày)
        fade_time = mjd[-1] - mjd[peak_idx] if peak_idx < len(flux)-1 else np.nan
        
        # Rise rate: tốc độ tăng đến peak
        if rise_time > 0 and peak_idx > 0:
            rise_rate = (flux[peak_idx] - flux[0]) / rise_time
        else:
            rise_rate = np.nan
            
        # Fade rate: tốc độ giảm sau peak
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

def enhanced_peak_analysis(group):
    """
    Phân tích peak nâng cao từ ELAsTiCC2
    """
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
        features['peak_prominence_std'] = np.std(properties['prominences'])
        
        if 'widths' in properties:
            features['peak_width_mean'] = np.mean(properties['widths'])
            features['peak_width_max'] = np.max(properties['widths'])
        
        # Main peak analysis
        main_peak_idx = peaks[np.argmax(properties['prominences'])]
        features['main_peak_flux'] = flux[main_peak_idx]
        
        # Peak symmetry (ELAsTiCC2 inspired)
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

def calculate_temporal_features(group):
    """
    Tính các đặc trưng thời gian nâng cao
    """
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
        features['gap_min'] = np.min(gaps)
        features['gap_skew'] = pd.Series(gaps).skew()
        
        # Observation density
        total_time = mjd[-1] - mjd[0]
        features['obs_density'] = len(mjd) / (total_time + 1e-9)
        features['obs_frequency'] = 1 / features['gap_mean'] if features['gap_mean'] > 0 else 0
        
        # Time-weighted flux
        time_weights = np.diff(mjd) / total_time
        if len(time_weights) == len(flux) - 1:
            weighted_flux = np.sum(flux[:-1] * time_weights)
            features['time_weighted_flux'] = weighted_flux
    
    # Time series autocorrelation
    if len(flux) >= 5:
        autocorr = np.correlate(flux - np.mean(flux), flux - np.mean(flux), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Lags 1-3
        for lag in [1, 2, 3]:
            if lag < len(autocorr):
                features[f'autocorr_lag{lag}'] = autocorr[lag] / autocorr[0]
    
    return pd.Series(features)

def calculate_band_correlation_features(df):
    """
    Tính correlation giữa các bands khác nhau
    """
    features_list = []
    
    for object_id in df['object_id'].unique():
        obj_data = df[df['object_id'] == object_id]
        
        features = {'object_id': object_id}
        
        # Get band columns
        band_means = {}
        for band in FILTERS:
            col_name = f'{band}_mean'
            if col_name in obj_data.columns:
                band_means[band] = obj_data[col_name].iloc[0]
        
        # Calculate correlations between bands
        if len(band_means) >= 2:
            bands = list(band_means.keys())
            values = list(band_means.values())
            
            # Create correlation matrix
            corr_matrix = np.corrcoef([values] * 2)  # Simple correlation
            
            # Extract features
            features['max_band_correlation'] = np.max(corr_matrix)
            features['min_band_correlation'] = np.min(corr_matrix)
            features['mean_band_correlation'] = np.mean(corr_matrix)
            
            # Specific band correlations
            if 'g' in band_means and 'r' in band_means:
                features['gr_correlation'] = abs(band_means['g'] - band_means['r']) / (abs(band_means['g']) + 1e-9)
        
        features_list.append(features)
    
    return pd.DataFrame(features_list)

def calculate_advanced_statistical_features(flux):
    """
    Tính các đặc trưng thống kê nâng cao
    """
    features = {}
    
    if len(flux) < 3:
        return features
    
    # Higher-order moments
    features['skewness'] = pd.Series(flux).skew()
    features['kurtosis'] = pd.Series(flux).kurtosis()
    
    # Percentile ratios (inspired by ELAsTiCC2)
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    percentile_values = np.percentile(flux, percentiles)
    
    for i, (p, val) in enumerate(zip(percentiles, percentile_values)):
        features[f'p{p}'] = val
    
    # Important ratios
    features['p90_p10_ratio'] = percentile_values[6] / (percentile_values[2] + 1e-9)  # p90/p10
    features['p75_p25_ratio'] = percentile_values[5] / (percentile_values[3] + 1e-9)  # p75/p25
    features['p95_p5_ratio'] = percentile_values[7] / (percentile_values[1] + 1e-9)   # p95/p5
    
    # MAD (Median Absolute Deviation)
    median = np.median(flux)
    mad = np.median(np.abs(flux - median))
    features['mad'] = mad
    features['mad_to_median'] = mad / (median + 1e-9)
    
    # IQR
    q75, q25 = np.percentile(flux, [75, 25])
    features['iqr'] = q75 - q25
    features['iqr_to_median'] = features['iqr'] / (median + 1e-9)
    
    return features

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
        
        # 7. Thêm advanced statistical features từ ELAsTiCC2
        advanced_stats = calculate_advanced_statistical_features(flux)
        features.update(advanced_stats)
        
        features['object_id'] = object_id
        features_list.append(features)
    
    return pd.DataFrame(features_list)

# ---------------------------------------------------------------------------------
# CÁC HÀM GỐC ĐƯỢC GIỮ LẠI
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

def weighted_slope(group):
    """Độ dốc có trọng số theo thời gian"""
    group = group.sort_values('mjd')
    if len(group) < 2:
        return 0
    delta_mjd = np.diff(group['mjd'])
    delta_flux = np.diff(group['flux_corrected'])
    slopes = delta_flux / (delta_mjd + 1e-9)
    return np.mean(slopes) if len(slopes) > 0 else 0

# ---------------------------------------------------------------------------------
# QUY TRÌNH XỬ LÝ CHÍNH ĐƯỢC TỐI ƯU
# ---------------------------------------------------------------------------------

def process_split_data(i, train_log_full):
    """
    Xử lý một split dữ liệu
    """
    split_id_str = str(i).zfill(2)
    
    # Load light curve data
    lc_path = f'{BASE_PATH}split_{split_id_str}/train_full_lightcurves.csv'
    if not os.path.exists(lc_path):
        print(f"Warning: File not found {lc_path}")
        return None
    
    lc = pd.read_csv(lc_path)
    lc.rename(columns={'Time (MJD)': 'mjd', 'Flux': 'flux', 'Flux_err': 'flux_err'}, inplace=True)
    
    log = train_log_full[train_log_full['split'] == f'split_{split_id_str}']
    if log.empty or lc.empty:
        return None
    
    # 1. Hiệu chỉnh flux
    processed_dfs = []
    for object_id in log['object_id'].unique():
        object_lc = lc[lc['object_id'] == object_id].copy()
        if object_lc.empty:
            continue
        
        object_log = log[log['object_id'] == object_id].iloc[0]
        ebv = object_log['EBV']
        A_v = R_V * ebv
        
        flux_corrected_list = []
<<<<<<< HEAD
        for index, row in object_lc.iterrows(): #Hiệu chỉnh extinction
            A_lambda = extinction.fitzpatrick99(np.array([EFF_WAVELENGTHS[row['Filter']]]), A_v, R_V)[0]
=======
        for _, row in object_lc.iterrows():
            A_lambda = extinction.fitzpatrick99(
                np.array([EFF_WAVELENGTHS[row['Filter']]]), A_v, R_V
            )[0]
>>>>>>> a63d2e0 ( update 5.23)
            flux_corrected_list.append(row['flux'] * 10**(0.4 * A_lambda))
        
        object_lc['flux_corrected'] = flux_corrected_list
        processed_dfs.append(object_lc)
    
    if not processed_dfs:
        return None
    
    df = pd.concat(processed_dfs)
    
    # 2. Khoảng cách và độ sáng tuyệt đối
    df['distance_pc'] = df['object_id'].map(dist_map)
    df['flux_positive'] = df['flux_corrected'].clip(lower=1e-9)
    df['apparent_mag'] = -2.5 * np.log10(df['flux_positive'])
    df['absolute_mag'] = df['apparent_mag'] - 5 * (np.log10(df['distance_pc']) - 1)
    
    # 3. Tính toán cơ bản
    df['snr'] = df['flux_corrected'] / df['flux_err']
    df = df.sort_values(['object_id', 'mjd'])
    
    # 4. FEATURE ENGINEERING CHÍNH
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
    
    # C. Đặc trưng từ ELAsTiCC2
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
    improved_features = extract_improved_features(df)
    agg_features = agg_features.merge(improved_features, on='object_id', how='left')
    
    # G. ĐẶC TRƯNG MỚI TỪ ELAsTiCC2
    # 1. Color features
    color_features = calculate_color_features(agg_features)
    agg_features = agg_features.merge(color_features, on='object_id', how='left')
    
    # 2. Rise/Fade features
    rise_fade_features = calculate_rise_fade_features(df)
    agg_features = agg_features.merge(rise_fade_features, on='object_id', how='left')
    
    # 3. Peak analysis features
    peak_features = grouped.apply(enhanced_peak_analysis)
    peak_features = peak_features.reset_index()
    agg_features = agg_features.merge(peak_features, on='object_id', how='left')
    
    # 4. Temporal features
    temporal_features = grouped.apply(calculate_temporal_features)
    temporal_features = temporal_features.reset_index()
    agg_features = agg_features.merge(temporal_features, on='object_id', how='left')
    
    # 5. Band correlation features
    band_corr_features = calculate_band_correlation_features(agg_features)
    agg_features = agg_features.merge(band_corr_features, on='object_id', how='left')
    
    return agg_features

# ---------------------------------------------------------------------------------
# MAIN PROCESSING LOOP
# ---------------------------------------------------------------------------------

def main():
    print(f"\n--- Bắt đầu xử lý {NUM_SPLITS} phân mảnh dữ liệu huấn luyện ---")
    start_time = time.time()
    
    all_features_list = []
    
    for i in tqdm(range(1, NUM_SPLITS + 1), desc="Tiến độ tổng thể"):
        agg_features = process_split_data(i, train_log_full)
        if agg_features is not None:
            all_features_list.append(agg_features)
    
    if not all_features_list:
        print("Không có dữ liệu nào được xử lý!")
        return
    
    # 5. Tổng hợp cuối cùng
    print("\n--- Tất cả các phân mảnh đã xử lý xong, bắt đầu tổng hợp ---")
    full_feature_df = pd.concat(all_features_list)
    full_feature_df.reset_index(inplace=True)
    
    # 6. Gộp với metadata
    final_df = pd.merge(train_log_full, full_feature_df, on='object_id', how='right')
    
    # 7. Xử lý giá trị thiếu THÔNG MINH
    print("\n--- Xử lý giá trị thiếu ---")
    
    # Phân loại columns
    std_cols = [c for c in final_df.columns if 'std' in c.lower()]
    mean_cols = [c for c in final_df.columns if 'mean' in c.lower() and 'color' not in c.lower()]
    color_cols = [c for c in final_df.columns if 'color' in c.lower()]
    ratio_cols = [c for c in final_df.columns if 'ratio' in c.lower()]
    
    # Fill missing values với chiến lược khác nhau
    final_df[std_cols] = final_df[std_cols].fillna(0)
    final_df[mean_cols] = final_df[mean_cols].fillna(final_df[mean_cols].median())
    final_df[color_cols] = final_df[color_cols].fillna(0)  # Giả sử không có color difference
    final_df[ratio_cols] = final_df[ratio_cols].fillna(1)  # Giả sử ratio = 1
    
    # Fill remaining numeric columns
    numeric_cols = final_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if final_df[col].isnull().any():
            if col.startswith('p') and col[1:].isdigit():  # Percentile columns
                final_df[col] = final_df[col].fillna(final_df[col].median())
            else:
                final_df[col] = final_df[col].fillna(0)
    
    # 8. Tạo target column cho TDE
    print("\n--- Tạo target variable ---")
    # Giả sử TDE có trong 'SpecType' hoặc 'English Translation'
    if 'SpecType' in final_df.columns:
        final_df['target'] = (final_df['SpecType'] == 'TDE').astype(int)
    elif 'English Translation' in final_df.columns:
        final_df['target'] = (final_df['English Translation'] == 'Tidal Disruption Event').astype(int)
    else:
        print("Warning: Không tìm thấy column để tạo target. Cần manual check.")
        final_df['target'] = 0
    
    print(f"Số lượng TDE (target=1): {final_df['target'].sum()}")
    print(f"Tỷ lệ TDE: {final_df['target'].mean():.4f}")
    
    # 9. Lưu kết quả
    OUTPUT_FILE = 'processed_train_features_elastricc_enhanced.csv'
    final_df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n" + "="*80)
    print("✅ HOÀN TẤT XỬ LÝ DỮ LIỆU VỚI TÍCH HỢP ELAsTiCC2!")
    print("="*80)
    print(f"Tổng thời gian: {(time.time() - start_time)/60:.2f} phút")
    print(f"Kích thước dữ liệu đặc trưng: {final_df.shape}")
    print(f"Số lượng features: {len(final_df.columns)}")
    print(f"Số lượng objects: {len(final_df)}")
    print(f"File đã được lưu tại: '{OUTPUT_FILE}'")
    
    # 10. Thông tin thêm
    print("\n--- THÔNG TIN FEATURES MỚI ---")
    new_features = [
        'rise_time', 'fade_time', 'rise_fade_ratio', 'rise_rate', 'fade_rate',
        'peak_count', 'peak_prominence_mean', 'peak_symmetry', 'peak_asymmetry',
        'color_g_r_mean', 'color_r_i_mean', 'color_i_z_mean',
        'color_g_i_mean', 'color_u_g_mean', 'color_z_y_mean',
        'autocorr_lag1', 'autocorr_lag2', 'autocorr_lag3',
        'p90_p10_ratio', 'p75_p25_ratio', 'p95_p5_ratio',
        'max_band_correlation', 'min_band_correlation'
    ]
    
    available_new_features = [f for f in new_features if f in final_df.columns]
    print(f"Đã thêm {len(available_new_features)} features mới từ ELAsTiCC2:")
    for i, feat in enumerate(available_new_features[:20]):  # Hiển thị 20 features đầu
        print(f"  {i+1:2d}. {feat}")
    
    if len(available_new_features) > 20:
        print(f"  ... và {len(available_new_features) - 20} features khác")
    
    print("\n🎯 Các features quan trọng nhất cho TDE classification:")
    print("  1. Color features (g-r, r-i) - Phân biệt TDE vs SNe")
    print("  2. Rise/Fade times - TDE thường có rise nhanh, fade chậm")
    print("  3. Peak symmetry - TDE có peak đối xứng hơn")
    print("  4. Autocorrelation - TDE có temporal structure đặc trưng")

if __name__ == "__main__":
    main()