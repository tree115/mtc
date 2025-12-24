import numpy as np
import pandas as pd
from scipy.stats import theilslopes, iqr, gaussian_kde
from scipy.signal import find_peaks, correlate
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from config import FILTERS, EFF_WAVELENGTHS
from scipy.stats import theilslopes

# ============================================================
# HELPER FUNCTIONS 
# ============================================================

def robust_slope(x, y, min_points=3, flux_offset=1e-3):

    x = np.asarray(x)
    y = np.asarray(y) + flux_offset  # Offset to handle negative flux
    
    # Remove NaN or inf
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    
    if len(x) < min_points:
        return np.nan
    
    try:
        slope, _, _, _ = theilslopes(y, x)  # Robust slope against outliers
        return slope
    except:
        try:
            coef = np.polyfit(x, y, 1)
            return coef[0]
        except:
            return np.nan


def weighted_mean_std(values, errors, min_error=1e-6):

    values = np.asarray(values)
    errors = np.asarray(errors)
    
    mask = np.isfinite(values) & np.isfinite(errors)
    values, errors = values[mask], errors[mask]
    
    if len(values) == 0:
        return np.nan, np.nan
    
    # Avoid zero error
    errors_safe = np.where(errors > 0, errors, min_error)
    weights = 1 / errors_safe**2
    sum_weights = np.sum(weights)
    
    if sum_weights == 0:
        return np.nan, np.nan
    
    weighted_mean = np.sum(values * weights) / sum_weights
    weighted_var = np.sum(weights * (values - weighted_mean)**2) / sum_weights
    weighted_std = np.sqrt(weighted_var)
    
    return weighted_mean, weighted_std


def robust_polyfit(x, y, y_err=None, degree=1, min_points=None, flux_offset=1e-3):
    
    x = np.asarray(x)
    y = np.asarray(y) + flux_offset
    
    if y_err is None:
        y_err = np.ones_like(y)
    else:
        y_err = np.asarray(y_err)
    
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(y_err)
    x, y, y_err = x[mask], y[mask], y_err[mask]
    
    if min_points is None:
        min_points = degree + 2
    if len(x) < min_points:
        return np.full(degree + 1, np.nan), np.full((degree + 1, degree + 1), np.nan)
    
    # Safe weights
    y_err_safe = np.where(y_err > 0, y_err, 1e-6)
    weights = 1 / y_err_safe**2
    weights = np.where(np.isfinite(weights), weights, 0)
    
    if np.sum(weights) == 0:
        return np.full(degree + 1, np.nan), np.full((degree + 1, degree + 1), np.nan)
    
    try:
        coef, cov = np.polyfit(x, y, degree, w=weights, cov=True)
        return coef, cov
    except:
        return np.full(degree + 1, np.nan), np.full((degree + 1, degree + 1), np.nan)


def calculate_color_with_error(t, band1_flux, band2_flux, band1_err, band2_err, flux_offset=1e-3, min_points=3):
  
    band1_flux = np.asarray(band1_flux) + flux_offset
    band2_flux = np.asarray(band2_flux) + flux_offset
    band1_err = np.asarray(band1_err)
    band2_err = np.asarray(band2_err)
    t = np.asarray(t)
    
    mask = np.isfinite(band1_flux) & np.isfinite(band2_flux)
    if mask.sum() < min_points:
        return np.array([]), np.array([]), np.array([])
    
    t_valid = t[mask]
    mag1 = -2.5 * np.log10(band1_flux[mask])
    mag2 = -2.5 * np.log10(band2_flux[mask])
    
    # Error propagation
    mag1_err = 2.5 / np.log(10) * (band1_err[mask] / band1_flux[mask])
    mag2_err = 2.5 / np.log(10) * (band2_err[mask] / band2_flux[mask])
    
    color = mag1 - mag2
    color_err = np.sqrt(mag1_err**2 + mag2_err**2)
    
    return t_valid, color, color_err



# ============================================================
# ADVANCED PHYSICS FUNCTIONS
# ============================================================


FILTERS = ['u','g','r','i','z','y']

def fit_tde_decay_model(t, f, min_points=6, flux_offset=1e-6):
    """
    Robust TDE decay fit: f = A * (t - t0)^(-alpha)
    Handles negative flux by adding small offset.
    Returns alpha, R², chi²
    """
    f = np.asarray(f) + flux_offset
    t = np.asarray(t)
    
    if len(f) < min_points:
        return np.nan, np.nan, np.nan
    
    peak_idx = np.argmax(f)
    if peak_idx >= len(f) - 4:
        return np.nan, np.nan, np.nan
    
    t_decay = t[peak_idx:] - t[peak_idx] + 1e-6
    f_decay = f[peak_idx:]
    
    mask = np.isfinite(f_decay) & (f_decay > 0)
    if mask.sum() < 5:
        return np.nan, np.nan, np.nan
    
    t_fit, f_fit = t_decay[mask], f_decay[mask]
    
    def decay_model(t, alpha, A):
        return A * t**(-alpha)
    
    try:
        popt, pcov = curve_fit(decay_model, t_fit, f_fit, 
                               p0=[1.67, f_fit[0]],
                               bounds=([0.5, 0], [3.0, np.inf]),
                               maxfev=2000)
        alpha = popt[0]
        f_pred = decay_model(t_fit, *popt)
        ss_res = np.sum((f_fit - f_pred)**2)
        ss_tot = np.sum((f_fit - np.mean(f_fit))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
        chi2 = ss_res / (len(f_fit) - 2) if len(f_fit) > 2 else np.nan
        return alpha, r2, chi2
    except:
        return np.nan, np.nan, np.nan


def calculate_rise_fade_times(t, f, threshold_mag=1.0, flux_offset=1e-6):
    """
    Robust rise/fade times and rates from peak flux.
    Handles negative flux by offset.
    Returns: rise_time, fade_time, rise_rate, fade_rate
    """
    f = np.asarray(f) + flux_offset
    t = np.asarray(t)
    
    if len(f) < 5:
        return np.nan, np.nan, np.nan, np.nan
    
    peak_idx = np.argmax(f)
    peak_flux = f[peak_idx]
    peak_time = t[peak_idx]
    threshold_flux = peak_flux / (10**(0.4*threshold_mag))
    
    # Rise time
    rise_time = np.nan
    if peak_idx > 0:
        pre_idx = np.where(f[:peak_idx] <= threshold_flux)[0]
        if len(pre_idx) > 0:
            rise_time = peak_time - t[pre_idx[-1]]
    
    # Fade time
    fade_time = np.nan
    if peak_idx < len(f)-1:
        post_idx = np.where(f[peak_idx:] <= threshold_flux)[0]
        if len(post_idx) > 0:
            fade_time = t[peak_idx + post_idx[0]] - peak_time
    
    rise_rate = (peak_flux - np.median(f[:max(1, peak_idx)])) / (rise_time + 1e-9) if not np.isnan(rise_time) else np.nan
    fade_rate = (peak_flux - np.median(f[peak_idx:])) / (fade_time + 1e-9) if not np.isnan(fade_time) else np.nan
    
    return rise_time, fade_time, rise_rate, fade_rate

def calculate_band_correlations(df, object_id):
    """
    Robust band correlations, interpolated to common grid
    Returns dict of correlations
    """
    bands_data = {}
    for band in FILTERS:
        band_df = df[(df['object_id'] == object_id) & (df['Filter'] == band)].sort_values('mjd')
        if len(band_df) > 3:
            t_band = band_df['mjd'].values
            f_band = band_df['flux_corrected'].values + 1e-6  # flux offset
            t_min, t_max = t_band.min(), t_band.max()
            t_common = np.linspace(t_min, t_max, 100)
            f_interp = interp1d(t_band, f_band, bounds_error=False, fill_value=np.nan)(t_common)
            bands_data[band] = f_interp

    if len(bands_data) < 2:
        return {}

    bands = list(bands_data.keys())
    valid_data = []
    band_names = []
    for b in bands:
        data = bands_data[b][~np.isnan(bands_data[b])]
        if len(data) >= 5:
            valid_data.append(data)
            band_names.append(b)
    if len(valid_data) < 2:
        return {}

    try:
        corr_matrix = np.corrcoef(valid_data)
    except:
        return {}

    features = {
        'max_band_corr': np.nanmax(corr_matrix),
        'min_band_corr': np.nanmin(corr_matrix),
        'mean_band_corr': np.nanmean(corr_matrix),
        'band_corr_std': np.nanstd(corr_matrix)
    }

    # Specific band pairs
    for band1, band2 in [('g','r'), ('r','i'), ('g','i')]:
        if band1 in band_names and band2 in band_names:
            idx1, idx2 = band_names.index(band1), band_names.index(band2)
            features[f'{band1}_{band2}_corr'] = corr_matrix[idx1, idx2]

    return features

def calculate_advanced_peak_features(t, f, flux_offset=1e-6):
    """
    Robust peak features: prominence, width, symmetry, ratios
    Handles negative flux
    """
    f = np.asarray(f) + flux_offset
    t = np.asarray(t)
    
    if len(f) < 10:
        return {}
    
    try:
        std_f = np.std(f)
        peaks, props = find_peaks(f, prominence=0.1*std_f if std_f>0 else 0.1, width=1, distance=5)
    except:
        peaks, props = [], {}
    
    features = {
        'peak_count': len(peaks),
        'is_single_peak': int(len(peaks)==1)
    }
    
    if len(peaks) > 0:
        if 'prominences' in props and len(props['prominences'])>0:
            main_idx = peaks[np.argmax(props['prominences'])]
            features['main_peak_prominence'] = np.max(props['prominences'])
            features['peak_prominence_std'] = np.std(props['prominences'])
            
            # Peak symmetry
            if main_idx>0 and main_idx<len(f)-1:
                left_slope = robust_slope(t[:main_idx], f[:main_idx])
                right_slope = robust_slope(t[main_idx+1:], f[main_idx+1:])
                if not np.isnan(left_slope) and abs(left_slope)>1e-9 and not np.isnan(right_slope):
                    features['peak_symmetry'] = -right_slope / (left_slope+1e-9)
                    features['peak_asymmetry'] = np.abs(features['peak_symmetry']-1)
                else:
                    features['peak_symmetry'] = np.nan
                    features['peak_asymmetry'] = np.nan
        
        if 'widths' in props and len(props['widths'])>0:
            features['peak_width_mean'] = np.mean(props['widths'])
            features['peak_width_std'] = np.std(props['widths'])
    
    median_flux = np.median(f)
    mean_flux = np.mean(f)
    if median_flux>0:
        features['peak_to_median'] = np.max(f)/(median_flux+1e-9)
    if mean_flux>0:
        features['peak_to_mean'] = np.max(f)/(mean_flux+1e-9)
    
    if 'peak_to_median' in features and features['peak_to_median']>1:
        std_f = np.std(f)
        if std_f>0:
            features['peak_significance'] = (np.max(f)-median_flux)/(std_f+1e-9)
    
    return features


# ============================================================
# TEMPORAL FEATURES 
# ============================================================
def calculate_temporal_features(t, f):
    """
    Đặc trưng thời gian nâng cao (temporal features)
    - Xử lý flux âm / gaps lớn / short lightcurves
    - Bao gồm: gap statistics, total duration, obs density, 
      time-weighted flux statistics, autocorrelation
    """
    t = np.asarray(t)
    f = np.asarray(f) + 1e-6  # offset flux âm để tránh lỗi khi tính weighted stats
    features = {}

    # =======================
    # Time gaps statistics
    # =======================
    if len(t) >= 2:
        gaps = np.diff(t)
        finite_gaps = gaps[np.isfinite(gaps)]
        if len(finite_gaps) > 0:
            features['gap_mean'] = np.mean(finite_gaps)
            features['gap_std'] = np.std(finite_gaps)
            features['gap_median'] = np.median(finite_gaps)
            features['gap_max'] = np.max(finite_gaps)
            features['gap_min'] = np.min(finite_gaps)
            
            # Skewness and kurtosis (robust)
            try:
                gap_series = pd.Series(finite_gaps)
                features['gap_skew'] = gap_series.skew()
                features['gap_kurtosis'] = gap_series.kurtosis()
            except:
                features['gap_skew'] = np.nan
                features['gap_kurtosis'] = np.nan
            
            # Regularity metrics
            if features['gap_mean'] > 0:
                features['gap_cv'] = features['gap_std'] / features['gap_mean']  # coefficient of variation
                features['gap_regularity'] = 1 / (features['gap_cv'] + 1e-9)

        # Total duration and observation density
        total_duration = t[-1] - t[0]
        features['total_duration'] = total_duration
        features['obs_density'] = len(t) / total_duration if total_duration > 0 else np.nan
        features['obs_frequency'] = 1 / features['gap_mean'] if 'gap_mean' in features and features['gap_mean'] > 0 else np.nan

    # =======================
    # Time-weighted flux statistics
    # =======================
    if len(t) >= 2 and np.ptp(t) > 0:
        gaps = np.diff(t)
        time_intervals = gaps / np.ptp(t)
        time_weights = np.append(time_intervals, time_intervals[-1] if len(time_intervals)>0 else 1.0)
        time_weights = np.where(np.isfinite(time_weights), time_weights, 1.0)
        sum_weights = np.sum(time_weights)
        if sum_weights > 0:
            weighted_mean = np.sum(f * time_weights) / sum_weights
            features['time_weighted_flux_mean'] = weighted_mean
            weighted_var = np.sum(time_weights * (f - weighted_mean)**2) / sum_weights
            features['time_weighted_flux_std'] = np.sqrt(weighted_var)
    
    # =======================
    # Autocorrelation features
    # =======================
    if len(f) >= 5:
        # Detrend flux robust
        x = np.arange(len(f))
        try:
            slope, _, _, _ = theilslopes(f, x)
            detrended = f - (slope * x + np.median(f))
        except:
            detrended = f - np.mean(f)
        
        # Autocorrelation
        try:
            autocorr = np.correlate(detrended, detrended, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            if autocorr[0] != 0:
                autocorr /= autocorr[0]
            else:
                autocorr = np.zeros_like(autocorr)
        except:
            autocorr = np.zeros(min(10, len(f)))
        
        # Lags 1-3
        for lag in [1,2,3]:
            features[f'autocorr_lag{lag}'] = autocorr[lag] if lag < len(autocorr) else np.nan
        
        # Autocorr decay slope
        max_lag = min(5, len(autocorr)-1)
        if max_lag >= 2:
            valid_lags = np.arange(1, max_lag+1)
            valid_ac = autocorr[1:max_lag+1]
            try:
                ac_slope, _, _, _ = theilslopes(valid_ac, valid_lags)
                features['autocorr_decay_slope'] = ac_slope
            except:
                features['autocorr_decay_slope'] = np.nan

    return features

# ============================================================
# COLOR EVOLUTION FEATURES 
# ============================================================

def calculate_color_evolution_features(df, object_id):
    """
    Tính color evolution features (giống ELAsTiCC2), robust với:
    - Flux âm / negative flux
    - Gaps lớn, short LC
    - Interpolation, weighted stats, pre-peak/post-peak slope
    """
    obj_data = df[df['object_id'] == object_id].sort_values('mjd')
    features = {}

    # Band pairs quan trọng
    for band1, band2 in [('g', 'r'), ('r', 'i'), ('g', 'i')]:
        # Lấy data từng band
        band1_data = obj_data[obj_data['Filter'] == band1]
        band2_data = obj_data[obj_data['Filter'] == band2]
        if len(band1_data) < 3 or len(band2_data) < 3:
            continue

        # Tạo timeline chung (interpolation)
        all_times = np.sort(np.unique(np.concatenate([band1_data['mjd'].values, band2_data['mjd'].values])))
        try:
            f1_interp = interp1d(band1_data['mjd'], band1_data['flux_corrected']+1e-6, 
                                 bounds_error=False, fill_value=np.nan)(all_times)
            f2_interp = interp1d(band2_data['mjd'], band2_data['flux_corrected']+1e-6, 
                                 bounds_error=False, fill_value=np.nan)(all_times)
        except:
            continue

        # Mask: chỉ flux dương & hợp lệ
        mask = ~np.isnan(f1_interp) & ~np.isnan(f2_interp) & (f1_interp > 0) & (f2_interp > 0)
        if mask.sum() < 5:
            continue

        t_color = all_times[mask]
        f1_vals = f1_interp[mask]
        f2_vals = f2_interp[mask]

        # Magnitudes & color
        mag1 = -2.5 * np.log10(f1_vals)
        mag2 = -2.5 * np.log10(f2_vals)
        color = mag1 - mag2

        # Xác định peak dựa trên total flux
        total_flux = f1_vals + f2_vals
        peak_idx = np.argmax(total_flux)
        if peak_idx < 2 or peak_idx >= len(color)-2:
            continue

        # -------- Pre-peak (rise phase) --------
        color_pre = color[:peak_idx]
        t_pre = t_color[:peak_idx]
        if len(color_pre) >= 3:
            errors = np.ones_like(color_pre) * 0.1
            mean_pre, std_pre = weighted_mean_std(color_pre, errors)
            features[f'{band1}{band2}_color_pre_mean'] = mean_pre
            features[f'{band1}{band2}_color_pre_std'] = std_pre

            coef_pre, cov_pre = robust_polyfit(t_pre - t_pre[0], color_pre, errors, degree=1)
            if not np.isnan(coef_pre[0]):
                features[f'{band1}{band2}_color_slope_pre'] = coef_pre[0]
                if cov_pre is not None and not np.any(np.isnan(cov_pre)):
                    features[f'{band1}{band2}_color_slope_pre_err'] = np.sqrt(cov_pre[0, 0])

        # -------- Post-peak (decline phase) --------
        color_post = color[peak_idx:]
        t_post = t_color[peak_idx:]
        if len(color_post) >= 3:
            errors = np.ones_like(color_post) * 0.1
            mean_post, std_post = weighted_mean_std(color_post, errors)
            features[f'{band1}{band2}_color_post_mean'] = mean_post
            features[f'{band1}{band2}_color_post_std'] = std_post

            coef_post, cov_post = robust_polyfit(t_post - t_post[0], color_post, errors, degree=1)
            if not np.isnan(coef_post[0]):
                features[f'{band1}{band2}_color_slope_post'] = coef_post[0]
                if cov_post is not None and not np.any(np.isnan(cov_post)):
                    features[f'{band1}{band2}_color_slope_post_err'] = np.sqrt(cov_post[0, 0])

        # Color change (post - pre)
        if (f'{band1}{band2}_color_pre_mean' in features and 
            f'{band1}{band2}_color_post_mean' in features):
            features[f'{band1}{band2}_color_change'] = (
                features[f'{band1}{band2}_color_post_mean'] - features[f'{band1}{band2}_color_pre_mean']
            )

    # -------- Color ratios--------
    try:
        if ('gr_color_pre_mean' in features and 'ri_color_pre_mean' in features and 
            features['ri_color_pre_mean'] != 0):
            features['color_ratio_pre_gr_ri'] = features['gr_color_pre_mean'] / features['ri_color_pre_mean']
    except:
        features['color_ratio_pre_gr_ri'] = np.nan

    try:
        if ('gr_color_post_mean' in features and 'ri_color_post_mean' in features and 
            features['ri_color_post_mean'] != 0):
            features['color_ratio_post_gr_ri'] = features['gr_color_post_mean'] / features['ri_color_post_mean']
    except:
        features['color_ratio_post_gr_ri'] = np.nan

    return features

# ============================================================
# ADVANCED STATISTICAL FEATURES
# ============================================================

def calculate_advanced_statistics(f):
    """
    Advanced statistical features: percentiles, moments, variability, flux asymmetry.
    Robust với:
    - Short lightcurves (<5 points)
    - Negative flux
    - NaN/Inf
    """
    f = np.asarray(f)
    # Loại bỏ NaN/Inf
    f = f[np.isfinite(f)]
    if len(f) < 5:
        return {}

    features = {}

    # -------- Percentiles --------
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    try:
        percentile_values = np.percentile(f, percentiles)
        for p, val in zip(percentiles, percentile_values):
            features[f'p{p}'] = val

        # Important ratios với kiểm tra >0 để tránh chia 0
        if percentile_values[2] != 0:
            features['p90_p10_ratio'] = percentile_values[6] / percentile_values[2]
        else:
            features['p90_p10_ratio'] = np.nan
        if percentile_values[3] != 0:
            features['p75_p25_ratio'] = percentile_values[5] / percentile_values[3]
        else:
            features['p75_p25_ratio'] = np.nan
        if percentile_values[1] != 0:
            features['p95_p5_ratio'] = percentile_values[7] / percentile_values[1]
        else:
            features['p95_p5_ratio'] = np.nan
    except:
        pass

    # -------- Robust statistics --------
    median = np.median(f)
    mad = np.median(np.abs(f - median))
    features['mad'] = mad
    features['mad_to_median'] = mad / (median + 1e-9)

    # IQR
    try:
        q75, q25 = np.percentile(f, [75, 25])
        features['iqr'] = q75 - q25
        features['iqr_to_median'] = features['iqr'] / (median + 1e-9)
    except:
        features['iqr'] = np.nan
        features['iqr_to_median'] = np.nan

    # Higher moments
    try:
        f_series = pd.Series(f)
        features['skewness'] = f_series.skew()
        features['kurtosis'] = f_series.kurtosis()
    except:
        features['skewness'] = np.nan
        features['kurtosis'] = np.nan

    # -------- Variability indices --------
    mean_f = np.mean(f)
    features['variability_index'] = np.std(f) / (mean_f + 1e-9)
    features['robust_variability'] = iqr(f) / (median + 1e-9)

    # -------- Flux asymmetry --------
    flux_positive = f[f > 0]
    flux_negative = f[f < 0]
    features['positive_flux_ratio'] = len(flux_positive) / len(f)
    if len(flux_positive) > 0 and len(flux_negative) > 0:
        mean_positive = np.mean(flux_positive)
        features['flux_asymmetry'] = np.mean(np.abs(flux_negative)) / (mean_positive + 1e-9)
    else:
        features['flux_asymmetry'] = 0.0

    return features
# ============================================================
# New feature 
# ============================================================

def calculate_flux_derivative_features(t, f, flux_offset=1e-6, min_points=5):
    """
    Tính các đặc trưng từ đạo hàm bậc 1 (slope) và bậc 2 (curvature) của lightcurve.
    Robust với:
      - Flux âm
      - Short lightcurves
      - NaN/Inf
    Trả về dict:
      max_slope, min_slope, median_slope, std_slope
      slope_p90, slope_p10
      max_curvature, mean_curvature, std_curvature
    """
    t = np.asarray(t)
    f = np.asarray(f) + flux_offset
    
    # Mask hợp lệ
    mask = np.isfinite(t) & np.isfinite(f)
    t, f = t[mask], f[mask]
    
    if len(t) < min_points:
        return {
            'max_slope': np.nan, 'min_slope': np.nan, 'median_slope': np.nan, 'std_slope': np.nan,
            'slope_p90': np.nan, 'slope_p10': np.nan,
            'max_curvature': np.nan, 'mean_curvature': np.nan, 'std_curvature': np.nan
        }
    
    # Sort theo thời gian
    sort_idx = np.argsort(t)
    t, f = t[sort_idx], f[sort_idx]
    
    # Đạo hàm bậc 1 (slope)
    dt = np.diff(t)
    df = np.diff(f)
    slopes = df / (dt + 1e-9)
    
    # Đạo hàm bậc 2 (curvature)
    dslopes = np.diff(slopes)
    dt_mid = (dt[:-1] + dt[1:]) / 2
    curvature = dslopes / (dt_mid + 1e-9)
    
    features = {
        'max_slope': np.max(slopes) if len(slopes) > 0 else np.nan,
        'min_slope': np.min(slopes) if len(slopes) > 0 else np.nan,
        'median_slope': np.median(slopes) if len(slopes) > 0 else np.nan,
        'std_slope': np.std(slopes) if len(slopes) > 0 else np.nan,
        'slope_p90': np.percentile(slopes, 90) if len(slopes) > 0 else np.nan,
        'slope_p10': np.percentile(slopes, 10) if len(slopes) > 0 else np.nan,
        'max_curvature': np.max(curvature) if len(curvature) > 0 else np.nan,
        'mean_curvature': np.mean(curvature) if len(curvature) > 0 else np.nan,
        'std_curvature': np.std(curvature) if len(curvature) > 0 else np.nan
    }
    
    return features

def calculate_phase_auc_features(t, f, flux_offset=1e-6, min_points=5):
    """
    Tính AUC và shape moments (skew/kurtosis) của pre-peak & post-peak phases.
    Robust với:
      - Negative flux
      - Gaps lớn
      - Short lightcurves (<5 points)
    Trả về dict:
      pre_auc, post_auc
      pre_skew, post_skew
      pre_kurtosis, post_kurtosis
      rise_fade_auc_ratio
    """
    t = np.asarray(t)
    f = np.asarray(f) + flux_offset
    
    mask = np.isfinite(t) & np.isfinite(f)
    t, f = t[mask], f[mask]
    
    if len(t) < min_points:
        return {
            'pre_auc': np.nan, 'post_auc': np.nan,
            'pre_skew': np.nan, 'post_skew': np.nan,
            'pre_kurtosis': np.nan, 'post_kurtosis': np.nan,
            'rise_fade_auc_ratio': np.nan
        }
    
    # Sort theo thời gian
    sort_idx = np.argsort(t)
    t, f = t[sort_idx], f[sort_idx]
    
    peak_idx = np.argmax(f)
    
    # Pre-peak
    if peak_idx >= 2:
        t_pre, f_pre = t[:peak_idx+1], f[:peak_idx+1]
        pre_auc = np.trapz(f_pre, t_pre)
        pre_skew = pd.Series(f_pre).skew()
        pre_kurt = pd.Series(f_pre).kurtosis()
    else:
        pre_auc = pre_skew = pre_kurt = np.nan
    
    # Post-peak
    if peak_idx <= len(f) - 3:
        t_post, f_post = t[peak_idx:], f[peak_idx:]
        post_auc = np.trapz(f_post, t_post)
        post_skew = pd.Series(f_post).skew()
        post_kurt = pd.Series(f_post).kurtosis()
    else:
        post_auc = post_skew = post_kurt = np.nan
    
    # AUC ratio
    if not np.isnan(pre_auc) and not np.isnan(post_auc) and post_auc > 1e-9:
        rise_fade_auc_ratio = pre_auc / post_auc
    else:
        rise_fade_auc_ratio = np.nan
    
    features = {
        'pre_auc': pre_auc,
        'post_auc': post_auc,
        'pre_skew': pre_skew,
        'post_skew': post_skew,
        'pre_kurtosis': pre_kurt,
        'post_kurtosis': post_kurt,
        'rise_fade_auc_ratio': rise_fade_auc_ratio
    }
    
    return features

# ============================
# NEW: Rest-frame & luminosity proxy features
# ============================
def calculate_restframe_luminosity_features(df, object_id, z=None, distance_pc=None, flux_col='flux_corrected', time_col='mjd', min_points=5):
    """
    Tính các feature rest-frame và luminosity proxy cho một object.
    Trả về dict:
      t_peak_obs, t_peak_rest, peak_flux_obs, log_peak_lum_proxy,
      rest_decay_alpha, rest_decay_r2, rest_rise_time, rest_fade_time, peak_epoch_count_near
    Tự động thử lấy Z/distance_pc từ df nếu không truyền.
    """
    obj = df[df['object_id'] == object_id].sort_values(time_col)
    out = {
        't_peak_obs': np.nan, 't_peak_rest': np.nan,
        'peak_flux_obs': np.nan, 'log_peak_lum_proxy': np.nan,
        'rest_decay_alpha': np.nan, 'rest_decay_r2': np.nan,
        'rest_rise_time': np.nan, 'rest_fade_time': np.nan,
        'peak_epoch_count_near': np.nan
    }

    if obj.shape[0] < min_points:
        return out

    t = obj[time_col].values
    f = obj[flux_col].values

    mask = np.isfinite(t) & np.isfinite(f)
    if mask.sum() < min_points:
        return out
    t = t[mask]; f = f[mask]

    # observed peak
    peak_idx = np.argmax(f)
    t_peak = t[peak_idx]
    peak_flux = f[peak_idx]
    out['t_peak_obs'] = float(t_peak)
    out['peak_flux_obs'] = float(peak_flux)

    # count epochs near peak (window ~ 5*median(dt) fallback)
    if len(t) >= 2:
        dt_med = np.median(np.diff(np.sort(t)))
        window = max(1.0, 5 * dt_med)
    else:
        window = 30.0
    out['peak_epoch_count_near'] = int(np.sum(np.abs(t - t_peak) <= window))

    # distance_pc: try to read from obj if not provided
    if distance_pc is None and 'distance_pc' in obj.columns and obj['distance_pc'].notna().any():
        try:
            distance_pc = float(obj['distance_pc'].dropna().iloc[0])
        except:
            distance_pc = None

    # luminosity proxy (log)
    try:
        if distance_pc is not None and not pd.isna(distance_pc):
            lum_proxy = np.abs(peak_flux) * (float(distance_pc) ** 2)
            out['log_peak_lum_proxy'] = float(np.log10(lum_proxy + 1e-12))
        else:
            out['log_peak_lum_proxy'] = np.nan
    except:
        out['log_peak_lum_proxy'] = np.nan

    # redshift z: try to read from obj if not provided
    if z is None and 'Z' in obj.columns and obj['Z'].notna().any():
        try:
            z = float(obj['Z'].dropna().iloc[0])
        except:
            z = None

    # rest-frame computations if z available
    if z is not None and not pd.isna(z) and z >= 0:
        t_rest = t / (1.0 + z)
        out['t_peak_rest'] = float(t_peak / (1.0 + z))

        # reuse existing functions (fit_tde_decay_model, calculate_rise_fade_times)
        try:
            alpha_rest, r2_rest, chi2_rest = fit_tde_decay_model(t_rest, f, min_points=min_points)
            out['rest_decay_alpha'] = float(alpha_rest) if not pd.isna(alpha_rest) else np.nan
            out['rest_decay_r2'] = float(r2_rest) if not pd.isna(r2_rest) else np.nan
        except:
            out['rest_decay_alpha'] = np.nan
            out['rest_decay_r2'] = np.nan

        try:
            rt_rest, ft_rest, _, _ = calculate_rise_fade_times(t_rest, f)
            out['rest_rise_time'] = float(rt_rest) if not pd.isna(rt_rest) else np.nan
            out['rest_fade_time'] = float(ft_rest) if not pd.isna(ft_rest) else np.nan
        except:
            out['rest_rise_time'] = np.nan
            out['rest_fade_time'] = np.nan

    return out


def calculate_band_derivative_features(df, object_id, filters=FILTERS, min_points=5, flux_offset=1e-6):
    """
    Tính các đặc trưng derivative theo band & cross-band color derivative.
    Trả về dict:
        - band_slope_{band}: slope của flux theo thời gian
        - band_slope_std_{band}: độ ổn định slope (std residual)
        - color_slope_{band1}_{band2}: slope của color (mag1 - mag2)
    Robust với:
        - Negative flux
        - Gaps lớn
        - Short LCs (<5 points)
    """
    features = {}
    obj_data = df[df['object_id'] == object_id]

    # ---------- Band-wise flux slopes ----------
    for band in filters:
        band_data = obj_data[obj_data['Filter'] == band].sort_values('mjd')
        if len(band_data) < min_points:
            continue
        t_band = band_data['mjd'].values
        f_band = band_data['flux_corrected'].values + flux_offset
        
        mask = np.isfinite(t_band) & np.isfinite(f_band)
        t_band, f_band = t_band[mask], f_band[mask]
        if len(t_band) < min_points:
            continue
        
        slope = robust_slope(t_band, f_band)
        features[f'band_slope_{band}'] = slope

        # Residual stability
        if not np.isnan(slope):
            pred = slope * (t_band - t_band[0]) + np.median(f_band)
            residuals = f_band - pred
            features[f'band_slope_std_{band}'] = np.std(residuals) / (np.std(f_band) + 1e-9)
        else:
            features[f'band_slope_std_{band}'] = np.nan

    # ---------- Cross-band color slopes ----------
    band_pairs = [('g','r'), ('r','i'), ('g','i')]
    for b1, b2 in band_pairs:
        b1_data = obj_data[obj_data['Filter'] == b1].sort_values('mjd')
        b2_data = obj_data[obj_data['Filter'] == b2].sort_values('mjd')
        if len(b1_data) < min_points or len(b2_data) < min_points:
            continue

        # Interpolate to common timeline
        all_times = np.sort(np.unique(np.concatenate([b1_data['mjd'].values, b2_data['mjd'].values])))
        try:
            f1_interp = interp1d(b1_data['mjd'], b1_data['flux_corrected'].values + flux_offset,
                                 bounds_error=False, fill_value=np.nan)(all_times)
            f2_interp = interp1d(b2_data['mjd'], b2_data['flux_corrected'].values + flux_offset,
                                 bounds_error=False, fill_value=np.nan)(all_times)
        except:
            continue

        mask = ~np.isnan(f1_interp) & ~np.isnan(f2_interp) & (f1_interp > 0) & (f2_interp > 0)
        if mask.sum() < min_points:
            continue

        t_color = all_times[mask]
        color_vals = -2.5 * np.log10(f1_interp[mask]) + 2.5 * np.log10(f2_interp[mask])  # mag1 - mag2

        # Color slope
        coef, _ = robust_polyfit(t_color - t_color[0], color_vals, degree=1)
        if not np.isnan(coef[0]):
            features[f'color_slope_{b1}_{b2}'] = coef[0]
        else:
            features[f'color_slope_{b1}_{b2}'] = np.nan

    return features

from scipy.signal import welch

def calculate_variability_features(t, f, min_points=8, flux_offset=1e-6):
    """
    Tính các đặc trưng biến thiên (variability) dựa trên PSD/FFT.
    Trả về dict:
        - psd_peak_freq: tần số chính (max PSD)
        - psd_peak_power: công suất tại peak
        - psd_power_ratio: ratio power cao/total
        - flux_rms: độ lệch chuẩn flux
        - flux_var_index: biến thiên flux (std/mean)
    Robust với:
        - Short LC
        - Negative flux
        - Gaps lớn
    """
    features = {}
    t = np.asarray(t)
    f = np.asarray(f) + flux_offset

    mask = np.isfinite(t) & np.isfinite(f)
    t, f = t[mask], f[mask]
    if len(f) < min_points:
        return features

    # Detrend flux bằng Theil-Sen slope
    try:
        slope = robust_slope(t, f)
        f_detrended = f - (slope * (t - t[0]) + np.median(f))
    except:
        f_detrended = f - np.mean(f)

    # RMS flux & variability index
    features['flux_rms'] = np.std(f)
    features['flux_var_index'] = np.std(f) / (np.mean(f) + 1e-9)

    # PSD bằng Welch
    try:
        if len(f_detrended) >= 8:
            fs = 1 / np.median(np.diff(t))  # sampling freq
            f_freq, psd = welch(f_detrended, fs=fs, nperseg=min(8, len(f_detrended)))
            if len(psd) > 0:
                idx_peak = np.argmax(psd)
                features['psd_peak_freq'] = f_freq[idx_peak]
                features['psd_peak_power'] = psd[idx_peak]
                features['psd_power_ratio'] = np.sum(psd[idx_peak:]) / (np.sum(psd) + 1e-9)
            else:
                features['psd_peak_freq'] = np.nan
                features['psd_peak_power'] = np.nan
                features['psd_power_ratio'] = np.nan
        else:
            features['psd_peak_freq'] = np.nan
            features['psd_peak_power'] = np.nan
            features['psd_power_ratio'] = np.nan
    except:
        features['psd_peak_freq'] = np.nan
        features['psd_peak_power'] = np.nan
        features['psd_power_ratio'] = np.nan

    return features

def calculate_cross_band_flare_features(df, object_id, filters=FILTERS):
    """
    Tính các đặc trưng tương quan giữa các band:
    - Peak flux ratio
    - Rise/Fade time ratio
    - Decay slope ratio
    - Color-flux correlation
    """
    obj_data = df[df['object_id'] == object_id]
    features = {}

    band_features = {}
    for band in filters:
        band_df = obj_data[obj_data['Filter']==band].sort_values('mjd')
        if len(band_df) < 5:
            continue
        t_band = band_df['mjd'].values
        f_band = band_df['flux_corrected'].values + 1e-6
        mask = np.isfinite(t_band) & np.isfinite(f_band)
        t_band, f_band = t_band[mask], f_band[mask]
        if len(f_band) < 5:
            continue
        
        # Peak flux
        peak_flux = np.max(f_band)
        peak_idx = np.argmax(f_band)

        # Rise/fade
        rise_time, fade_time, _, _ = calculate_rise_fade_times(t_band, f_band)
        # Decay slope
        decay_alpha, _, _ = fit_tde_decay_model(t_band, f_band)
        
        band_features[band] = {
            'peak_flux': peak_flux,
            'rise_time': rise_time,
            'fade_time': fade_time,
            'decay_alpha': decay_alpha
        }

    # Cross-band ratios
    band_list = list(band_features.keys())
    for i in range(len(band_list)):
        for j in range(i+1, len(band_list)):
            b1, b2 = band_list[i], band_list[j]
            bf1, bf2 = band_features[b1], band_features[b2]
            
            # Peak flux ratio
            if bf2['peak_flux'] > 1e-9:
                features[f'peak_flux_ratio_{b1}_{b2}'] = bf1['peak_flux'] / bf2['peak_flux']
            
            # Rise/fade time ratio
            if bf2['rise_time'] and bf2['fade_time']:
                if bf2['rise_time'] > 1e-9:
                    features[f'rise_time_ratio_{b1}_{b2}'] = (bf1['rise_time'] or 0) / bf2['rise_time']
                if bf2['fade_time'] > 1e-9:
                    features[f'fade_time_ratio_{b1}_{b2}'] = (bf1['fade_time'] or 0) / bf2['fade_time']
            
            # Decay slope ratio
            if bf2['decay_alpha'] and bf2['decay_alpha'] > 1e-9:
                features[f'decay_alpha_ratio_{b1}_{b2}'] = (bf1['decay_alpha'] or 0) / bf2['decay_alpha']

    return features


# ============================================================
# MAIN FEATURE ENGINEER CLASS
# ============================================================

class UltimateFeatureEngineer:
    def __init__(self, filters=None):
        # Nếu không truyền filters, dùng FILTERS mặc định
        self.filters = filters if filters else FILTERS
        self.band_pairs = [('g', 'r'), ('r', 'i'), ('g', 'i'), ('u', 'g'), ('i', 'z'), ('z', 'y')]

    def extract_object_features(self, df):
        """Object-level features (all bands combined)"""
        rows = []

        for oid in df["object_id"].unique():
            d = df[df.object_id == oid].sort_values("mjd")
            if len(d) < 8:
                continue

            t = d.mjd.values
            f = d.flux_corrected.values
            f_err = d.flux_err.values

            # Robust filtering
            mask = np.isfinite(f) & np.isfinite(t) & np.isfinite(f_err)
            t, f, f_err = t[mask], f[mask], f_err[mask]
            if len(t) < 5:
                continue

            row = {"object_id": oid}

            # 1. BASIC FLUX STATISTICS
            row["flux_mean"] = np.mean(f)
            row["flux_std"] = np.std(f)
            row["flux_median"] = np.median(f)
            row["flux_min"] = np.min(f)
            row["flux_max"] = np.max(f)

            # 2. ADVANCED STATISTICS
            adv_stats = calculate_advanced_statistics(f)
            row.update(adv_stats)

            rest_feats = calculate_restframe_luminosity_features(df, oid)
            row.update(rest_feats)

            deriv_feats = calculate_flux_derivative_features(t, f)
            row.update(deriv_feats)

            # 3. SNR FEATURES
            valid_snr = (f_err > 0) & np.isfinite(f)
            if valid_snr.any():
                snr = f[valid_snr] / f_err[valid_snr]
                row.update({
                    "snr_mean": np.mean(snr),
                    "snr_median": np.median(snr),
                    "snr_std": np.std(snr),
                    "snr_min": np.min(snr),
                    "snr_max": np.max(snr)
                })
            else:
                row.update({k: np.nan for k in ["snr_mean","snr_median","snr_std","snr_min","snr_max"]})

            # 4. PEAK ANALYSIS
            peak_features = calculate_advanced_peak_features(t, f)
            row.update(peak_features)

            # 5. RISE/FADE TIMES AND RATES
            rise_time, fade_time, rise_rate, fade_rate = calculate_rise_fade_times(t, f)
            row.update({
                "rise_time": rise_time,
                "fade_time": fade_time,
                "rise_rate": rise_rate,
                "fade_rate": fade_rate
            })
            if not np.isnan(rise_time) and not np.isnan(fade_time) and fade_time > 1e-9:
                row["rise_fade_ratio"] = rise_time / fade_time
            if not np.isnan(rise_rate) and not np.isnan(fade_rate) and fade_rate > 1e-9:
                row["rise_fade_rate_ratio"] = rise_rate / fade_rate
            
            phase_feats = calculate_phase_auc_features(t, f)
            row.update(phase_feats)

            # 6. TDE DECAY PHYSICS
            decay_alpha, decay_r2, decay_chi2 = fit_tde_decay_model(t, f)
            row.update({
                "decay_alpha": decay_alpha,
                "decay_r2": decay_r2,
                "decay_chi2": decay_chi2,
                "is_tde_like_decay": int(1.4 <= decay_alpha <= 2.0 and decay_r2 > 0.6) 
                                     if not np.isnan(decay_alpha) and not np.isnan(decay_r2) else 0
            })

            # 7. TEMPORAL FEATURES
            temp_features = calculate_temporal_features(t, f)
            row.update(temp_features)

            # 8. TREND AND RESIDUALS
            trend = robust_slope(t, f)
            row["trend_slope"] = trend
            if not np.isnan(trend):
                pred = trend * (t - t[0]) + np.median(f)
                residuals = f - pred
                std_f = np.std(f)
                if std_f > 1e-9:
                    row["trend_stability"] = np.std(residuals) / std_f
                if len(residuals) >= 5:
                    try:
                        ac = correlate(residuals - np.mean(residuals), residuals - np.mean(residuals), mode='full')
                        ac = ac[len(ac)//2:]
                        row["residual_autocorr1"] = ac[1]/ac[0] if len(ac) > 1 and ac[0] != 0 else np.nan
                        row["residual_autocorr2"] = ac[2]/ac[0] if len(ac) > 2 and ac[0] != 0 else np.nan
                    except:
                        row["residual_autocorr1"] = np.nan
                        row["residual_autocorr2"] = np.nan

            # 9. BAND COVERAGE
            unique_filters = d['Filter'].nunique()
            row["filter_coverage"] = unique_filters / 6.0 if unique_filters > 0 else 0
            row["obs_per_filter_avg"] = len(d) / unique_filters if unique_filters > 0 else np.nan

            # 8a. BAND-WISE DERIVATIVE FEATURES
            band_deriv_feats = calculate_band_derivative_features(df, oid)
            row.update(band_deriv_feats)

            # 8b. VARIABILITY / PSD FEATURES
            var_feats = calculate_variability_features(t, f)
            row.update(var_feats)

            # Sau band-specific features
            cross_band_feats = calculate_cross_band_flare_features(df, oid, self.filters)
            row.update(cross_band_feats)

            rows.append(row)

        return pd.DataFrame(rows)

    def extract_color_features(self, df):
        """Color features inspired by ELAsTiCC2"""
        rows = []

        for oid in df["object_id"].unique():
            d = df[df.object_id == oid]
            row = {"object_id": oid}

            band_medians = {}
            for band in self.filters:
                band_data = d[d.Filter==band]['flux_corrected'].values
                band_medians[band] = np.median(band_data) if len(band_data) >= 3 else np.nan

            for band1, band2 in self.band_pairs:
                if not np.isnan(band_medians.get(band1, np.nan)) and not np.isnan(band_medians.get(band2, np.nan)):
                    row[f"color_{band1}_{band2}_mean"] = band_medians[band1] - band_medians[band2]

            color_evo_features = calculate_color_evolution_features(df, oid)
            row.update(color_evo_features)
            rows.append(row)

        return pd.DataFrame(rows)

    def extract_band_specific_features(self, df):
        """Band-specific decay consistency"""
        rows = []

        for oid in df["object_id"].unique():
            d = df[df.object_id == oid]
            row = {"object_id": oid}
            decay_alphas, rise_times, fade_times, band_peaks = [], [], [], []

            for band in self.filters:
                band_data = d[d.Filter==band].sort_values('mjd')
                if len(band_data) < 6:
                    continue
                t_band, f_band = band_data['mjd'].values, band_data['flux_corrected'].values
                mask = np.isfinite(t_band) & np.isfinite(f_band)
                t_band, f_band = t_band[mask], f_band[mask]
                if len(t_band) < 5:
                    continue

                alpha, r2, _ = fit_tde_decay_model(t_band, f_band)
                if not np.isnan(alpha) and r2 > 0.4:
                    decay_alphas.append(alpha)

                rt, ft, _, _ = calculate_rise_fade_times(t_band, f_band)
                if not np.isnan(rt): rise_times.append(rt)
                if not np.isnan(ft): fade_times.append(ft)

                if len(f_band) > 0: band_peaks.append(np.max(f_band))

            if len(decay_alphas) >= 2:
                mean_alpha, std_alpha = np.mean(decay_alphas), np.std(decay_alphas)
                row.update({
                    "band_decay_alpha_mean": mean_alpha,
                    "band_decay_alpha_std": std_alpha,
                    "band_decay_alpha_cv": std_alpha/mean_alpha if mean_alpha>1e-9 else np.nan
                })
            if len(rise_times) >= 2:
                mean_rise, std_rise = np.mean(rise_times), np.std(rise_times)
                row.update({
                    "band_rise_time_std": std_rise,
                    "band_rise_time_cv": std_rise/mean_rise if mean_rise>1e-9 else np.nan
                })
            if len(fade_times) >= 2:
                mean_fade, std_fade = np.mean(fade_times), np.std(fade_times)
                row.update({
                    "band_fade_time_std": std_fade,
                    "band_fade_time_cv": std_fade/mean_fade if mean_fade>1e-9 else np.nan
                })
            if len(band_peaks) >= 2:
                max_peak, min_peak = np.max(band_peaks), np.min(band_peaks)
                row.update({
                    "band_peak_ratio": max_peak/min_peak if min_peak>1e-9 else np.nan,
                    "band_peak_std": np.std(band_peaks)
                })

            corr_features = calculate_band_correlations(df, oid)
            row.update(corr_features)
            rows.append(row)

        return pd.DataFrame(rows)

    def extract_metadata_features(self, df):
        """Metadata-based features"""
        metadata_cols = [c for c in ['object_id', 'Z', 'EBV', 'distance_pc', 'absolute_mag'] if c in df.columns]
        if not metadata_cols: return pd.DataFrame()

        metadata_df = df[metadata_cols].drop_duplicates('object_id')

        if 'Z' in metadata_df.columns and 'EBV' in metadata_df.columns:
            metadata_df['Z_times_EBV'] = metadata_df['Z'] * metadata_df['EBV']
        if 'absolute_mag' in metadata_df.columns:
            try:
                metadata_df['abs_mag_bin'] = pd.cut(metadata_df['absolute_mag'],
                                                   bins=[-30, -25, -20, -15, -10],
                                                   labels=[1, 2, 3, 4])
            except:
                metadata_df['abs_mag_bin'] = np.nan
        return metadata_df

    def extract_all_features(self, df):
        obj_features = self.extract_object_features(df)
        color_features = self.extract_color_features(df)
        band_features = self.extract_band_specific_features(df)
        meta_features = self.extract_metadata_features(df)

        # Merge all features
        features = obj_features
        for feat_df in [color_features, band_features, meta_features]:
            if not feat_df.empty:
                features = features.merge(feat_df, on='object_id', how='left')

        # Fill NaN
        for col in features.columns:
            if col == 'object_id': continue
            if pd.api.types.is_numeric_dtype(features[col]):
                median_val = features[col].median()
                features[col] = features[col].fillna(median_val)
            elif pd.api.types.is_categorical_dtype(features[col]):
                if 'missing' not in features[col].cat.categories:
                    features[col] = features[col].cat.add_categories('missing')
                features[col] = features[col].fillna('missing')
            else:
                features[col] = features[col].fillna('missing')

        # Remove constant numeric features
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        constant_cols = [c for c in numeric_cols if features[c].std() < 1e-6]
        if constant_cols:
            features = features.drop(columns=constant_cols)

        return features
