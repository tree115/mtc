# gp_features.py
"""
GP features extractor for MALLORN (gp_features.csv generator)

Features:
 - GP-based peak, rise/fade, asymmetry
 - GP hyperparameters
 - Decay-law alpha (fit to GP mean post-peak)
 - CI-based features (variance percentiles and pre/post ratios)
 - GP-based colors (g-r, r-i, g-i) pre/post peak & slopes
 - Multiprocessing support, split-aware
 - Save to data/process/gp_train_features.csv and gp_test_features.csv
 - Merge helpers to combine with base train/test features (from config)

Usage:
 - create_gp_train_dataset()
 - create_gp_test_dataset()
 - merge_all_train_features(), merge_all_test_features()
"""

import os
import time
import gc
import math
import logging
import warnings
from pathlib import Path
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import minimize, curve_fit

# GP
import george
from george import kernels

# project imports (must exist in your repo)
from config import TRAIN_FEATURES, TEST_FEATURES
from data_loader import DataLoader
from preprocessor import Preprocessor

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger("gp_features")
logger.setLevel(logging.INFO)

# Output paths
OUT_DIR = Path("/content/data/processed/")
GP_TRAIN = OUT_DIR / "gp_train_features.csv"
GP_TEST = OUT_DIR / "gp_test_features.csv"
ALL_TRAIN = OUT_DIR / "all_train_features.csv"
ALL_TEST = OUT_DIR / "all_test_features.csv"

# LSST wavelengths
LSST_WAVELENGTH = {
    'u': 3670.69,
    'g': 4826.85,
    'r': 6223.24,
    'i': 7545.98,
    'z': 8590.90,
    'y': 9710.28
}

# ======================================================================
# Helper utils
# ======================================================================

def _safe_get_col(df, choices):
    """Return the first existing column name from choices present in df, else None."""
    for c in choices:
        if c in df.columns:
            return c
    return None

def _normalize_columns(df):
    """
    Normalize column names so GP functions can use:
      Time column -> 'Time (MJD)'
      Flux -> 'Flux'
      Flux_err -> 'Flux_err'
      Filter -> 'Filter'
    Also ensures object_id and split preserved.
    """
    df = df.copy()
    col_map = {}

    # time
    time_col = _safe_get_col(df, ["Time (MJD)", "mjd", "MJD", "Time_MJD", "time", "time_mjd"])
    if time_col and time_col != "Time (MJD)":
        col_map[time_col] = "Time (MJD)"

    flux_col = _safe_get_col(df, ["Flux", "flux", "flux_corrected", "FLUXCAL", "flux_obs"])
    if flux_col and flux_col != "Flux":
        col_map[flux_col] = "Flux"

    ferr_col = _safe_get_col(df, ["Flux_err", "flux_err", "FluxErr", "FLUXCALERR"])
    if ferr_col and ferr_col != "Flux_err":
        col_map[ferr_col] = "Flux_err"

    filter_col = _safe_get_col(df, ["Filter", "filter", "BAND", "band"])
    if filter_col and filter_col != "Filter":
        col_map[filter_col] = "Filter"

    oid_col = _safe_get_col(df, ["object_id", "OBJECTID", "objectId", "id"])
    if oid_col and oid_col != "object_id":
        col_map[oid_col] = "object_id"

    split_col = _safe_get_col(df, ["split", "Split"])
    if split_col and split_col != "split":
        col_map[split_col] = "split"

    if col_map:
        df = df.rename(columns=col_map)

    return df

# ======================================================================
# GP core functions (based on your snippets, made robust)
# ======================================================================

def prepare_gp_inputs_mallorn(df_obj, min_points=6):
    """
    Prepare (X, t, wave, flux, flux_err, baseline, scale)
    Accepts df_obj with columns: Time (MJD), Flux, Flux_err, Filter
    """
    try:
        df = _normalize_columns(df_obj)
        t = np.asarray(df["Time (MJD)"], dtype=float)
        flux = np.asarray(df["Flux"], dtype=float)
        ferr = np.asarray(df["Flux_err"], dtype=float)
        filt = np.asarray(df["Filter"], dtype=str)

        wave = np.array([LSST_WAVELENGTH.get(b.strip(), np.nan) if isinstance(b, str) else np.nan for b in filt])

        mask = np.isfinite(t) & np.isfinite(flux) & np.isfinite(ferr) & np.isfinite(wave)
        if mask.sum() < min_points:
            return None

        t, flux, ferr, wave = t[mask], flux[mask], ferr[mask], wave[mask]

        baseline = np.median(flux)
        flux_centered = flux - baseline

        scale = np.nanpercentile(np.abs(flux_centered), 90)
        if not np.isfinite(scale) or scale <= 0:
            scale = np.std(flux_centered) + 1e-6

        X = np.vstack([t, wave]).T

        return {
            "X": X,
            "t": t,
            "wave": wave,
            "flux": flux_centered,
            "flux_err": ferr,
            "baseline": baseline,
            "scale": scale
        }
    except Exception as e:
        logger.debug("prepare_gp_inputs_mallorn failed: %s", e)
        return None

def fit_george_gp_mallorn(gp_data, n_retries=3):
    """
    Fit 2D George GP; returns dict with 'gp', 'baseline', 'params' or None
    """
    if gp_data is None:
        return None

    X = gp_data["X"]
    y = gp_data["flux"]
    yerr = gp_data["flux_err"]
    scale = float(gp_data["scale"])

    try:
        kernel = (0.5 * scale)**2 * kernels.Matern32Kernel(metric=[100.0**2, 6000.0**2], ndim=2)
        gp = george.GP(kernel, solver=george.HODLRSolver)
        gp.compute(X, yerr)

        def neg_ln_like(p):
            gp.set_parameter_vector(p)
            return -gp.log_likelihood(y)

        def grad_neg_ln_like(p):
            gp.set_parameter_vector(p)
            return -gp.grad_log_likelihood(y)

        p0 = gp.get_parameter_vector()
        best = None

        for attempt in range(n_retries):
            try:
                res = minimize(neg_ln_like, p0, jac=grad_neg_ln_like, method="L-BFGS-B")
                if best is None or (res.success and res.fun < best.fun):
                    best = res
                if res.success:
                    break
                p0 = p0 + np.random.normal(0, 1e-2, size=len(p0))
            except Exception:
                p0 = p0 + np.random.normal(0, 1e-2, size=len(p0))
                continue

        if best is None or not getattr(best, "success", False):
            # Set to last p0 (best guess) but still return gp object
            try:
                gp.set_parameter_vector(p0)
            except Exception:
                return None
        else:
            gp.set_parameter_vector(best.x)

        return {"gp": gp, "baseline": gp_data["baseline"], "params": gp.get_parameter_vector()}
    except Exception as e:
        logger.debug("fit_george_gp_mallorn error: %s", e)
        return None

def select_anchor_band(df_obj, min_points=3):
    """Select anchor band (highest median SNR)"""
    df = _normalize_columns(df_obj)
    if "Filter" not in df.columns:
        return None

    best_band = None
    best_score = -np.inf
    for band in pd.unique(df["Filter"]):
        sub = df[df["Filter"] == band]
        if len(sub) < min_points:
            continue
        snr = np.abs(sub["Flux"].values) / (sub["Flux_err"].values + 1e-6)
        score = np.nanmedian(snr)
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_band = band
    return best_band

def gp_predict_band(gp, gp_flux, band, t_min, t_max, n_grid=800, margin_pre=50, margin_post=100):
    wave = LSST_WAVELENGTH.get(band, np.nan)
    if not np.isfinite(wave):
        return None, None, None
    t_grid = np.linspace(t_min - margin_pre, t_max + margin_post, n_grid)
    X_pred = np.vstack([t_grid, wave * np.ones_like(t_grid)]).T
    mean, var = gp.predict(gp_flux, X_pred, return_var=True)
    return t_grid, mean, var

def gp_peak_features(df_obj, gp_model, gp_flux, anchor_band="auto", min_snr_peak=5.0):
    df = _normalize_columns(df_obj)
    if gp_model is None or "gp" not in gp_model:
        return {"gp_peak_time": np.nan, "gp_peak_flux": np.nan, "gp_peak_snr": np.nan, "gp_anchor_band": None}

    if anchor_band == "auto":
        anchor_band = select_anchor_band(df)

    if anchor_band is None:
        return {"gp_peak_time": np.nan, "gp_peak_flux": np.nan, "gp_peak_snr": np.nan, "gp_anchor_band": None}

    try:
        t_min = float(df["Time (MJD)"].min())
        t_max = float(df["Time (MJD)"].max())
    except Exception:
        return {"gp_peak_time": np.nan, "gp_peak_flux": np.nan, "gp_peak_snr": np.nan, "gp_anchor_band": anchor_band}

    t_grid, mean, var = gp_predict_band(gp_model["gp"], gp_flux, anchor_band, t_min, t_max)
    if mean is None or len(mean) == 0 or np.all(~np.isfinite(mean)):
        return {"gp_peak_time": np.nan, "gp_peak_flux": np.nan, "gp_peak_snr": np.nan, "gp_anchor_band": anchor_band}

    idx_peak = int(np.nanargmax(mean))
    peak_flux = float(mean[idx_peak])
    peak_time = float(t_grid[idx_peak])
    peak_err = float(np.sqrt(var[idx_peak])) if var is not None else np.nan
    peak_snr = peak_flux / (peak_err + 1e-6)

    if not np.isfinite(peak_snr) or peak_snr < min_snr_peak:
        peak_flux = np.nan
        peak_time = np.nan
        peak_snr = np.nan

    return {"gp_peak_time": peak_time, "gp_peak_flux": peak_flux, "gp_peak_snr": peak_snr, "gp_anchor_band": anchor_band}

def gp_rise_fade_features(df_obj, gp_model, gp_flux, peak_time, anchor_band, mag_drop=2.512, max_window_pre=200, max_window_post=400):
    if gp_model is None or "gp" not in gp_model:
        return {"gp_rise_time": np.nan, "gp_fade_time": np.nan, "gp_asymmetry": np.nan}

    if not np.isfinite(peak_time) or anchor_band is None:
        return {"gp_rise_time": np.nan, "gp_fade_time": np.nan, "gp_asymmetry": np.nan}

    df = _normalize_columns(df_obj)
    try:
        t_min = float(df["Time (MJD)"].min())
        t_max = float(df["Time (MJD)"].max())
    except Exception:
        return {"gp_rise_time": np.nan, "gp_fade_time": np.nan, "gp_asymmetry": np.nan}

    t_grid, mean, var = gp_predict_band(gp_model["gp"], gp_flux, anchor_band, t_min, t_max, n_grid=1200, margin_pre=100, margin_post=300)
    if mean is None or len(mean) < 10 or np.all(~np.isfinite(mean)):
        return {"gp_rise_time": np.nan, "gp_fade_time": np.nan, "gp_asymmetry": np.nan}

    idx_peak = int(np.nanargmax(mean))
    peak_flux = mean[idx_peak]
    if not np.isfinite(peak_flux):
        return {"gp_rise_time": np.nan, "gp_fade_time": np.nan, "gp_asymmetry": np.nan}

    threshold = peak_flux / mag_drop

    # rise
    rise_time = np.nan
    try:
        pre_mask = (t_grid < peak_time) & (t_grid > peak_time - max_window_pre)
        idx_pre = np.where(pre_mask)[0]
        if len(idx_pre) > 3:
            below = np.where(mean[idx_pre] <= threshold)[0]
            if len(below) > 0:
                t_cross = float(t_grid[idx_pre[below[-1]]])
                rise_time = float(peak_time - t_cross)
    except Exception:
        rise_time = np.nan

    # fade
    fade_time = np.nan
    try:
        post_mask = (t_grid > peak_time) & (t_grid < peak_time + max_window_post)
        idx_post = np.where(post_mask)[0]
        if len(idx_post) > 3:
            below = np.where(mean[idx_post] <= threshold)[0]
            if len(below) > 0:
                t_cross = float(t_grid[idx_post[below[0]]])
                fade_time = float(t_cross - peak_time)
    except Exception:
        fade_time = np.nan

    asymmetry = np.nan
    if np.isfinite(rise_time) and np.isfinite(fade_time) and rise_time > 0:
        asymmetry = float(fade_time / (rise_time + 1e-12))

    return {"gp_rise_time": rise_time, "gp_fade_time": fade_time, "gp_asymmetry": asymmetry}

def gp_predict_color(gp, gp_flux, t_grid, band1, band2, min_flux=1e-8):
    w1 = LSST_WAVELENGTH.get(band1, np.nan)
    w2 = LSST_WAVELENGTH.get(band2, np.nan)
    if not np.isfinite(w1) or not np.isfinite(w2):
        return np.array([]), np.array([]), np.array([])

    X1 = np.vstack([t_grid, w1 * np.ones_like(t_grid)]).T
    X2 = np.vstack([t_grid, w2 * np.ones_like(t_grid)]).T
    f1, v1 = gp.predict(gp_flux, X1, return_var=True)
    f2, v2 = gp.predict(gp_flux, X2, return_var=True)

    f1c = np.clip(f1, min_flux, None)
    f2c = np.clip(f2, min_flux, None)
    mag1 = -2.5 * np.log10(f1c)
    mag2 = -2.5 * np.log10(f2c)
    color = mag1 - mag2

    with np.errstate(divide="ignore", invalid="ignore"):
        err1 = 2.5 / np.log(10) * np.sqrt(np.clip(v1, 0, None)) / f1c
        err2 = 2.5 / np.log(10) * np.sqrt(np.clip(v2, 0, None)) / f2c
        color_err = np.sqrt(err1**2 + err2**2)

    mask = np.isfinite(color) & np.isfinite(color_err)
    return t_grid[mask], color[mask], color_err[mask]

def gp_color_features(df_obj, gp_model, gp_flux, peak_time, band1, band2, rise_time, fade_time, max_color_err=1.0):
    if gp_model is None or "gp" not in gp_model or not np.isfinite(peak_time):
        return {f'gp_mean_color_pre_{band1}{band2}': np.nan,
                f'gp_mean_color_post_{band1}{band2}': np.nan,
                f'gp_color_slope_pre_{band1}{band2}': np.nan,
                f'gp_color_slope_post_{band1}{band2}': np.nan}

    pre_win = float(rise_time) if np.isfinite(rise_time) else 50.0
    post_win = float(fade_time) if np.isfinite(fade_time) else 100.0
    t_pre = np.linspace(peak_time - pre_win, peak_time, 200)
    t_post = np.linspace(peak_time, peak_time + post_win, 200)

    t1, c1, e1 = gp_predict_color(gp_model["gp"], gp_flux, t_pre, band1, band2)
    mask1 = (e1 < max_color_err)
    t1, c1, e1 = t1[mask1], c1[mask1], e1[mask1]
    if len(c1) >= 3:
        w1 = 1.0 / (e1**2 + 1e-6)
        mean_pre = float(np.sum(c1 * w1) / np.sum(w1))
        p1 = np.polyfit(t1 - peak_time, c1, 1, w=w1)
        slope_pre = float(p1[0])
    else:
        mean_pre = np.nan
        slope_pre = np.nan

    t2, c2, e2 = gp_predict_color(gp_model["gp"], gp_flux, t_post, band1, band2)
    mask2 = (e2 < max_color_err)
    t2, c2, e2 = t2[mask2], c2[mask2], e2[mask2]
    if len(c2) >= 3:
        w2 = 1.0 / (e2**2 + 1e-6)
        mean_post = float(np.sum(c2 * w2) / np.sum(w2))
        p2 = np.polyfit(t2 - peak_time, c2, 1, w=w2)
        slope_post = float(p2[0])
    else:
        mean_post = np.nan
        slope_post = np.nan

    return {f'gp_mean_color_pre_{band1}{band2}': mean_pre,
            f'gp_mean_color_post_{band1}{band2}': mean_post,
            f'gp_color_slope_pre_{band1}{band2}': slope_pre,
            f'gp_color_slope_post_{band1}{band2}': slope_post}

def gp_hyperparameter_features(gp_model):
    feats = {'gp_log_amp': np.nan, 'gp_log_ell_time': np.nan, 'gp_log_ell_wave': np.nan, 'gp_ell_ratio_time_wave': np.nan}
    if gp_model is None or "gp" not in gp_model:
        return feats
    try:
        gp = gp_model["gp"]
        kernel = gp.kernel

        # try to deduce parameters robustly
        params = gp.get_parameter_vector()
        feats["gp_params_len"] = len(params)

        # attempt to extract lengthscales/constants by reading kernel attributes if present
        try:
            # george kernel objects differ across versions; use safe checks
            # if kernel is scaled Matern: kernel.constant * Matern(...); .log_constant may exist
            # we'll inspect parameter vector and guess
            feats["gp_log_amp"] = float(getattr(kernel, "log_constant", np.nan))
        except:
            feats["gp_log_amp"] = np.nan

        # try to extract length scales (best-effort)
        try:
            # many george kernels expose `log_length_scale`
            log_len = getattr(kernel, "log_length_scale", None)
            if log_len is not None:
                ell = np.exp(log_len)
                if hasattr(ell, "__len__") and len(ell) >= 2:
                    feats["gp_log_ell_time"] = float(np.log(ell[0])) if ell[0] > 0 else np.nan
                    feats["gp_log_ell_wave"] = float(np.log(ell[1])) if ell[1] > 0 else np.nan
                    feats["gp_ell_ratio_time_wave"] = float(ell[0] / (ell[1] + 1e-12)) if ell[1] > 0 else np.nan
                elif np.isfinite(ell):
                    feats["gp_log_ell_time"] = float(np.log(ell))
        except Exception:
            pass
    except Exception:
        pass
    return feats
# ---------------------------------------------------------------------
# 1) Achromaticity features
# ---------------------------------------------------------------------
def gp_achromaticity_features(df_obj, gp_model, gp_flux, peak_time, anchor_band=None, bands=None, eps=1e-12):
    """
    Measure achromaticity at (or near) the GP peak.
    Returns dict:
      - gp_achro_std: std of relative fluxes across bands at peak
      - gp_achro_maxdiff: max-min relative flux ratio across bands
      - gp_achro_chi2: chi2 of fluxes around median (using GP variances)
      - gp_achro_score: 1/(1+std) as normalized "achromaticity" (higher = more achromatic)
      - gp_achro_nbands: number of bands used
    Inputs:
      df_obj: original object's observations (used to know available bands)
      gp_model: dict produced by fit_george_gp_mallorn (contains 'gp')
      gp_flux: array used as training flux in gp.predict
      peak_time: float MJD peak time (may be np.nan)
      anchor_band: optional band to normalize to (string)
      bands: list of bands to evaluate; by default uses bands present in df_obj
    """
    out = {
        "gp_achro_std": np.nan,
        "gp_achro_maxdiff": np.nan,
        "gp_achro_chi2": np.nan,
        "gp_achro_score": np.nan,
        "gp_achro_nbands": 0,
    }
    try:
        if gp_model is None or "gp" not in gp_model or not np.isfinite(peak_time):
            return out
        gp = gp_model["gp"]

        # determine bands to evaluate
        df = _normalize_columns(df_obj)
        present = []
        if "Filter" in df.columns:
            present = [b for b in pd.unique(df["Filter"]) if b in LSST_WAVELENGTH]
        if bands is None:
            bands_use = present if len(present) > 0 else list(LSST_WAVELENGTH.keys())
        else:
            bands_use = [b for b in bands if b in LSST_WAVELENGTH]
            if len(bands_use) == 0 and len(present) > 0:
                bands_use = present

        if len(bands_use) == 0:
            return out

        fluxes = []
        errs = []
        for b in bands_use:
            w = LSST_WAVELENGTH.get(b)
            if not np.isfinite(w):
                fluxes.append(np.nan); errs.append(np.nan); continue
            Xp = np.vstack([np.array([peak_time]), np.array([w])]).T
            try:
                m, v = gp.predict(gp_flux, Xp, return_var=True)
                fluxes.append(float(m[0]) if np.isfinite(m).any() else np.nan)
                errs.append(float(np.sqrt(max(v[0], 0.0))) if np.isfinite(v).any() else np.nan)
            except Exception:
                fluxes.append(np.nan); errs.append(np.nan)

        fluxes = np.asarray(fluxes, dtype=float)
        errs = np.asarray(errs, dtype=float)

        valid = np.isfinite(fluxes)
        if valid.sum() < 2:
            return out

        fluxes_v = fluxes[valid]
        errs_v = errs[valid]
        nb = len(fluxes_v)
        out["gp_achro_nbands"] = int(nb)

        # normalization: anchor band or median absolute
        norm = None
        if anchor_band and anchor_band in bands_use:
            idx = bands_use.index(anchor_band)
            if np.isfinite(fluxes[idx]):
                norm = np.abs(fluxes[idx])
        if norm is None:
            norm = np.nanmax(np.abs(fluxes_v))
        if not np.isfinite(norm) or norm <= 0:
            norm = np.nanmedian(np.abs(fluxes_v)) + eps
        rel = fluxes_v / (norm + eps)  # relative flux across bands

        # stats
        std_rel = float(np.nanstd(rel))
        maxdiff = float(np.nanmax(rel) - np.nanmin(rel))

        # chi2 around median (use GP errors)
        med = np.nanmedian(fluxes_v)
        denom = (errs_v**2 + eps)
        chi2 = float(np.nansum(((fluxes_v - med)**2) / denom)) / max(1, nb - 1)

        score = float(1.0 / (1.0 + std_rel))

        out.update({
            "gp_achro_std": std_rel,
            "gp_achro_maxdiff": maxdiff,
            "gp_achro_chi2": chi2,
            "gp_achro_score": score
        })
        return out
    except Exception:
        return out

# ---------------------------------------------------------------------
# 2) Post-peak smoothness features
# ---------------------------------------------------------------------
def gp_post_peak_smoothness_features(df_obj, gp_model, gp_flux, peak_time, anchor_band=None, post_window=None, n_grid=1000):
    """
    Metrics that quantify smoothness / irregularity of GP mean after peak.
    Returns dict:
      - gp_post_slope_median: median absolute slope (post-peak)
      - gp_post_slope_std: std of slopes
      - gp_post_curvature_rms: RMS of second derivative (curvature)
      - gp_post_turns: number of sign changes in first derivative (indicative of wiggles)
      - gp_post_npoints: number of grid points in post region
    Inputs:
      post_window: time window after peak to consider (if None, use df_obj max - peak_time)
    """
    out = {
        "gp_post_slope_median": np.nan,
        "gp_post_slope_std": np.nan,
        "gp_post_curvature_rms": np.nan,
        "gp_post_turns": np.nan,
        "gp_post_npoints": 0
    }
    try:
        if gp_model is None or "gp" not in gp_model or not np.isfinite(peak_time):
            return out
        gp = gp_model["gp"]

        df = _normalize_columns(df_obj)
        try:
            t_max = float(df["Time (MJD)"].max())
        except Exception:
            return out

        if post_window is None:
            post_window = max(50.0, t_max - peak_time)

        t_grid, mean, var = gp_predict_band(gp, gp_flux, anchor_band if anchor_band is not None else select_anchor_band(df), peak_time, peak_time + post_window, n_grid=n_grid, margin_pre=0, margin_post=0)
        if mean is None or len(mean) < 6 or np.all(~np.isfinite(mean)):
            return out

        # ensure post portion starts at peak index
        # find index closest to peak_time
        idx_peak = int(np.nanargmax(mean))  # mean's own peak index
        # define post region from idx_peak+1 to end
        if idx_peak >= len(mean) - 2:
            return out

        t_post = t_grid[idx_peak+1:]
        m_post = mean[idx_peak+1:]

        # derivatives
        dt = np.diff(t_post)
        dy = np.diff(m_post)
        # slopes at midpoints
        slopes = dy / (dt + 1e-12)
        if len(slopes) < 2:
            return out

        # second derivative approx
        d2 = np.diff(slopes) / (dt[1:] + 1e-12)  # length len(slopes)-1

        slope_med = float(np.nanmedian(np.abs(slopes)))
        slope_std = float(np.nanstd(slopes))
        curvature_rms = float(np.sqrt(np.nanmean((d2[np.isfinite(d2)])**2))) if np.any(np.isfinite(d2)) else np.nan

        # turns: number of sign changes in slopes
        ssign = np.sign(slopes)
        # count non-zero sign changes
        valid_sign_idx = np.where(ssign != 0)[0]
        turns = 0
        if valid_sign_idx.size > 1:
            ssign2 = ssign[valid_sign_idx]
            turns = int(np.sum(np.abs(np.diff(ssign2)) > 0))

        out.update({
            "gp_post_slope_median": slope_med,
            "gp_post_slope_std": slope_std,
            "gp_post_curvature_rms": curvature_rms,
            "gp_post_turns": turns,
            "gp_post_npoints": int(len(t_post))
        })
        return out
    except Exception:
        return out

# ---------------------------------------------------------------------
# 3) Fluence features
# ---------------------------------------------------------------------
def gp_fluence_features(df_obj, gp_model, gp_flux, peak_time, anchor_band=None, pre_window=None, post_window=None, bands=None, n_grid=1200):
    """
    Fluence (integrated flux) metrics using GP mean.
    Returns dict:
      - gp_fluence_total_anchor: trapezoidal integral of mean (anchor band) over window
      - gp_fluence_pre_anchor: integral pre-peak
      - gp_fluence_post_anchor: integral post-peak
      - gp_fluence_ratio_post_pre_anchor: post/pre ratio
      - gp_fluence_band_<b>: per-band fluence for b in bands (g,r,i default)
    Inputs:
      pre_window/post_window: time windows (if None, use default 50/100 or derived from data)
      bands: list of bands to compute per-band fluence (defaults to ['g','r','i'])
    """
    out = {
        "gp_fluence_total_anchor": np.nan,
        "gp_fluence_pre_anchor": np.nan,
        "gp_fluence_post_anchor": np.nan,
        "gp_fluence_ratio_post_pre_anchor": np.nan
    }
    try:
        if gp_model is None or "gp" not in gp_model or not np.isfinite(peak_time):
            return out
        gp = gp_model["gp"]
        df = _normalize_columns(df_obj)

        # windows
        try:
            t_min_data = float(df["Time (MJD)"].min())
            t_max_data = float(df["Time (MJD)"].max())
        except Exception:
            t_min_data, t_max_data = peak_time - 50.0, peak_time + 100.0

        if pre_window is None:
            pre_window = min( max(20.0, peak_time - t_min_data), 50.0 )
        if post_window is None:
            post_window = min( max(40.0, t_max_data - peak_time), 100.0 )

        t_start = peak_time - pre_window
        t_end = peak_time + post_window
        # anchor band selection
        if anchor_band is None:
            anchor_band = select_anchor_band(df) or "g"

        # predict dense grid for anchor band
        t_grid, mean, var = gp_predict_band(gp, gp_flux, anchor_band, t_start, t_end, n_grid=n_grid, margin_pre=0, margin_post=0)
        if mean is None or len(mean) < 5 or np.all(~np.isfinite(mean)):
            return out

        # split to pre/post at nearest index to peak_time
        idx_peak = int(np.nanargmax(mean))
        # However if grid not centered on peak_time, find nearest index by time
        idx_nearest = int(np.argmin(np.abs(t_grid - peak_time)))
        # take pre as t_grid[:idx_nearest], post idx_nearest:
        t_pre = t_grid[:idx_nearest+1]
        m_pre = mean[:idx_nearest+1]
        t_post = t_grid[idx_nearest:]
        m_post = mean[idx_nearest:]

        # integrals (trapz)
        try:
            flu_pre = float(np.trapz(np.clip(m_pre, 0, None), t_pre))
            flu_post = float(np.trapz(np.clip(m_post, 0, None), t_post))
            flu_total = float(np.trapz(np.clip(mean, 0, None), t_grid))
        except Exception:
            flu_pre = flu_post = flu_total = np.nan

        ratio = np.nan
        if np.isfinite(flu_pre) and flu_pre > 0:
            ratio = float(flu_post / (flu_pre + 1e-12))

        out["gp_fluence_total_anchor"] = flu_total
        out["gp_fluence_pre_anchor"] = flu_pre
        out["gp_fluence_post_anchor"] = flu_post
        out["gp_fluence_ratio_post_pre_anchor"] = ratio

        # per-band fluence for important bands (default g,r,i)
        if bands is None:
            bands = ["g", "r", "i"]
        for b in bands:
            try:
                tg, mg, vg = gp_predict_band(gp, gp_flux, b, t_start, t_end, n_grid=min(800, n_grid))
                if mg is None or len(mg) < 3:
                    out[f"gp_fluence_band_{b}"] = np.nan
                else:
                    out[f"gp_fluence_band_{b}"] = float(np.trapz(np.clip(mg, 0, None), tg))
            except Exception:
                out[f"gp_fluence_band_{b}"] = np.nan

        return out
    except Exception:
        return out



# ======================================================================
# Decay alpha from GP mean
# ======================================================================

def estimate_decay_alpha_from_gp(gp, gp_flux, band, t_min, t_max, fit_start_offset=1.0, min_points=5):
    """
    Given a gp and band, predict mean on dense grid, find peak and fit
    decay law: f = A * (t - t0)^(-alpha) on post-peak times.
    Returns alpha, alpha_err, r2
    """
    try:
        t_grid, mean, var = gp_predict_band(gp, gp_flux, band, t_min, t_max, n_grid=1200)
        if mean is None or len(mean) < min_points or np.all(~np.isfinite(mean)):
            return np.nan, np.nan, np.nan

        idx_peak = int(np.nanargmax(mean))
        t_peak = t_grid[idx_peak]
        # consider post-peak region (avoid immediate peak point)
        t_post = t_grid[idx_peak + 1:]
        f_post = mean[idx_peak + 1:]

        mask = np.isfinite(t_post) & np.isfinite(f_post) & (f_post > 0)
        if mask.sum() < min_points:
            return np.nan, np.nan, np.nan

        t_post, f_post = t_post[mask], f_post[mask]
        # use (t - t_peak + eps)
        x = t_post - t_peak + 1e-6
        y = f_post

        # Fit log-log: log y = log A - alpha * log x
        logx = np.log(x)
        logy = np.log(y)

        # linear fit
        coef = np.polyfit(logx, logy, 1)
        alpha_est = float(-coef[0])
        # compute r2
        pred_logy = np.polyval(coef, logx)
        ss_res = np.sum((logy - pred_logy)**2)
        ss_tot = np.sum((logy - np.mean(logy))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        return alpha_est, np.nan, r2
    except Exception:
        return np.nan, np.nan, np.nan

# ======================================================================
# Single-object processor (module-level for pickling)
# ======================================================================
def _process_single_object(item):
    """
    item: (object_id, df_obj)
    returns: dict of GP-based features (ONE ROW)
    """
    oid, df_obj = item
    row = {"object_id": oid}
    t0 = time.time()

    try:
        # -------------------------------------------------
        # Normalize columns
        # -------------------------------------------------
        df_local = _normalize_columns(df_obj)

        if "split" in df_local.columns:
            row["split"] = df_local["split"].iloc[0]

        # -------------------------------------------------
        # Prepare GP input
        # -------------------------------------------------
        gp_input = prepare_gp_inputs_mallorn(df_local)
        if gp_input is None:
            row.update({
                "gp_success": 0,
                "gp_error": "insufficient_data"
            })
            return row

        # -------------------------------------------------
        # Fit GP
        # -------------------------------------------------
        gp_model = fit_george_gp_mallorn(gp_input)
        if gp_model is None:
            row.update({
                "gp_success": 0,
                "gp_error": "gp_fit_failed"
            })
            return row

        gp = gp_model["gp"]
        gp_flux = gp_input["flux"]

        # -------------------------------------------------
        # PEAK FEATURES
        # -------------------------------------------------
        anchor_band = select_anchor_band(df_local)
        peak_feats = gp_peak_features(
            df_local,
            gp_model,
            gp_flux,
            anchor_band=anchor_band
        )
        row.update(peak_feats)

        # -------------------------------------------------
        # RISE / FADE / ASYMMETRY
        # -------------------------------------------------
        rf_feats = gp_rise_fade_features(
            df_local,
            gp_model,
            gp_flux,
            peak_feats["gp_peak_time"],
            peak_feats["gp_anchor_band"]
        )
        row.update(rf_feats)

        # -------------------------------------------------
        # GP HYPERPARAMETERS
        # -------------------------------------------------
        row.update(gp_hyperparameter_features(gp_model))

        # -------------------------------------------------
        # CI-BASED VARIANCE FEATURES
        # -------------------------------------------------
        try:
            if peak_feats["gp_anchor_band"] is not None and np.isfinite(peak_feats["gp_peak_time"]):
                t_min = float(df_local["Time (MJD)"].min())
                t_max = float(df_local["Time (MJD)"].max())

                t_grid, mean, var = gp_predict_band(
                    gp,
                    gp_flux,
                    peak_feats["gp_anchor_band"],
                    t_min,
                    t_max,
                    n_grid=800
                )

                if var is not None and len(var) > 0:
                    std = np.sqrt(np.clip(var, 0, None))
                    pct = np.nanpercentile(std[np.isfinite(std)], [10, 50, 90])

                    row["gp_var_p10"], row["gp_var_p50"], row["gp_var_p90"] = map(float, pct)

                    idx_p = int(np.nanargmax(mean))
                    pre = std[:idx_p]
                    post = std[idx_p:]

                    row["gp_var_pre_p50"] = np.nanpercentile(pre, 50) if len(pre) else np.nan
                    row["gp_var_post_p50"] = np.nanpercentile(post, 50) if len(post) else np.nan
                    row["gp_var_ratio_post_pre"] = (
                        (row["gp_var_post_p50"] + 1e-12) /
                        (row["gp_var_pre_p50"] + 1e-12)
                    )
        except Exception:
            pass

        # -------------------------------------------------
        # GP COLOR FEATURES
        # -------------------------------------------------
        for b1, b2 in [("g", "r"), ("r", "i"), ("g", "i")]:
            try:
                row.update(
                    gp_color_features(
                        df_local,
                        gp_model,
                        gp_flux,
                        peak_feats["gp_peak_time"],
                        b1,
                        b2,
                        rf_feats["gp_rise_time"],
                        rf_feats["gp_fade_time"]
                    )
                )
            except Exception:
                pass

        # -------------------------------------------------
        # DECAY ALPHA (POWER-LAW)
        # -------------------------------------------------
        try:
            if peak_feats["gp_anchor_band"] and np.isfinite(peak_feats["gp_peak_time"]):
                alpha, _, r2 = estimate_decay_alpha_from_gp(
                    gp,
                    gp_flux,
                    peak_feats["gp_anchor_band"],
                    df_local["Time (MJD)"].min(),
                    df_local["Time (MJD)"].max()
                )
                row["gp_decay_alpha"] = alpha
                row["gp_decay_alpha_r2"] = r2
        except Exception:
            row["gp_decay_alpha"] = np.nan
            row["gp_decay_alpha_r2"] = np.nan

        # -------------------------------------------------
        # NEW (A) ACHROMATICITY
        # -------------------------------------------------
        try:
            achro = gp_achromaticity_features(
                df_local,
                gp_model,
                gp_flux,
                peak_feats["gp_peak_time"]
            )
            row.update(achro)
        except Exception:
            pass

        # -------------------------------------------------
        # NEW (B) POST-PEAK SMOOTHNESS
        # -------------------------------------------------
        try:
            smooth = gp_post_peak_smoothness_features(
                df_local,
                gp_model,
                gp_flux,
                peak_feats["gp_peak_time"],
                peak_feats["gp_anchor_band"]
            )
            row.update(smooth)
        except Exception:
            pass

        # -------------------------------------------------
        # NEW (C) FLUENCE FEATURES
        # -------------------------------------------------
        try:
            flu = gp_fluence_features(
                df_local,
                gp_model,
                gp_flux,
                peak_feats["gp_peak_time"],
                peak_feats["gp_anchor_band"]
            )
            row.update(flu)
        except Exception:
            pass

        # -------------------------------------------------
        # SUCCESS FLAG
        # -------------------------------------------------
        row["gp_success"] = 1
        row["gp_error"] = ""

    except Exception as e:
        row["gp_success"] = 0
        row["gp_error"] = str(e)

    finally:
        row["gp_time_sec"] = time.time() - t0
        try:
            gc.collect()
        except Exception:
            pass

    return row

# ======================================================================
# Main GP feature builder class (wraps DataLoader / Preprocessor)
# ======================================================================

class GPFeatureBuilder:
    def __init__(self, use_multiprocessing=True, n_workers=None):
        self.loader = DataLoader()
        self.preprocessor = Preprocessor()
        self.use_multiprocessing = use_multiprocessing
        self.n_workers = n_workers if n_workers is not None else max(1, cpu_count() - 1)

    def _process_split(self, split, mode="train", sample_frac=1.0):
        """
        Load split data via DataLoader, preprocess, group by object and process GP per object.
        Returns DataFrame of rows.
        """
        logger.info("Processing split %s (mode=%s)", split, mode)
        df = self.loader.load_split_data(split=split, mode=mode, sample_frac=sample_frac)
        if df is None or len(df) == 0:
            logger.warning("Empty data for split %s", split)
            return pd.DataFrame()

        df = self.preprocessor.preprocess_pipeline(df)

        objects = df["object_id"].unique()
        items = [(oid, df[df["object_id"] == oid].copy()) for oid in objects]

        rows = []
        if self.use_multiprocessing and len(items) > 1:
            workers = min(self.n_workers, len(items))
            with Pool(workers) as pool:
                for r in tqdm(pool.imap_unordered(_process_single_object, items), total=len(items), desc=f"GP {split}", ncols=100):
                    if r is not None:
                        rows.append(r)
        else:
            for it in tqdm(items, desc=f"GP {split}", ncols=100):
                r = _process_single_object(it)
                if r is not None:
                    rows.append(r)

        if not rows:
            return pd.DataFrame()

        df_rows = pd.DataFrame(rows)
        # ensure object_id type consistent
        df_rows["object_id"] = df_rows["object_id"].astype(object)
        # keep split column if not present (fill)
        if "split" not in df_rows.columns:
            df_rows["split"] = split
        return df_rows

    def create_gp_dataset(self, mode="train", sample_frac=1.0, save=True):
        """
        Build GP features across all splits for mode in {'train', 'test'}.
        Save to data/process/gp_{mode}_features.csv
        """
        logger.info("create_gp_dataset mode=%s", mode)
        splits = self.loader.get_splits(mode)
        all_rows = []
        for sp in splits:
            df_sp = self._process_split(sp, mode=mode, sample_frac=sample_frac)
            if df_sp is None or df_sp.empty:
                continue
            df_sp["split"] = sp
            all_rows.append(df_sp)

        if not all_rows:
            logger.warning("No GP features extracted for mode=%s", mode)
            return pd.DataFrame()

        out = pd.concat(all_rows, ignore_index=True)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_file = GP_TRAIN if mode == "train" else GP_TEST
        if save:
            out.to_csv(out_file, index=False)
            logger.info("Saved GP features to %s", out_file)
        return out

# ======================================================================
# Merge helpers
# ======================================================================

def _safe_merge_base_and_gp(base_path, gp_path, out_path, on=["object_id", "split"]):
    """
    Merge two csvs (base features and gp features) safely:
    - If gp_path doesn't exist, warn and copy base to out_path
    - Avoid columns *_y empty by preferring base values
    - Convert abs_mag_bin to numeric if exists and string
    """
    if not Path(base_path).exists():
        raise FileNotFoundError(f"Base features file not found: {base_path}")

    base = pd.read_csv(base_path)
    if not Path(gp_path).exists():
        logger.warning("GP features file not found: %s -- copying base to %s", gp_path, out_path)
        base.to_csv(out_path, index=False)
        return pd.read_csv(out_path)

    gp = pd.read_csv(gp_path)

    merged = base.merge(gp, on=on, how="left", suffixes=("", "_gp"))
    # If merge created columns with _gp duplicates, keep base if base non-null else gp
    for col in merged.columns:
        if col.endswith("_gp"):
            base_col = col[:-3]
            if base_col in merged.columns:
                merged[base_col] = merged[base_col].fillna(merged[col])
            else:
                # rename the gp-only column to base_col
                merged = merged.rename(columns={col: base_col})
            merged = merged.drop(columns=[col], errors="ignore")

    # convert abs_mag_bin if exists
    if "abs_mag_bin" in merged.columns:
        try:
            merged["abs_mag_bin"] = pd.to_numeric(merged["abs_mag_bin"], errors="coerce").fillna(-1).astype(int)
        except Exception:
            pass

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    logger.info("Saved merged features to %s", out_path)
    return merged

# ======================================================================
# Convenience API (top-level)
# ======================================================================

def create_gp_train_dataset(sample_frac=1.0, save=True, use_multiprocessing=True, n_workers=None):
    builder = GPFeatureBuilder(use_multiprocessing=use_multiprocessing, n_workers=n_workers)
    out = builder.create_gp_dataset(mode="train", sample_frac=sample_frac, save=save)
    return out

def create_gp_test_dataset(sample_frac=1.0, save=True, use_multiprocessing=True, n_workers=None):
    builder = GPFeatureBuilder(use_multiprocessing=use_multiprocessing, n_workers=n_workers)
    out = builder.create_gp_dataset(mode="test", sample_frac=sample_frac, save=save)
    return out

def merge_all_train_features():
    """Merge gp_train_features.csv with TRAIN_FEATURES into data/process/all_train_features.csv"""
    return _safe_merge_base_and_gp(TRAIN_FEATURES, GP_TRAIN, ALL_TRAIN)

def merge_all_test_features():
    """Merge gp_test_features.csv with TEST_FEATURES into data/process/all_test_features.csv"""
    return _safe_merge_base_and_gp(TEST_FEATURES, GP_TEST, ALL_TEST)

# ======================================================================
# if run as script: build both sets (small log only)
# ======================================================================
if __name__ == "__main__":
    logger.info("Running gp_features.py as script: building GP train + test and merging")
    create_gp_train_dataset(save=True)
    create_gp_test_dataset(save=True)
    merge_all_train_features()
    merge_all_test_features()
    logger.info("Done.")
