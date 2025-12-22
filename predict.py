"""
predict.py - Use best model strategy WITH THRESHOLD SELECTION
"""

import pandas as pd
import numpy as np
import joblib
import argparse
from datetime import datetime
from pathlib import Path
import sys

from config import MODELS_DIR, SUBMISSIONS_DIR, TEST_FEATURES
from model import predict_with_calibration, find_optimal_threshold
from dataset import create_test_dataset


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='TDE MALLORN Prediction')
    
    parser.add_argument('--threshold', type=float, default=None,
                       help='Custom threshold for prediction (overrides model threshold)')
    
    parser.add_argument('--model', type=str, default=None,
                       help='Specific model file to use (default: latest best model)')
    
    parser.add_argument('--method', type=str, default='median',
                       choices=['mean', 'median', 'weighted'],
                       help='Prediction aggregation method')
    
    parser.add_argument('--auto-threshold', action='store_true',
                       help='Auto-optimize threshold on validation data')
    
    parser.add_argument('--explore-thresholds', action='store_true',
                       help='Explore multiple thresholds and show stats')
    
    parser.add_argument('--target-positives', type=int, default=None,
                       help='Set threshold to get approximately N positive predictions')
    
    return parser.parse_args()


def load_validation_data_for_threshold(model_data):
    """
    Try to load validation data for threshold optimization
    """
    try:
        # Check if we have train features for validation
        from config import TRAIN_FEATURES
        
        if not TRAIN_FEATURES.exists():
            return None, None
        
        print("📊 Loading validation data for threshold optimization...")
        df_val = pd.read_csv(TRAIN_FEATURES)
        
        # Get features used by model
        features = model_data['features']
        
        # Prepare validation data
        X_val = df_val[features] if all(f in df_val.columns for f in features) else None
        
        if X_val is not None:
            # Get predictions on validation
            prob_val, _ = predict_with_calibration(
                model_data['models'], 
                X_val, 
                threshold=0.5, 
                method='mean'
            )
            
            # Get true labels if available
            y_val = df_val['target'].values if 'target' in df_val.columns else None
            
            return prob_val, y_val
        
    except Exception as e:
        print(f"⚠️  Could not load validation data: {e}")
    
    return None, None


def explore_thresholds(probabilities, model_threshold, y_true=None):
    """
    Explore multiple thresholds and show statistics
    """
    print("\n🔍 EXPLORING THRESHOLDS")
    print("="*60)
    
    thresholds_to_try = [
        0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5
    ]
    
    # Add model's optimal threshold
    if model_threshold not in thresholds_to_try:
        thresholds_to_try.append(model_threshold)
        thresholds_to_try.sort()
    
    results = []
    
    for t in thresholds_to_try:
        pred = (probabilities >= t).astype(int)
        n_positives = pred.sum()
        pos_rate = n_positives / len(pred)
        
        result = {
            'threshold': t,
            'positives': n_positives,
            'pos_rate': pos_rate,
            'is_model_optimal': abs(t - model_threshold) < 0.001
        }
        
        # If we have true labels, calculate metrics
        if y_true is not None:
            from sklearn.metrics import f1_score, precision_score, recall_score
            f1 = f1_score(y_true, pred, zero_division=0)
            precision = precision_score(y_true, pred, zero_division=0)
            recall = recall_score(y_true, pred, zero_division=0)
            
            result.update({
                'f1': f1,
                'precision': precision,
                'recall': recall
            })
        
        results.append(result)
    
    # Create DataFrame for display
    df_results = pd.DataFrame(results)
    
    print("\n📊 Threshold Analysis:")
    print("-" * 80)
    
    if y_true is not None:
        # With validation labels
        print(f"{'Threshold':>10} {'Positives':>10} {'Pos Rate':>10} {'F1':>8} {'Precision':>10} {'Recall':>10} {'Optimal':>8}")
        print("-" * 80)
        
        for _, row in df_results.iterrows():
            star = "★" if row['is_model_optimal'] else ""
            print(f"{row['threshold']:>10.3f} {row['positives']:>10} {row['pos_rate']:>10.2%} "
                  f"{row['f1']:>8.4f} {row['precision']:>10.4f} {row['recall']:>10.4f} {star:>8}")
    else:
        # Without validation labels
        print(f"{'Threshold':>10} {'Positives':>10} {'Pos Rate':>10} {'Optimal':>8}")
        print("-" * 80)
        
        for _, row in df_results.iterrows():
            star = "★" if row['is_model_optimal'] else ""
            print(f"{row['threshold']:>10.3f} {row['positives']:>10} {row['pos_rate']:>10.2%} {star:>8}")
    
    # Find threshold for target number of positives
    if args.target_positives:
        print(f"\n🎯 Targeting ~{args.target_positives} positives:")
        
        # Find threshold that gives closest to target
        df_results['diff'] = abs(df_results['positives'] - args.target_positives)
        best_idx = df_results['diff'].idxmin()
        best_row = df_results.loc[best_idx]
        
        print(f"   Threshold {best_row['threshold']:.3f} gives {best_row['positives']} positives "
              f"({best_row['pos_rate']:.2%})")
        
        if y_true is not None:
            print(f"   F1: {best_row['f1']:.4f}, Precision: {best_row['precision']:.4f}, "
                  f"Recall: {best_row['recall']:.4f}")
        
        return best_row['threshold']
    
    return None


def auto_optimize_threshold(probabilities, y_true):
    """
    Automatically find optimal threshold on validation data
    """
    if y_true is None:
        print("❌ Cannot auto-optimize: No validation labels available")
        return None
    
    print("\n🤖 AUTO-THRESHOLD OPTIMIZATION")
    print("="*60)
    
    # Find optimal threshold
    optimal_threshold, optimal_f1 = find_optimal_threshold(y_true, probabilities)
    
    print(f"✅ Optimal threshold: {optimal_threshold:.4f}")
    print(f"   F1 score: {optimal_f1:.4f}")
    
    # Show comparison with model threshold
    pred_model = (probabilities >= model_threshold).astype(int)
    f1_model = f1_score(y_true, pred_model, zero_division=0)
    
    print(f"\n📊 Comparison with model threshold ({model_threshold:.4f}):")
    print(f"   Model F1: {f1_model:.4f}")
    print(f"   Improvement: {optimal_f1 - f1_model:+.4f}")
    
    if optimal_f1 > f1_model:
        print("✨ New threshold improves F1!")
        return optimal_threshold
    else:
        print("📌 Keeping model threshold (better or equal)")
        return model_threshold


def main():
    global args, model_threshold
    args = parse_args()
    
    print("=" * 80)
    print("🚀 TDE MALLORN PREDICTION - THRESHOLD SELECTION")
    print("=" * 80)
    
    # -------------------------------------------------
    # Load model
    # -------------------------------------------------
    if args.model:
        # Use specified model
        model_path = MODELS_DIR / args.model
        if not model_path.exists():
            raise FileNotFoundError(f"❌ Model not found: {model_path}")
    else:
        # Find best model
        model_files = list(MODELS_DIR.glob("tde_lgbm_best_*.pkl"))
        if not model_files:
            # Fallback to old naming
            model_files = list(MODELS_DIR.glob("tde_lgbm_strong_*.pkl"))
        
        if not model_files:
            raise FileNotFoundError("❌ No trained model found")
        
        # Sort by creation time
        model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        model_path = model_files[0]
    
    print(f"📥 Loading model: {model_path.name}")
    data = joblib.load(model_path)
    
    models = data["models"]
    feature_names = data["features"]
    model_threshold = data["threshold"]
    strategy = data.get("strategy", "unknown")
    
    print(f"🧠 Strategy      : {strategy}")
    print(f"📏 Features      : {len(feature_names)}")
    print(f"🎯 Model threshold: {model_threshold:.4f}")
    print(f"📈 OOF F1        : {data['oof_f1']:.4f}")
    
    # -------------------------------------------------
    # Load test data
    # -------------------------------------------------
    if not TEST_FEATURES.exists():
        print("📦 Creating test_features.csv ...")
        test_df = create_test_dataset(save=True)
    else:
        print("📦 Loading test_features.csv")
        test_df = pd.read_csv(TEST_FEATURES)
    
    if "object_id" not in test_df.columns:
        raise ValueError("❌ object_id missing in test features")
    
    object_ids = test_df["object_id"].values
    
    # -------------------------------------------------
    # Align features
    # -------------------------------------------------
    missing_features = set(feature_names) - set(test_df.columns)
    extra_features = set(test_df.columns) - set(feature_names)
    
    if missing_features:
        print(f"⚠️  Adding missing {len(missing_features)} features (set to 0)")
        for col in missing_features:
            test_df[col] = 0
    
    if extra_features:
        print(f"⚠️  Removing {len(extra_features)} extra features not in training")
    
    X_test = test_df[feature_names]
    print(f"📊 Test samples  : {len(X_test)}")
    
    # -------------------------------------------------
    # Get predictions
    # -------------------------------------------------
    print(f"\n🔮 Predicting with {args.method} aggregation...")
    prob, _ = predict_with_calibration(
        models, X_test, threshold=0.5, method=args.method
    )
    
    # -------------------------------------------------
    # Threshold selection logic
    # -------------------------------------------------
    final_threshold = model_threshold
    
    if args.auto_threshold:
        # Try to auto-optimize threshold
        prob_val, y_val = load_validation_data_for_threshold(data)
        if y_val is not None:
            final_threshold = auto_optimize_threshold(prob_val, y_val)
        else:
            print("❌ Cannot auto-optimize: No validation data available")
    
    elif args.explore_thresholds:
        # Explore multiple thresholds
        prob_val, y_val = load_validation_data_for_threshold(data)
        target_threshold = explore_thresholds(prob, model_threshold, y_val)
        
        if target_threshold is not None and args.target_positives:
            final_threshold = target_threshold
            print(f"\n✅ Using threshold for target positives: {final_threshold:.4f}")
        else:
            # Ask user to choose
            print("\n🎯 Choose threshold (or press Enter for model threshold):")
            try:
                user_input = input(f"   Threshold [{model_threshold:.4f}]: ").strip()
                if user_input:
                    final_threshold = float(user_input)
                    if not 0 <= final_threshold <= 1:
                        raise ValueError("Threshold must be between 0 and 1")
            except ValueError as e:
                print(f"⚠️  Invalid input: {e}, using model threshold")
                final_threshold = model_threshold
            except KeyboardInterrupt:
                print("\n⏹️  Using model threshold")
                final_threshold = model_threshold
    
    elif args.threshold is not None:
        # Use custom threshold from command line
        final_threshold = args.threshold
        if not 0 <= final_threshold <= 1:
            print(f"⚠️  Threshold must be between 0 and 1, using {model_threshold:.4f}")
            final_threshold = model_threshold
        else:
            print(f"✅ Using custom threshold: {final_threshold:.4f}")
    
    elif args.target_positives:
        # Find threshold for target number of positives
        print(f"\n🎯 Finding threshold for ~{args.target_positives} positives...")
        
        # Binary search for threshold
        low, high = 0.0, 1.0
        for _ in range(20):  # Max 20 iterations
            mid = (low + high) / 2
            n_pos = (prob >= mid).sum()
            
            if abs(n_pos - args.target_positives) < 5:  # Within 5 of target
                final_threshold = mid
                break
            elif n_pos > args.target_positives:
                low = mid
            else:
                high = mid
        else:
            final_threshold = (low + high) / 2
        
        n_pos_final = (prob >= final_threshold).sum()
        print(f"   Threshold {final_threshold:.4f} gives {n_pos_final} positives")
    
    # Apply final threshold
    pred = (prob >= final_threshold).astype(int)
    
    print(f"\n✅ Final threshold: {final_threshold:.4f}")
    print(f"   TDE predicted: {pred.sum():,} ({pred.mean():.2%})")
    
    # -------------------------------------------------
    # Save submission
    # -------------------------------------------------
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create informative filename
    if args.threshold is not None:
        th_str = f"custom_{final_threshold:.3f}"
    elif args.auto_threshold:
        th_str = f"auto_{final_threshold:.3f}"
    elif args.target_positives:
        th_str = f"target{args.target_positives}_{final_threshold:.3f}"
    else:
        th_str = f"model_{final_threshold:.3f}"
    
    sub_path = SUBMISSIONS_DIR / f"submission_{strategy}_{th_str}_{ts}.csv"
    
    submission = pd.DataFrame({
        "object_id": object_ids,
        "predicted": pred
    })
    
    submission.to_csv(sub_path, index=False)
    
    # -------------------------------------------------
    # Detailed stats
    # -------------------------------------------------
    print("\n📊 FINAL PREDICTION STATS")
    print("-" * 60)
    print(f"   Total objects        : {len(pred):,}")
    print(f"   TDE predicted        : {pred.sum():,} ({pred.mean():.2%})")
    print(f"   Probability range    : [{prob.min():.4f}, {prob.max():.4f}]")
    print(f"   Probability mean     : {prob.mean():.4f}")
    print(f"   Probability median   : {np.median(prob):.4f}")
    print(f"   Threshold used       : {final_threshold:.4f}")
    print(f"   Aggregation method   : {args.method}")
    
    # Probability distribution
    print("\n📈 Probability distribution:")
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, bin_edges = np.histogram(prob, bins=bins)
    
    for i in range(len(bins)-1):
        pct = hist[i] / len(prob) * 100
        bar = "█" * int(pct / 3)  # Scale for display
        threshold_indicator = "← threshold" if bin_edges[i] <= final_threshold <= bin_edges[i+1] else ""
        print(f"   {bins[i]:.1f}-{bins[i+1]:.1f}: {bar:<40} {hist[i]:,} ({pct:.1f}%) {threshold_indicator}")
    
    print("\n" + "=" * 80)
    print("✅ PREDICTION COMPLETED")
    print(f"📤 Submission saved: {sub_path}")
    print("=" * 80)
    
    # Save prediction probabilities for analysis
    prob_path = SUBMISSIONS_DIR / f"probabilities_{strategy}_{ts}.csv"
    pd.DataFrame({
        "object_id": object_ids,
        "probability": prob,
        "predicted": pred,
        "threshold": final_threshold
    }).to_csv(prob_path, index=False)
    print(f"💾 Probabilities saved: {prob_path}")


if __name__ == "__main__":
    main()