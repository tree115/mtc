"""
SHAP Analysis for TDE Mallorn
Explain model predictions and feature importance
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

from config import MODELS_DIR, OUTPUTS_DIR, TRAIN_FEATURES
from model import predict_with_calibration


class ShapAnalyzer:
    def __init__(self, model_path=None):
        """Initialize SHAP analyzer with trained model"""
        if model_path is None:
            # Find latest model
            model_files = list(MODELS_DIR.glob("tde_lgbm_best_*.pkl"))
            if not model_files:
                raise FileNotFoundError("No trained model found")
            model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            model_path = model_files[0]
        
        print(f" Loading model: {model_path.name}")
        self.model_data = joblib.load(model_path)
        self.models = self.model_data["models"]
        self.features = self.model_data["features"]
        self.threshold = self.model_data["threshold"]
        
        # Use first model for SHAP (or create ensemble explainer)
        self.explainer_model = self.models[0]
        
        # Create SHAP explainer
        self.explainer = shap.TreeExplainer(self.explainer_model)
        
        # Create output directory
        self.shap_dir = OUTPUTS_DIR / "shap_analysis"
        self.shap_dir.mkdir(parents=True, exist_ok=True)
    
    def load_training_data(self, sample_size=1000):
        """Load training data for SHAP background"""
        if not TRAIN_FEATURES.exists():
            raise FileNotFoundError(f"Train features not found: {TRAIN_FEATURES}")
        
        df = pd.read_csv(TRAIN_FEATURES)
        
        # Prepare features
        X = df[self.features] if all(f in df.columns for f in self.features) else None
        if X is None:
            raise ValueError("Features mismatch between model and data")
        
        # Sample for background
        if sample_size < len(X):
            X_background = X.sample(sample_size, random_state=42)
        else:
            X_background = X
        
        # Get true labels if available
        y = df['target'].values if 'target' in df.columns else None
        
        return X, X_background, y
    
    def compute_shap_values(self, X, save=True):
        """Compute SHAP values for given data"""
        print(" Computing SHAP values...")
        
        # For ensemble models, compute SHAP for each model and average
        all_shap_values = []
        for i, model in enumerate(self.models[:3]):  # Limit to 3 models for speed
            print(f"  Model {i+1}/{min(3, len(self.models))}")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            # For binary classification, shap_values[1] is for class 1 (TDE)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Get SHAP for positive class
            
            all_shap_values.append(shap_values)
        
        # Average SHAP values across models
        shap_values_avg = np.mean(all_shap_values, axis=0)
        
        if save:
            # Save SHAP values
            shap_path = self.shap_dir / "shap_values.npy"
            np.save(shap_path, shap_values_avg)
            print(f" SHAP values saved: {shap_path}")
        
        return shap_values_avg
    
    def plot_summary(self, X, shap_values, max_display=20):
        """Create SHAP summary plot"""
        print(" Creating SHAP summary plot...")
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, X,
            max_display=max_display,
            plot_type="dot",
            show=False
        )
        plt.tight_layout()
        
        save_path = self.shap_dir / "shap_summary.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f" Summary plot saved: {save_path}")
    
    def plot_bar(self, X, shap_values, max_display=20):
        """Create SHAP bar plot (mean absolute SHAP)"""
        print(" Creating SHAP bar plot...")
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X,
            max_display=max_display,
            plot_type="bar",
            show=False
        )
        plt.tight_layout()
        
        save_path = self.shap_dir / "shap_bar.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f" Bar plot saved: {save_path}")
    
    def plot_dependence(self, X, shap_values, feature_names=None, top_features=5):
        """Create dependence plots for top features"""
        print(" Creating dependence plots...")
        
        # Get top features by mean absolute SHAP
        shap_df = pd.DataFrame(shap_values, columns=X.columns)
        mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
        top_features_list = mean_abs_shap.head(top_features).index.tolist()
        
        for i, feature in enumerate(top_features_list):
            print(f"  Feature {i+1}/{top_features}: {feature}")
            
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(
                feature,
                shap_values,
                X,
                display_features=X,
                show=False
            )
            plt.tight_layout()
            
            save_path = self.shap_dir / f"shap_dependence_{feature}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f" Dependence plots saved to {self.shap_dir}")
    
    def analyze_specific_samples(self, X, shap_values, sample_indices=None, n_samples=5):
        """Analyze specific samples with force plots"""
        print(" Analyzing specific samples...")
        
        if sample_indices is None:
            # Find interesting samples: high confidence TDE, borderline, etc.
            probabilities, _ = predict_with_calibration(
                self.models, X, threshold=self.threshold, method='mean'
            )
            
            # Get top TDE predictions
            tde_indices = np.argsort(probabilities)[::-1][:n_samples]
            
            # Get borderline predictions
            borderline_mask = (probabilities > self.threshold - 0.1) & (probabilities < self.threshold + 0.1)
            borderline_indices = np.where(borderline_mask)[0][:n_samples]
            
            sample_indices = list(tde_indices) + list(borderline_indices)
        
        for idx in sample_indices[:10]:  # Limit to 10 samples
            prob = probabilities[idx] if 'probabilities' in locals() else None
            
            # Create force plot
            plt.figure(figsize=(12, 4))
            shap.force_plot(
                self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) 
                else self.explainer.expected_value,
                shap_values[idx, :],
                X.iloc[idx, :],
                matplotlib=True,
                show=False
            )
            plt.title(f"Sample {idx} - Probability: {prob:.3f}" if prob is not None else f"Sample {idx}")
            plt.tight_layout()
            
            save_path = self.shap_dir / f"shap_force_sample_{idx}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f" Force plots saved to {self.shap_dir}")
    
    def create_feature_importance_report(self, X, shap_values, save_csv=True):
        """Create detailed feature importance report"""
        print(" Creating feature importance report...")
        
        # Calculate SHAP-based importance
        shap_df = pd.DataFrame(shap_values, columns=X.columns)
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'shap_mean_abs': shap_df.abs().mean().values,
            'shap_std': shap_df.std().values,
            'shap_mean': shap_df.mean().values,
            'shap_min': shap_df.min().values,
            'shap_max': shap_df.max().values
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('shap_mean_abs', ascending=False)
        
        # Add correlation with target if available
        try:
            X_full, _, y = self.load_training_data()
            if y is not None and len(y) == len(X_full):
                for feature in importance_df['feature']:
                    if feature in X_full.columns:
                        corr = np.corrcoef(X_full[feature].fillna(0), y)[0, 1]
                        importance_df.loc[importance_df['feature'] == feature, 'target_correlation'] = corr
        except:
            pass
        
        if save_csv:
            csv_path = self.shap_dir / "shap_feature_importance.csv"
            importance_df.to_csv(csv_path, index=False)
            print(f" Feature importance report saved: {csv_path}")
        
        return importance_df
    
    def compare_with_model_importance(self, importance_df):
        """Compare SHAP importance with model's feature importance"""
        print(" Comparing SHAP vs Model importance...")
        
        # Get model feature importance
        model_importance = []
        for i, model in enumerate(self.models[:3]):
            imp = pd.DataFrame({
                'feature': self.features,
                f'importance_model_{i}': model.feature_importances_
            })
            model_importance.append(imp)
        
        # Merge importances
        merged = importance_df.copy()
        for i, imp_df in enumerate(model_importance):
            merged = merged.merge(imp_df, on='feature', how='left')
        
        # Calculate average model importance
        model_cols = [c for c in merged.columns if 'importance_model' in c]
        if model_cols:
            merged['model_importance_mean'] = merged[model_cols].mean(axis=1)
            merged['model_importance_std'] = merged[model_cols].std(axis=1)
        
        # Create comparison plot
        if 'model_importance_mean' in merged.columns:
            plt.figure(figsize=(10, 8))
            
            top_n = 20
            top_features = merged.head(top_n)
            
            x = np.arange(len(top_features))
            width = 0.35
            
            plt.bar(x - width/2, top_features['shap_mean_abs'], width, label='SHAP Importance')
            plt.bar(x + width/2, top_features['model_importance_mean'], width, label='Model Importance')
            
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.title('SHAP vs Model Feature Importance (Top 20)')
            plt.xticks(x, top_features['feature'], rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            
            save_path = self.shap_dir / "shap_vs_model_importance.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f" Comparison plot saved: {save_path}")
        
        return merged
    
    def run_full_analysis(self, sample_size=1000, max_display=20):
        """Run complete SHAP analysis"""
        print("="*80)
        print(" STARTING COMPLETE SHAP ANALYSIS")
        print("="*80)
        
        # 1. Load data
        X, X_background, y = self.load_training_data(sample_size)
        
        # 2. Compute SHAP values
        shap_values = self.compute_shap_values(X_background)
        
        # 3. Create plots
        self.plot_summary(X_background, shap_values, max_display)
        self.plot_bar(X_background, shap_values, max_display)
        self.plot_dependence(X_background, shap_values, top_features=5)
        
        # 4. Create reports
        importance_df = self.create_feature_importance_report(X_background, shap_values)
        comparison_df = self.compare_with_model_importance(importance_df)
        
        # 5. Analyze samples
        self.analyze_specific_samples(X_background, shap_values)
        
        print("\n" + "="*80)
        print(" SHAP ANALYSIS COMPLETED")
        print(f" Results saved in: {self.shap_dir}")
        print("="*80)
        
        return {
            'shap_values': shap_values,
            'importance_df': importance_df,
            'comparison_df': comparison_df,
            'X_background': X_background,
            'y': y
        }


def analyze_model_shap(model_path=None, sample_size=1000):
    """Convenience function to run SHAP analysis"""
    analyzer = ShapAnalyzer(model_path)
    results = analyzer.run_full_analysis(sample_size)
    return results


if __name__ == "__main__":
    # Run analysis when executed as script
    results = analyze_model_shap(sample_size=500)
