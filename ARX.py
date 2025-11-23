import os
import numpy as np
import pandas as pd
import pickle
import joblib
from typing import Dict, List, Tuple, Optional, Union
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ARXFeatureEngineering(BaseEstimator, TransformerMixin):
    """Feature engineering transformer for ARX model"""
    
    def __init__(self, H: int = 2, low_var_thr: float = 1e-4):
        self.H = H
        self.low_var_thr = low_var_thr
        self.feature_names_ = None
        
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._build_features(X)
    
    def _build_features(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """Build modeling features with temporal components"""
        df = df_in.sort_values(["user_id", "week"]).copy()
        
        # Create rolling means
        for col in ["v_true_mean", "a_true_mean"]:
            rm2 = f"{col}_rm2"
            if rm2 not in df.columns:
                df[rm2] = df.groupby("user_id")[col].transform(
                    lambda s: s.rolling(2, min_periods=1).mean()
                )
            
            # Future targets
            tgt_level = f"{col}_level_t+{self.H}"
            if tgt_level not in df.columns:
                df[tgt_level] = df.groupby("user_id")[rm2].shift(-self.H)
            
            # Direction labels
            tgt_dir = f"{col}_dir_t+{self.H}"
            if tgt_dir not in df.columns:
                future_delta = df.groupby("user_id")[rm2].shift(-self.H) - df[rm2]
                thr = future_delta.abs().quantile(0.35) 
                df[tgt_dir] = pd.cut(
                    future_delta,
                    bins=[-np.inf, -thr, thr, np.inf],
                    labels=[-1, 0, 1]
                ).astype("float")
        
        # Anchors (no leakage)
        df["v_last"] = df.groupby("user_id")["v_true_mean_rm2"].shift(1)
        df["a_last"] = df.groupby("user_id")["a_true_mean_rm2"].shift(1)
        
        # Temporal statistics
        for col in ["v_true_mean_rm2", "a_true_mean_rm2"]:
            df[f"{col}_ema3"] = df.groupby("user_id")[col].transform(
                lambda s: s.ewm(span=3, adjust=False).mean()
            )
            df[f"{col}_slope3"] = df.groupby("user_id")[col].transform(
                lambda s: s.diff().rolling(3, min_periods=1).mean()
            )
            df[f"{col}_std4"] = df.groupby("user_id")[col].transform(
                lambda s: s.rolling(4, min_periods=1).std()
            )
        
        return df

class PersonalizedARXModel:
    """Personalized ARX model with global and user-specific parameters"""
    
    def __init__(self, alpha_global: float = 1.0, alpha_user: float = 5.0, 
                 shrink: int = 50, user_col: str = 'user_id'):
        self.alpha_global = alpha_global
        self.alpha_user = alpha_user
        self.shrink = shrink
        self.user_col = user_col
        self.scaler = StandardScaler()
        self.global_model = None
        self.user_models = {}
        self.feature_names = None
        
    def fit(self, df: pd.DataFrame, y_col: str, last_col: str, X_cols: List[str]):
        """Fit personalized ARX model"""
        # Prepare data
        Z = df[[last_col] + X_cols].fillna(0.0).values
        y = df[y_col].values
        
        # Scale features 
        Z[:, 1:] = self.scaler.fit_transform(Z[:, 1:])
        
        # Fit global model
        self.global_model = Ridge(alpha=self.alpha_global).fit(Z, y)
        g_coef, g_int = self.global_model.coef_, self.global_model.intercept_
        
        # Fit user-specific models
        for u, d in df.groupby(self.user_col):
            Zu = d[[last_col] + X_cols].fillna(0.0).values
            yu = d[y_col].values
            Zu[:, 1:] = self.scaler.transform(Zu[:, 1:])
            
            user_ridge = Ridge(alpha=self.alpha_user).fit(Zu, yu)
            n = len(d)
            w = n / (n + self.shrink)
            
            # Shrinkage towards global parameters
            self.user_models[u] = {
                'intercept': w * user_ridge.intercept_ + (1-w) * g_int,
                'coef': w * user_ridge.coef_ + (1-w) * g_coef
            }
        
        self.feature_names = [last_col] + X_cols
        return self
    
    def predict(self, df: pd.DataFrame, y_last_col: str, X_cols: List[str]) -> np.ndarray:
        """Make predictions using personalized models"""
        Z = df[[y_last_col] + X_cols].fillna(0.0).values
        Z[:, 1:] = self.scaler.transform(Z[:, 1:])
        
        predictions = []
        for i, u in enumerate(df[self.user_col]):
            if u in self.user_models:
                model = self.user_models[u]
            else:
                # Fallback to global model for unseen users
                model = {
                    'intercept': self.global_model.intercept_,
                    'coef': self.global_model.coef_
                }
            
            pred = model['intercept'] + Z[i].dot(model['coef'])
            predictions.append(pred)
        
        return np.array(predictions)

class ARXPipeline:
    """Complete ARX pipeline for mental health prediction"""
    
    def __init__(self, feature_sets: Dict[str, List[str]], H: int = 2):
        self.feature_sets = feature_sets
        self.H = H
        self.feature_engineer = ARXFeatureEngineering(H=H)
        self.models = {}
        self.performance_metrics = {}
        
    def prepare_data(self, cx_path: str, llm_path: str) -> pd.DataFrame:
        """Load and merge datasets"""
        logger.info("Loading datasets...")
        cx = pd.read_csv(cx_path)
        llm = pd.read_csv(llm_path)
        
        # Rename columns
        cx = cx.rename(columns={
            "valence_z_mean": "v_true_mean",
            "arousal_z_mean": "a_true_mean", 
            "radius_z_mean": "r_true_mean",
            "n_posts": "posts_true"
        })
        
        # Convert dates
        cx["week"] = pd.to_datetime(cx["week"])
        llm["week"] = pd.to_datetime(llm["week"])
        
        # Merge datasets
        merged = pd.merge(cx, llm, on=["user_id", "week"], how="inner")
        merged = merged.sort_values(by=["user_id", "week"]).reset_index(drop=True)
        
        if "posts" in merged.columns and "posts_true" in merged.columns:
            merged = merged.drop(columns=["posts"])
            
        merged = self.add_ema_features(merged, alpha=0.5)
        
        logger.info(f"Merged dataset shape: {merged.shape}")
        return merged
    
    def add_ema_features(self, df: pd.DataFrame, alpha: float = 0.5) -> pd.DataFrame:
        """Add exponential moving average features"""
        ema_source_cols = [
            "v_true_mean", "a_true_mean", "r_true_mean", 
            "v_hat_mean", "a_hat_mean", "r_hat_mean", "posts_true"
        ]
        
        ema_source_cols = [c for c in ema_source_cols if c in df.columns]
        
        for c in ema_source_cols:
            df[f"{c}_ema3"] = df.groupby("user_id")[c].transform(
                lambda s: s.ewm(alpha=alpha, adjust=False).mean()
            )
        
        return df
    
    def filter_low_variance_features(self, df: pd.DataFrame, 
                                   feature_sets: Dict[str, List[str]], 
                                   thr: float = 1e-4) -> Dict[str, List[str]]:
        """Filter out low variance features"""
        filtered_sets = {}
        for name, feats in feature_sets.items():
            filtered_sets[name] = [
                f for f in feats 
                if f in df.columns and df[f].std() > thr
            ]
        return filtered_sets
    
    def train_models(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Train ARX models for all feature sets"""
        logger.info("Training ARX models...")
        
        for name, feats in self.feature_sets.items():
            logger.info(f"Training {name} model...")
            
            # Build features
            df_feat = self.feature_engineer.transform(df)
            
            # Define model features
            extra_feats = [
                "v_last", "a_last",
                "v_true_mean_rm2_ema3", "a_true_mean_rm2_ema3",
                "v_true_mean_rm2_slope3", "a_true_mean_rm2_slope3", 
                "v_true_mean_rm2_std4", "a_true_mean_rm2_std4"
            ]
            if "posts_true" in df_feat.columns:
                extra_feats.append("posts_true")
            
            mr_feats = list(dict.fromkeys(
                [f for f in feats if f in df_feat.columns] + extra_feats
            ))
            
            # Filter variance and drop NaNs
            mr_feats = [f for f in mr_feats if df_feat[f].std() > 1e-4]
            
            target_cols = [f"v_true_mean_level_t+{self.H}", f"a_true_mean_level_t+{self.H}"]
            df_clean = df_feat.dropna(subset=["v_last", "a_last"] + target_cols).copy()
            
            # Train valence and arousal models
            v_model = PersonalizedARXModel()
            a_model = PersonalizedARXModel()
            
            v_model.fit(df_clean, f"v_true_mean_level_t+{self.H}", 
                       "v_true_mean_rm2", mr_feats)
            a_model.fit(df_clean, f"a_true_mean_level_t+{self.H}", 
                       "a_true_mean_rm2", mr_feats)
            
            self.models[name] = {
                'valence_model': v_model,
                'arousal_model': a_model,
                'features': mr_feats,
                'data': df_clean
            }
        
        return self.models
    
    def evaluate_models(self) -> pd.DataFrame:
        """Evaluate all trained models"""
        logger.info("Evaluating models...")
        
        results = []
        for name, model_dict in self.models.items():
            v_model = model_dict['valence_model']
            a_model = model_dict['arousal_model']
            features = model_dict['features']
            df_clean = model_dict['data']
            
            # Make predictions
            v_pred = v_model.predict(df_clean, "v_true_mean_rm2", features)
            a_pred = a_model.predict(df_clean, "a_true_mean_rm2", features)
            
            # Calculate metrics
            y_v = df_clean[f"v_true_mean_level_t+{self.H}"].values
            y_a = df_clean[f"a_true_mean_level_t+{self.H}"].values
            
            mae_v = np.mean(np.abs(y_v - v_pred))
            mae_a = np.mean(np.abs(y_a - a_pred))
            
            # Direction accuracy
            v_delta = v_pred - df_clean["v_true_mean_rm2"].values
            a_delta = a_pred - df_clean["a_true_mean_rm2"].values
            
            thr_v = np.nanquantile(np.abs(v_delta), 0.35)
            thr_a = np.nanquantile(np.abs(a_delta), 0.35)
            
            v_hat = np.where(v_delta > thr_v, 1, np.where(v_delta < -thr_v, -1, 0))
            a_hat = np.where(a_delta > thr_a, 1, np.where(a_delta < -thr_a, -1, 0))
            
            dir_v_true = df_clean[f"v_true_mean_dir_t+{self.H}"].astype(int).values
            dir_a_true = df_clean[f"a_true_mean_dir_t+{self.H}"].astype(int).values
            
            dir_acc_v = np.mean(v_hat == dir_v_true)
            dir_acc_a = np.mean(a_hat == dir_a_true)
            
            results.append([name, len(features), mae_v, mae_a, dir_acc_v, dir_acc_a])
        
        summary = pd.DataFrame(results, 
                             columns=['FeatureSet', 'n_feats', 'MAE_V', 'MAE_A', 
                                    'DirAcc_V', 'DirAcc_A'])
        
        self.performance_metrics = summary
        return summary
    
    def rolling_evaluation(self, df: pd.DataFrame, feature_set: str, 
                          min_len: int = 12, train_frac: float = 0.7) -> Tuple[pd.DataFrame, Dict]:
        """Perform rolling evaluation for a specific feature set"""
        logger.info(f"Running rolling evaluation for {feature_set}...")
        
        if feature_set not in self.models:
            raise ValueError(f"Model {feature_set} not trained yet")
        
        features = self.models[feature_set]['features']
        df_feat = self.feature_engineer.transform(df)
        
        target_cols = [f"v_true_mean_level_t+{self.H}", f"a_true_mean_level_t+{self.H}"]
        df_clean = df_feat.dropna(subset=["v_last", "a_last"] + target_cols).copy()
        
        rows = []
        for user_id, user_data in df_clean.groupby('user_id'):
            user_data = user_data.sort_values('week')
            n = len(user_data)
            
            if n < min_len:
                continue
            
            cut = int(round(train_frac * n))
            train_data, test_data = user_data.iloc[:cut].copy(), user_data.iloc[cut:].copy()
            
            if len(test_data) == 0:
                continue
            
            # Train user-specific models
            v_user_model = PersonalizedARXModel()
            a_user_model = PersonalizedARXModel()
            
            v_user_model.fit(train_data, f"v_true_mean_level_t+{self.H}", 
                           "v_true_mean_rm2", features)
            a_user_model.fit(train_data, f"a_true_mean_level_t+{self.H}", 
                           "a_true_mean_rm2", features)
            
            # Predict on test
            v_pred = v_user_model.predict(test_data, "v_true_mean_rm2", features)
            a_pred = a_user_model.predict(test_data, "a_true_mean_rm2", features)
            
            # Calculate metrics
            v_true = test_data[f"v_true_mean_level_t+{self.H}"].values
            a_true = test_data[f"a_true_mean_level_t+{self.H}"].values
            
            mae_v = np.mean(np.abs(v_true - v_pred))
            mae_a = np.mean(np.abs(a_true - a_pred))
            
            rows.append([user_id, mae_v, mae_a, len(test_data)])
        
        per_user = pd.DataFrame(rows, columns=['user_id', 'mae_v', 'mae_a', 'n_test'])
        macro = per_user[['mae_v', 'mae_a']].mean().to_dict() if len(per_user) else {}
        
        return per_user, macro
    
    def predict_new_data(self, df: pd.DataFrame, feature_set: str = 'fusion') -> Dict[str, np.ndarray]:
        """Make predictions on new data"""
        if feature_set not in self.models:
            raise ValueError(f"Model {feature_set} not trained yet")
        
        # Prepare features
        df_feat = self.feature_engineer.transform(df)
        features = self.models[feature_set]['features']
        
        # Get models
        v_model = self.models[feature_set]['valence_model']
        a_model = self.models[feature_set]['arousal_model']
        
        # Make predictions
        v_pred = v_model.predict(df_feat, "v_true_mean_rm2", features)
        a_pred = a_model.predict(df_feat, "a_true_mean_rm2", features)
        
        return {
            'valence': v_pred,  # Changed from 'valence_predictions'
            'arousal': a_pred,  # Changed from 'arousal_predictions'
            'user_ids': df_feat['user_id'].values,
            'weeks': df_feat['week'].values
        }
    
    def save_pipeline(self, filepath: str):
        """Save the entire pipeline"""
        pipeline_data = {
            'models': self.models,
            'feature_sets': self.feature_sets,
            'H': self.H,
            'performance_metrics': self.performance_metrics
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(pipeline_data, f)
        
        logger.info(f"Pipeline saved to {filepath}")
    
    @classmethod
    def load_pipeline(cls, filepath: str):
        """Load a saved pipeline"""
        with open(filepath, 'rb') as f:
            pipeline_data = pickle.load(f)
        
        pipeline = cls(pipeline_data['feature_sets'], pipeline_data['H'])
        pipeline.models = pipeline_data['models']
        pipeline.performance_metrics = pipeline_data.get('performance_metrics', {})
        
        logger.info(f"Pipeline loaded from {filepath}")
        return pipeline

def main():
    """Example usage of the ARX pipeline"""
    
    # Define feature sets
    feature_sets = {
        'circ': ['valence_w', 'arousal_w', 'radius_w', 'v_true_mean_ema3', 'a_true_mean_ema3'],
        'roberta': ['v_hat_mean', 'a_hat_mean', 'r_hat_mean', 'v_hat_mean_ema3', 'a_hat_mean_ema3'],
        'symptom': ['anx_score', 'dep_score', 'ptsd_score', 'mania_score', 'sui_score']
    }
    
    # Initialize pipeline
    pipeline = ARXPipeline(feature_sets, H=2)
    
    # Load and prepare data
    merged_data = pipeline.prepare_data("./data/outputs/symptoms_weekly_user.csv", 
                                      "./data/llm_weekly_user.csv")
    
    # Add EMA features
    merged_data = pipeline.add_ema_features(merged_data)
    
    # Add fusion features
    fusion_feats = (
        feature_sets['circ'] + feature_sets['roberta'] +
        [f for f in merged_data.columns if f.startswith('empath_') and '_z' in f] +
        ["posts_true", "posts_true_ema3"]
    )
    pipeline.feature_sets['fusion'] = fusion_feats
    
    # Filter low variance features
    pipeline.feature_sets = pipeline.filter_low_variance_features(
        merged_data, pipeline.feature_sets
    )
    
    # Train models
    pipeline.train_models(merged_data)
    
    # Evaluate models
    results = pipeline.evaluate_models()
    print("\nModel Performance:")
    print(results.round(3))
    
    # Rolling evaluation for best model
    per_user, macro = pipeline.rolling_evaluation(merged_data, 'fusion')
    print(f"\nRolling evaluation macro: {macro}")
    
    # Save pipeline
    pipeline.save_pipeline("arx_pipeline.pkl")
    
    return pipeline

if __name__ == "__main__":
    pipeline = main()