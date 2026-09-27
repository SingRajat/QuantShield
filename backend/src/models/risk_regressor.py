import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

from backend.src.models.purged_cv import PurgedGroupTimeSeriesSplit

logger = logging.getLogger(__name__)

class RiskRegressor:
    """
    Multi-output continuous risk forecasting model using CatBoost.
    Predicts both Forward_Vol and Forward_MaxDD over a specified horizon.
    """
    
    FEATURE_SETS = {
        "v1_baseline": [
            "Vol_t", "Vol_t5", "Vol_t21", "Vol_t63",
            "MaxDD_t", "MaxDD_t5", "MaxDD_t21", "MaxDD_t63"
        ],
        "family_a": [
            "Vol_t", "Vol_t5", "Vol_t21", "Vol_t63",
            "MaxDD_t", "MaxDD_t5", "MaxDD_t21", "MaxDD_t63"
        ],
        "family_b": [
            "Vol_t", "Vol_t5", "Vol_t21", "Vol_t63",
            "MaxDD_t", "MaxDD_t5", "MaxDD_t21", "MaxDD_t63",
            "Delta_Vol_5", "Delta_Vol_21", "Delta_Vol_63",
            "Vol_Ratio_21_63",
            "Delta_MaxDD_5", "Delta_MaxDD_21", "Delta_MaxDD_63"
        ],
        "v1_1": [
            "Vol_t", "Vol_t5", "Vol_t21", "Vol_t63",
            "MaxDD_t", "MaxDD_t5", "MaxDD_t21", "MaxDD_t63",
            "Delta_Vol_5", "Delta_Vol_21", "Delta_Vol_63",
            "Vol_Ratio_21_63",
            "Delta_MaxDD_5", "Delta_MaxDD_21", "Delta_MaxDD_63"
        ],
        "family_b_ewma": [
            "Vol_t", "Vol_t5", "Vol_t21", "Vol_t63",
            "MaxDD_t", "MaxDD_t5", "MaxDD_t21", "MaxDD_t63",
            "Delta_Vol_5", "Delta_Vol_21", "Delta_Vol_63",
            "Vol_Ratio_21_63",
            "Delta_MaxDD_5", "Delta_MaxDD_21", "Delta_MaxDD_63",
            "EWMA_Vol_t"
        ],
        "family_b_plus_ewma": [
            "Vol_t", "Vol_t5", "Vol_t21", "Vol_t63",
            "MaxDD_t", "MaxDD_t5", "MaxDD_t21", "MaxDD_t63",
            "Delta_Vol_5", "Delta_Vol_21", "Delta_Vol_63",
            "Vol_Ratio_21_63",
            "Delta_MaxDD_5", "Delta_MaxDD_21", "Delta_MaxDD_63",
            "EWMA_Vol_t"
        ]
    }
    FEATURES = FEATURE_SETS["family_b_ewma"]
    TARGETS = ["Forward_Vol", "Forward_MaxDD"]
    
    def __init__(
        self, 
        iterations: int = 300, 
        depth: int = 6, 
        learning_rate: float = 0.05, 
        random_seed: int = 42,
        feature_set: str = "v1_baseline",
        features: Optional[List[str]] = None
    ):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.feature_set = feature_set
        
        if features is not None:
            self.features = features
        else:
            self.features = self.FEATURE_SETS.get(feature_set, self.FEATURE_SETS["v1_baseline"])
        
        self.model = CatBoostRegressor(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            loss_function='MultiRMSE',
            random_seed=self.random_seed,
            verbose=0
        )
        self.is_fitted = False
        
    def train_and_evaluate(
        self, 
        panel_df: pd.DataFrame, 
        n_splits: int = 5, 
        embargo_days: int = 0
    ) -> Dict[str, Any]:
        """
        Trains and evaluates the multi-output model using forward-target purged cross-validation.
        Evaluates out-of-fold metrics strictly on observations that received test-fold predictions.
        Benchmarks against RiskMetrics EWMA on the exact same evaluated test observations.
        
        Args:
            panel_df: The panel dataset containing features, targets, 'Date_t', and 'Target_End'.
            n_splits: Number of time-series splits.
            embargo_days: Buffer days after max training target date to account for persistence.
            
        Returns:
            Dictionary containing OOF regression metrics and EWMA baseline comparison.
        """
        # Ensure data is chronologically sorted by forecast origin Date_t
        df = panel_df.sort_values(by="Date_t").reset_index(drop=True)
        
        X = df[self.features]
        Y = df[self.TARGETS]
        
        cv = PurgedGroupTimeSeriesSplit(n_splits=n_splits, embargo_days=embargo_days)
        
        test_preds_list = []
        test_true_list = []
        test_ewma_list = []
        fold_summaries = []
        
        fold = 1
        for train_idx, test_idx in cv.split(df):
            if len(test_idx) == 0:
                logger.warning(f"Fold {fold}: Empty test set after purge/embargo. Skipping fold.")
                continue
                
            X_train, Y_train = X.iloc[train_idx], Y.iloc[train_idx]
            X_test, Y_test = X.iloc[test_idx], Y.iloc[test_idx]
            
            logger.info(
                f"Fold {fold}: Train size={len(train_idx)}, Test size={len(test_idx)} | "
                f"Train Date_t max={df.iloc[train_idx]['Date_t'].max().strftime('%Y-%m-%d')}, "
                f"Test Date_t min={df.iloc[test_idx]['Date_t'].min().strftime('%Y-%m-%d')}"
            )
            
            fold_model = CatBoostRegressor(
                iterations=self.iterations,
                depth=self.depth,
                learning_rate=self.learning_rate,
                loss_function='MultiRMSE',
                random_seed=self.random_seed,
                verbose=0
            )
            fold_model.fit(X_train, Y_train, eval_set=(X_test, Y_test), early_stopping_rounds=20)
            
            preds = fold_model.predict(X_test)
            test_preds_list.append(preds)
            test_true_list.append(Y_test.values)
            
            # Per-fold metrics
            fold_vol_mae = float(mean_absolute_error(Y_test.values[:, 0], preds[:, 0]))
            fold_vol_rmse = float(np.sqrt(mean_squared_error(Y_test.values[:, 0], preds[:, 0])))
            fold_mdd_mae = float(mean_absolute_error(Y_test.values[:, 1], preds[:, 1]))
            fold_mdd_rmse = float(np.sqrt(mean_squared_error(Y_test.values[:, 1], preds[:, 1])))
            
            fold_summary = {
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "vol_mae": fold_vol_mae,
                "vol_rmse": fold_vol_rmse,
                "maxdd_mae": fold_mdd_mae,
                "maxdd_rmse": fold_mdd_rmse,
            }
            
            if "EWMA_Vol_t" in df.columns:
                ewma_test = df.iloc[test_idx]["EWMA_Vol_t"].values
                test_ewma_list.append(ewma_test)
                fold_ewma_mae = float(mean_absolute_error(Y_test.values[:, 0], ewma_test))
                fold_summary["ewma_vol_mae"] = fold_ewma_mae
                fold_summary["lift_vs_ewma"] = fold_ewma_mae - fold_vol_mae
                
            fold_summaries.append(fold_summary)
            fold += 1
            
        if len(test_preds_list) == 0:
            raise RuntimeError("No test folds were evaluated during cross-validation.")
            
        # Fit final model on the entire dataset for serving/inference
        logger.info("Fitting production model on full dataset...")
        self.model.fit(X, Y)
        self.is_fitted = True
        
        # Pool predictions strictly across evaluated test observations (no unevaluated warm-up rows)
        all_preds = np.vstack(test_preds_list)
        all_true = np.vstack(test_true_list)
        all_ewma = np.concatenate(test_ewma_list) if len(test_ewma_list) > 0 else None
        
        return self._calculate_pooled_metrics(all_true, all_preds, all_ewma, fold_summaries)
        
    def _calculate_pooled_metrics(
        self, 
        all_true: np.ndarray, 
        all_preds: np.ndarray, 
        all_ewma: Optional[np.ndarray],
        fold_summaries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculates pooled out-of-fold regression metrics strictly over evaluated test samples."""
        
        true_vol = all_true[:, 0]
        true_maxdd = all_true[:, 1]
        
        pred_vol = all_preds[:, 0]
        pred_maxdd = all_preds[:, 1]
        
        def compute_metrics(y_true, y_pred):
            mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
            y_t, y_p = y_true[mask], y_pred[mask]
            if len(y_t) < 2:
                return {"oof_mae": np.nan, "oof_rmse": np.nan, "oof_correlation": np.nan}
            corr, _ = pearsonr(y_t, y_p)
            return {
                "oof_mae": float(mean_absolute_error(y_t, y_p)),
                "oof_rmse": float(np.sqrt(mean_squared_error(y_t, y_p))),
                "oof_correlation": float(corr)
            }
            
        vol_metrics = compute_metrics(true_vol, pred_vol)
        maxdd_metrics = compute_metrics(true_maxdd, pred_maxdd)
        
        # Benchmark comparison against RiskMetrics EWMA on the exact same test observations
        if all_ewma is not None:
            mask = ~np.isnan(true_vol) & ~np.isnan(all_ewma)
            y_t, e_p = true_vol[mask], all_ewma[mask]
            ewma_mae = float(mean_absolute_error(y_t, e_p))
            ewma_rmse = float(np.sqrt(mean_squared_error(y_t, e_p)))
            ewma_corr, _ = pearsonr(y_t, e_p)
            
            vol_metrics["ewma_baseline_mae"] = ewma_mae
            vol_metrics["ewma_baseline_rmse"] = ewma_rmse
            vol_metrics["ewma_baseline_correlation"] = float(ewma_corr)
            
            # Positive lift indicates ML has lower error than EWMA baseline
            lift_mae = ewma_mae - vol_metrics["oof_mae"]
            vol_metrics["lift_vs_ewma_mae"] = float(lift_mae)
            
        return {
            "total_evaluated_observations": len(all_true),
            "vol_forecast": vol_metrics,
            "maxdd_forecast": maxdd_metrics,
            "folds": fold_summaries
        }

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Predicts continuous risk metrics for new data.
        Returns a 2D array of shape (n_samples, 2) where col 0 is Vol and col 1 is MaxDD.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
            
        return self.model.predict(features_df[self.features])
