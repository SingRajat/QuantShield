"""
QuantShield v3 — Forecasting Evaluation Module.
Provides continuous regression metrics, predicted-vs-realized generation,
realized-volatility tercile evaluation, and temporal stability tracking.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
from catboost import CatBoostRegressor

from backend.src.models.purged_cv import PurgedGroupTimeSeriesSplit
from backend.src.models.risk_regressor import RiskRegressor

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes comprehensive continuous regression metrics."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 2:
        return {
            "mae": np.nan, "rmse": np.nan, "medae": np.nan,
            "pearson_r": np.nan, "spearman_rho": np.nan, "r2": np.nan
        }
    
    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    medae = float(median_absolute_error(yt, yp))
    pr, _ = pearsonr(yt, yp)
    sr, _ = spearmanr(yt, yp)
    r2 = float(r2_score(yt, yp))
    
    return {
        "mae": mae,
        "rmse": rmse,
        "medae": medae,
        "pearson_r": float(pr),
        "spearman_rho": float(sr),
        "r2": r2
    }

def diebold_mariano_test(
    e1: np.ndarray, 
    e2: np.ndarray, 
    loss: str = 'absolute', 
    h: int = 1, 
    max_lags: int = None
) -> Dict[str, Any]:
    """
    Computes the Diebold-Mariano (1995) test with Harvey-Leybourne-Newbold (1997)
    finite-sample adjustment and Bartlett HAC long-run variance estimator.
    
    H0: Both forecasts have equal predictive accuracy (E[d_t] = 0).
    H1: Forecasts have significantly different predictive accuracy (E[d_t] != 0).
    
    Args:
        e1: Forecast errors from Model 1 (e1 = y_pred1 - y_true).
        e2: Forecast errors from Model 2 (e2 = y_pred2 - y_true).
        loss: 'absolute' (MAE comparison) or 'squared' (MSE comparison).
        h: Forecast horizon steps ahead (default 1).
        max_lags: Lags for Bartlett HAC estimator. Defaults to floor(4*(T/100)^(2/9)).
        
    Returns:
        Dict containing mean loss differential d_bar, DM stat, HLN stat, p-value, and interpretation.
    """
    from scipy import stats
    
    mask = ~np.isnan(e1) & ~np.isnan(e2)
    e1_clean, e2_clean = np.asarray(e1)[mask], np.asarray(e2)[mask]
    T = len(e1_clean)
    
    if T < 10:
        return {"d_bar": np.nan, "dm_stat": np.nan, "p_value": np.nan, "interpretation": "Insufficient samples"}
        
    if loss == 'absolute':
        d = np.abs(e1_clean) - np.abs(e2_clean)
    elif loss == 'squared':
        d = e1_clean**2 - e2_clean**2
    else:
        raise ValueError(f"Unknown loss type '{loss}'. Choose 'absolute' or 'squared'.")
        
    d_bar = float(np.mean(d))
    d_demean = d - d_bar
    gamma_0 = float(np.var(d, ddof=0))
    
    if max_lags is None:
        max_lags = int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
    max_lags = max(1, min(max_lags, T // 4))
    
    gamma = []
    for k in range(1, max_lags + 1):
        gamma_k = float(np.sum(d_demean[k:] * d_demean[:-k]) / T)
        gamma.append(gamma_k)
        
    # Long-run variance with Bartlett kernel weights: w_k = 1 - k / (max_lags + 1)
    lr_var = gamma_0 + 2.0 * sum((1.0 - (k + 1) / (max_lags + 1)) * gamma[k] for k in range(max_lags))
    lr_var = max(lr_var, 1e-12)
    
    dm_stat = float(d_bar / np.sqrt(lr_var / T))
    
    # Harvey, Leybourne, Newbold (1997) finite-sample adjustment
    hln_adj = np.sqrt(max(1e-12, (T + 1 - 2 * h + h * (h - 1) / T) / T))
    dm_hln_stat = float(dm_stat * hln_adj)
    
    # Two-sided p-value using Student's t distribution with T-1 degrees of freedom
    p_value = float(2.0 * (1.0 - stats.t.cdf(np.abs(dm_hln_stat), df=T - 1)))
    
    if p_value < 0.05:
        superior_model = "Model 1" if d_bar < 0 else "Model 2"
        interpretation = f"Statistically significant difference (p={p_value:.4f}). {superior_model} is superior."
    else:
        interpretation = f"No statistically significant difference (p={p_value:.4f}) at alpha=0.05."
        
    return {
        "loss": loss,
        "d_bar": d_bar,
        "dm_stat": dm_stat,
        "dm_hln_stat": dm_hln_stat,
        "p_value": p_value,
        "max_lags": max_lags,
        "T": T,
        "interpretation": interpretation
    }

def generate_oof_predictions(
    panel_df: pd.DataFrame, 
    n_splits: int = 5, 
    embargo_days: int = 21,
    iterations: int = 300,
    depth: int = 6,
    learning_rate: float = 0.05,
    random_seed: int = 42,
    feature_set: str = "v1_baseline",
    features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Executes walk-forward purged cross-validation and records sample-by-sample 
    out-of-fold predictions, benchmark comparisons, and residuals.
    
    Returns:
        pd.DataFrame containing OOF evaluations strictly for tested observations.
    """
    df = panel_df.sort_values(by="Date_t").reset_index(drop=True)
    if features is not None:
        feature_cols = features
    elif hasattr(RiskRegressor, "FEATURE_SETS") and feature_set in RiskRegressor.FEATURE_SETS:
        feature_cols = RiskRegressor.FEATURE_SETS[feature_set]
    else:
        feature_cols = RiskRegressor.FEATURES
        
    X = df[feature_cols]
    Y = df[RiskRegressor.TARGETS]
    
    cv = PurgedGroupTimeSeriesSplit(n_splits=n_splits, embargo_days=embargo_days)
    
    records = []
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(df), start=1):
        if len(test_idx) == 0:
            continue
            
        X_train, Y_train = X.iloc[train_idx], Y.iloc[train_idx]
        X_test, Y_test = X.iloc[test_idx], Y.iloc[test_idx]
        
        # Fit fold CatBoost MultiRMSE model
        fold_model = CatBoostRegressor(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            loss_function='MultiRMSE',
            random_seed=random_seed,
            verbose=0
        )
        fold_model.fit(X_train, Y_train, eval_set=(X_test, Y_test), early_stopping_rounds=20)
        
        preds = fold_model.predict(X_test)
        
        # Naïve benchmark for Forward_MaxDD: historical training-set mean
        train_naive_maxdd = float(Y_train["Forward_MaxDD"].mean())
        
        test_slice = df.iloc[test_idx].copy()
        
        for idx_rel, (orig_idx, row) in enumerate(test_slice.iterrows()):
            pred_vol = float(preds[idx_rel, 0])
            pred_maxdd = float(preds[idx_rel, 1])
            realized_vol = float(row["Forward_Vol"])
            realized_maxdd = float(row["Forward_MaxDD"])
            ewma_vol = float(row.get("EWMA_Vol_t", np.nan))
            naive_maxdd = train_naive_maxdd
            
            vol_err_ml = pred_vol - realized_vol
            vol_err_ewma = ewma_vol - realized_vol if not np.isnan(ewma_vol) else np.nan
            maxdd_err_ml = pred_maxdd - realized_maxdd
            maxdd_err_naive = naive_maxdd - realized_maxdd
            
            abs_vol_err_ml = abs(vol_err_ml)
            abs_vol_err_ewma = abs(vol_err_ewma) if not np.isnan(vol_err_ewma) else np.nan
            abs_maxdd_err_ml = abs(maxdd_err_ml)
            abs_maxdd_err_naive = abs(maxdd_err_naive)
            
            ml_vol_wins = bool(abs_vol_err_ml < abs_vol_err_ewma) if not np.isnan(abs_vol_err_ewma) else False
            ml_maxdd_wins = bool(abs_maxdd_err_ml < abs_maxdd_err_naive)
            
            records.append({
                "Portfolio_ID": row["Portfolio_ID"],
                "Date_t": pd.to_datetime(row["Date_t"]),
                "Target_Start": pd.to_datetime(row["Target_Start"]),
                "Target_End": pd.to_datetime(row["Target_End"]),
                "Fold": fold,
                "Realized_Vol": realized_vol,
                "Realized_MaxDD": realized_maxdd,
                "Pred_Vol_ML": pred_vol,
                "Pred_MaxDD_ML": pred_maxdd,
                "Pred_Vol_EWMA": ewma_vol,
                "Pred_MaxDD_Naive": naive_maxdd,
                "Vol_Err_ML": vol_err_ml,
                "Vol_Err_EWMA": vol_err_ewma,
                "MaxDD_Err_ML": maxdd_err_ml,
                "MaxDD_Err_Naive": maxdd_err_naive,
                "Abs_Vol_Err_ML": abs_vol_err_ml,
                "Abs_Vol_Err_EWMA": abs_vol_err_ewma,
                "Abs_MaxDD_Err_ML": abs_maxdd_err_ml,
                "Abs_MaxDD_Err_Naive": abs_maxdd_err_naive,
                "ML_Vol_Wins": ml_vol_wins,
                "ML_MaxDD_Wins": ml_maxdd_wins
            })
            
    oof_df = pd.DataFrame(records).sort_values(by=["Date_t", "Portfolio_ID"]).reset_index(drop=True)
    return oof_df

def compute_tercile_analysis(oof_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluates ML vs EWMA across empirical realized volatility terciles:
    Calm (lowest 33.3%), Normal (middle 33.3%), Stressed (top 33.3%).
    """
    q1 = oof_df["Realized_Vol"].quantile(1/3)
    q2 = oof_df["Realized_Vol"].quantile(2/3)
    
    terciles = {
        "Calm (Low Vol)": oof_df[oof_df["Realized_Vol"] <= q1],
        "Normal (Mid Vol)": oof_df[(oof_df["Realized_Vol"] > q1) & (oof_df["Realized_Vol"] <= q2)],
        "Stressed (High Vol)": oof_df[oof_df["Realized_Vol"] > q2]
    }
    
    results = {
        "tercile_thresholds": {"q1_33pct": float(q1), "q2_67pct": float(q2)},
        "breakdown": {}
    }
    
    for name, sub in terciles.items():
        ml_mae = float(sub["Abs_Vol_Err_ML"].mean())
        ewma_mae = float(sub["Abs_Vol_Err_EWMA"].mean())
        ml_rmse = float(np.sqrt(np.mean(sub["Vol_Err_ML"] ** 2)))
        ewma_rmse = float(np.sqrt(np.mean(sub["Vol_Err_EWMA"] ** 2)))
        ml_bias = float(sub["Vol_Err_ML"].mean()) # positive = over-predicts, negative = under-predicts
        ewma_bias = float(sub["Vol_Err_EWMA"].mean())
        ml_win_rate = float((sub["Abs_Vol_Err_ML"] < sub["Abs_Vol_Err_EWMA"]).mean())
        
        results["breakdown"][name] = {
            "n_obs": len(sub),
            "realized_vol_range": [float(sub["Realized_Vol"].min()), float(sub["Realized_Vol"].max())],
            "ml_mae": ml_mae,
            "ewma_mae": ewma_mae,
            "lift_mae": ewma_mae - ml_mae,
            "ml_rmse": ml_rmse,
            "ewma_rmse": ewma_rmse,
            "ml_bias": ml_bias,
            "ewma_bias": ewma_bias,
            "ml_win_rate": ml_win_rate
        }
        
    return results

def compute_rolling_mae(oof_df: pd.DataFrame, window_steps: int = 12) -> pd.DataFrame:
    """
    Computes rolling mean absolute error over unique chronological prediction dates.
    Window is in terms of distinct rolling dates (e.g. 12 steps ~ 1 year at 21d step size).
    """
    date_grp = oof_df.groupby("Date_t").agg({
        "Abs_Vol_Err_ML": "mean",
        "Abs_Vol_Err_EWMA": "mean",
        "Abs_MaxDD_Err_ML": "mean",
        "Abs_MaxDD_Err_Naive": "mean"
    }).reset_index().sort_values("Date_t")
    
    date_grp["Rolling_Vol_MAE_ML"] = date_grp["Abs_Vol_Err_ML"].rolling(window=window_steps, min_periods=3).mean()
    date_grp["Rolling_Vol_MAE_EWMA"] = date_grp["Abs_Vol_Err_EWMA"].rolling(window=window_steps, min_periods=3).mean()
    date_grp["Rolling_MaxDD_MAE_ML"] = date_grp["Abs_MaxDD_Err_ML"].rolling(window=window_steps, min_periods=3).mean()
    date_grp["Rolling_MaxDD_MAE_Naive"] = date_grp["Abs_MaxDD_Err_Naive"].rolling(window=window_steps, min_periods=3).mean()
    
    return date_grp
