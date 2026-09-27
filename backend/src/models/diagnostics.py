"""
QuantShield v3 — Forecasting Diagnostics Module.
Answers: When, where, and under what conditions does the model fail?
Provides bias analysis, residual distributions, tail errors, portfolio breakdowns,
chronological fold summaries, and leakage verification.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from scipy.stats import ttest_1samp, skew, kurtosis

def analyze_prediction_bias(oof_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes systematic over- or under-prediction bias across all targets and models.
    """
    targets = {
        "ML_Vol": oof_df["Vol_Err_ML"].dropna().values,
        "EWMA_Vol": oof_df["Vol_Err_EWMA"].dropna().values,
        "ML_MaxDD": oof_df["MaxDD_Err_ML"].dropna().values,
        "Naive_MaxDD": oof_df["MaxDD_Err_Naive"].dropna().values,
    }
    
    summary = {}
    for name, residuals in targets.items():
        mean_err = float(np.mean(residuals))
        median_err = float(np.median(residuals))
        std_err = float(np.std(residuals))
        sk = float(skew(residuals))
        kt = float(kurtosis(residuals))
        
        pct_over = float(np.mean(residuals > 0) * 100)
        pct_under = float(np.mean(residuals < 0) * 100)
        
        # Two-tailed t-test testing H0: mean_residual == 0
        t_stat, p_val = ttest_1samp(residuals, 0.0)
        
        summary[name] = {
            "mean_error_bias": mean_err,
            "median_error": median_err,
            "std_error": std_err,
            "skewness": sk,
            "kurtosis": kt,
            "pct_over_predicting": pct_over,
            "pct_under_predicting": pct_under,
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "statistically_biased": bool(p_val < 0.05)
        }
        
    return summary

def analyze_tail_errors(oof_df: pd.DataFrame, tail_quantile: float = 0.95) -> Dict[str, Any]:
    """
    Identifies extreme prediction failures and analyzes their concentration in time and assets.
    """
    vol_q95 = float(oof_df["Abs_Vol_Err_ML"].quantile(tail_quantile))
    vol_q99 = float(oof_df["Abs_Vol_Err_ML"].quantile(0.99))
    maxdd_q95 = float(oof_df["Abs_MaxDD_Err_ML"].quantile(tail_quantile))
    maxdd_q99 = float(oof_df["Abs_MaxDD_Err_ML"].quantile(0.99))
    
    # Extract top 5% worst MaxDD errors
    worst_mdd = oof_df[oof_df["Abs_MaxDD_Err_ML"] >= maxdd_q95].copy()
    worst_mdd_dates = worst_mdd["Date_t"].dt.strftime("%Y-%m-%d").value_counts().head(10).to_dict()
    worst_mdd_ports = worst_mdd["Portfolio_ID"].value_counts().head(10).to_dict()
    
    # Extract top 5% worst Vol errors
    worst_vol = oof_df[oof_df["Abs_Vol_Err_ML"] >= vol_q95].copy()
    worst_vol_dates = worst_vol["Date_t"].dt.strftime("%Y-%m-%d").value_counts().head(10).to_dict()
    worst_vol_ports = worst_vol["Portfolio_ID"].value_counts().head(10).to_dict()
    
    return {
        "thresholds": {
            "vol_abs_error_95th": vol_q95,
            "vol_abs_error_99th": vol_q99,
            "maxdd_abs_error_95th": maxdd_q95,
            "maxdd_abs_error_99th": maxdd_q99,
        },
        "maxdd_tail_analysis": {
            "total_tail_samples": len(worst_mdd),
            "mean_tail_realized_maxdd": float(worst_mdd["Realized_MaxDD"].mean()),
            "mean_tail_predicted_maxdd": float(worst_mdd["Pred_MaxDD_ML"].mean()),
            "most_frequent_dates": worst_mdd_dates,
            "most_frequent_portfolios": worst_mdd_ports
        },
        "vol_tail_analysis": {
            "total_tail_samples": len(worst_vol),
            "mean_tail_realized_vol": float(worst_vol["Realized_Vol"].mean()),
            "mean_tail_predicted_vol": float(worst_vol["Pred_Vol_ML"].mean()),
            "most_frequent_dates": worst_vol_dates,
            "most_frequent_portfolios": worst_vol_ports
        }
    }

def analyze_portfolio_breakdown(oof_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes per-portfolio MAE, RMSE, and benchmark lifts.
    """
    rows = []
    for pid, sub in oof_df.groupby("Portfolio_ID"):
        ml_vol_mae = float(sub["Abs_Vol_Err_ML"].mean())
        ewma_vol_mae = float(sub["Abs_Vol_Err_EWMA"].mean())
        vol_lift = ewma_vol_mae - ml_vol_mae
        
        ml_mdd_mae = float(sub["Abs_MaxDD_Err_ML"].mean())
        naive_mdd_mae = float(sub["Abs_MaxDD_Err_Naive"].mean())
        mdd_lift = naive_mdd_mae - ml_mdd_mae
        
        ml_vol_corr = float(sub[["Realized_Vol", "Pred_Vol_ML"]].corr().iloc[0, 1])
        ml_mdd_corr = float(sub[["Realized_MaxDD", "Pred_MaxDD_ML"]].corr().iloc[0, 1])
        
        vol_win_rate = float(sub["ML_Vol_Wins"].mean())
        mdd_win_rate = float(sub["ML_MaxDD_Wins"].mean())
        
        rows.append({
            "Portfolio_ID": pid,
            "Count": len(sub),
            "ML_Vol_MAE": ml_vol_mae,
            "EWMA_Vol_MAE": ewma_vol_mae,
            "Vol_MAE_Lift": vol_lift,
            "Vol_Win_Rate": vol_win_rate,
            "Vol_Correlation": ml_vol_corr,
            "ML_MaxDD_MAE": ml_mdd_mae,
            "Naive_MaxDD_MAE": naive_mdd_mae,
            "MaxDD_MAE_Lift": mdd_lift,
            "MaxDD_Win_Rate": mdd_win_rate,
            "MaxDD_Correlation": ml_mdd_corr
        })
        
    return pd.DataFrame(rows).sort_values("Vol_MAE_Lift", ascending=False).reset_index(drop=True)

def verify_chronological_folds(panel_df: pd.DataFrame, cv) -> Dict[str, Any]:
    """
    Verifies that train and test forward target intervals strictly do not overlap across all folds.
    Returns detailed fold date-range metadata and mathematical proof of zero leakage.
    """
    df = panel_df.sort_values("Date_t").reset_index(drop=True)
    date_t = pd.to_datetime(df["Date_t"])
    target_start = pd.to_datetime(df["Target_Start"])
    target_end = pd.to_datetime(df["Target_End"])
    
    fold_summaries = []
    all_clean = True
    
    for fold, (tr_idx, te_idx) in enumerate(cv.split(df), start=1):
        if len(te_idx) == 0:
            continue
            
        tr_date_min = date_t.iloc[tr_idx].min()
        tr_date_max = date_t.iloc[tr_idx].max()
        tr_target_end_max = target_end.iloc[tr_idx].max()
        
        te_date_min = date_t.iloc[te_idx].min()
        te_date_max = date_t.iloc[te_idx].max()
        te_target_start_min = target_start.iloc[te_idx].min()
        te_target_end_max = target_end.iloc[te_idx].max()
        
        calendar_gap_days = (te_date_min - tr_target_end_max).days
        
        # Zero leakage checks
        date_t_after_target = te_date_min > tr_target_end_max
        target_after_target = te_target_start_min > tr_target_end_max
        is_leakage_free = date_t_after_target and target_after_target
        if not is_leakage_free:
            all_clean = False
            
        fold_summaries.append({
            "fold": fold,
            "n_train": len(tr_idx),
            "n_test": len(te_idx),
            "train_date_t_range": [tr_date_min.strftime("%Y-%m-%d"), tr_date_max.strftime("%Y-%m-%d")],
            "train_target_end_max": tr_target_end_max.strftime("%Y-%m-%d"),
            "test_date_t_range": [te_date_min.strftime("%Y-%m-%d"), te_date_max.strftime("%Y-%m-%d")],
            "test_target_start_min": te_target_start_min.strftime("%Y-%m-%d"),
            "test_target_end_max": te_target_end_max.strftime("%Y-%m-%d"),
            "calendar_gap_days": calendar_gap_days,
            "zero_leakage_verified": is_leakage_free
        })
        
    return {
        "all_folds_leakage_free": all_clean,
        "fold_details": fold_summaries
    }
