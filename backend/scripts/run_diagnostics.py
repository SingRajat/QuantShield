import os
import sys
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'backend'))

from backend.src.models.purged_cv import PurgedGroupTimeSeriesSplit
from backend.src.models.evaluation import (
    generate_oof_predictions,
    compute_regression_metrics,
    diebold_mariano_test,
    compute_tercile_analysis,
    compute_rolling_mae
)
from backend.src.models.diagnostics import (
    analyze_prediction_bias,
    analyze_tail_errors,
    analyze_portfolio_breakdown,
    verify_chronological_folds
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="QuantShield Evaluation and Diagnostics Suite")
    parser.add_argument(
        "--feature-set",
        type=str,
        default="v1_baseline",
        choices=["v1_baseline", "family_a", "family_b", "v1_1", "family_b_ewma", "family_b_plus_ewma"],
        help="Feature set configuration ('v1_baseline' / 'family_a', 'family_b' / 'v1_1', or 'family_b_ewma')"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to training dataset CSV. Defaults to output_dir / training_dataset.csv"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save diagnostic artifacts. Defaults to backend/src/models for v1_baseline, or backend/src/models/experiments/{feature_set}"
    )
    parser.add_argument(
        "--v1-oof-path",
        type=str,
        default=None,
        help="Path to baseline oof_predictions.csv for comparative statistical tests"
    )
    return parser.parse_args()

def compute_date_level_dm_test(oof_current: pd.DataFrame, oof_baseline: pd.DataFrame, target: str = "Vol", loss: str = "absolute") -> Dict[str, Any]:
    """
    Computes panel-aware date-clustered Diebold-Mariano test using Bartlett HAC long-run variance.
    Collapses loss differential across portfolios for each unique Date_t:
    d_date = mean_{portfolios} (Loss_current - Loss_baseline).
    """
    from scipy import stats
    df_curr = oof_current.sort_values(by=["Date_t", "Portfolio_ID"]).reset_index(drop=True)
    df_base = oof_baseline.sort_values(by=["Date_t", "Portfolio_ID"]).reset_index(drop=True)
    
    if target == "Vol":
        if loss == "absolute":
            curr_loss = df_curr["Abs_Vol_Err_ML"].values
            base_loss = df_base["Abs_Vol_Err_ML"].values
        else:
            curr_loss = df_curr["Vol_Err_ML"].values ** 2
            base_loss = df_base["Vol_Err_ML"].values ** 2
    else: # MaxDD
        if loss == "absolute":
            curr_loss = df_curr["Abs_MaxDD_Err_ML"].values
            base_loss = df_base["Abs_MaxDD_Err_ML"].values
        else:
            curr_loss = df_curr["MaxDD_Err_ML"].values ** 2
            base_loss = df_base["MaxDD_Err_ML"].values ** 2
            
    df_merged = pd.DataFrame({
        "Date_t": df_curr["Date_t"],
        "loss_diff": curr_loss - base_loss
    })
    
    # Aggregate mean loss diff by Date_t
    date_diff = df_merged.groupby("Date_t")["loss_diff"].mean().values
    T = len(date_diff)
    d_bar = float(np.mean(date_diff))
    gamma_0 = float(np.var(date_diff, ddof=0))
    
    max_lags = int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
    max_lags = max(1, min(max_lags, T // 4))
    
    d_demean = date_diff - d_bar
    gamma = []
    for k in range(1, max_lags + 1):
        gamma_k = float(np.sum(d_demean[k:] * d_demean[:-k]) / T)
        gamma.append(gamma_k)
        
    lr_var = gamma_0 + 2.0 * sum((1.0 - (k + 1) / (max_lags + 1)) * gamma[k] for k in range(max_lags))
    lr_var = max(lr_var, 1e-12)
    
    dm_stat = float(d_bar / np.sqrt(lr_var / T))
    p_value = float(2.0 * (1.0 - stats.t.cdf(np.abs(dm_stat), df=T - 1)))
    
    superior = "Current Model" if d_bar < 0 else "Baseline"
    interp = f"{superior} is superior (p={p_value:.4f})" if p_value < 0.05 else f"No significant difference (p={p_value:.4f})"
    
    return {
        "target": target,
        "loss": loss,
        "T_dates": T,
        "d_bar": d_bar,
        "dm_stat": dm_stat,
        "p_value": p_value,
        "max_lags": max_lags,
        "interpretation": interp
    }

def main():
    args = parse_args()
    logger.info(f"Starting QuantShield Diagnostics Suite [Feature Set: {args.feature_set}]...")
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.feature_set in ["v1_baseline", "family_a"]:
        output_dir = project_root / 'backend' / 'src' / 'models'
    elif args.feature_set in ["family_b", "v1_1"]:
        output_dir = project_root / 'backend' / 'src' / 'models' / 'experiments' / 'family_b'
    elif args.feature_set in ["family_b_ewma", "family_b_plus_ewma"]:
        output_dir = project_root / 'backend' / 'src' / 'models' / 'experiments' / 'family_b_ewma'
    else:
        output_dir = project_root / 'backend' / 'src' / 'models' / 'experiments' / args.feature_set

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Diagnostic artifacts will be saved to: {output_dir}")

    # Determine dataset path
    if args.dataset_path:
        dataset_path = Path(args.dataset_path)
    else:
        dataset_path = output_dir / 'training_dataset.csv'
        if not dataset_path.exists():
            # Fall back to main training_dataset.csv if not found in output_dir
            fallback = project_root / 'backend' / 'src' / 'models' / 'training_dataset.csv'
            if fallback.exists():
                dataset_path = fallback

    if not dataset_path.exists():
        raise FileNotFoundError(f"Training dataset not found at {dataset_path}. Run train_model.py first.")
        
    df = pd.read_csv(dataset_path)
    logger.info(f"Loaded training dataset from {dataset_path}. Shape: {df.shape}")
    
    # 2. Generate OOF Predictions Dataset
    logger.info(f"Generating Out-of-Fold Predictions Dataset across 5 Purged Folds (feature_set={args.feature_set})...")
    oof_df = generate_oof_predictions(
        df, 
        n_splits=5, 
        embargo_days=21, 
        random_seed=42, 
        feature_set=args.feature_set
    )
    
    oof_csv_path = output_dir / 'oof_predictions.csv'
    oof_df.to_csv(oof_csv_path, index=False)
    logger.info(f"Saved OOF Predictions Dataset to: {oof_csv_path} (Shape: {oof_df.shape})")
    
    # 3. Overall Regression Metrics
    logger.info("\n--- OVERALL REGRESSION METRICS ---")
    vol_metrics_ml = compute_regression_metrics(oof_df["Realized_Vol"].values, oof_df["Pred_Vol_ML"].values)
    vol_metrics_ewma = compute_regression_metrics(oof_df["Realized_Vol"].values, oof_df["Pred_Vol_EWMA"].values)
    mdd_metrics_ml = compute_regression_metrics(oof_df["Realized_MaxDD"].values, oof_df["Pred_MaxDD_ML"].values)
    mdd_metrics_naive = compute_regression_metrics(oof_df["Realized_MaxDD"].values, oof_df["Pred_MaxDD_Naive"].values)
    
    ml_vol_win_rate = float(oof_df["ML_Vol_Wins"].mean() * 100)
    ml_mdd_win_rate = float(oof_df["ML_MaxDD_Wins"].mean() * 100)
    
    # 3B. Diebold-Mariano Forecast Comparison Tests vs Benchmarks
    logger.info("\n--- DIEBOLD-MARIANO STATISTICAL FORECAST COMPARISON TESTS (VS BENCHMARKS) ---")
    dm_vol_abs = diebold_mariano_test(oof_df["Vol_Err_ML"].values, oof_df["Vol_Err_EWMA"].values, loss='absolute')
    dm_vol_sq = diebold_mariano_test(oof_df["Vol_Err_ML"].values, oof_df["Vol_Err_EWMA"].values, loss='squared')
    dm_mdd_abs = diebold_mariano_test(oof_df["MaxDD_Err_ML"].values, oof_df["MaxDD_Err_Naive"].values, loss='absolute')
    dm_mdd_sq = diebold_mariano_test(oof_df["MaxDD_Err_ML"].values, oof_df["MaxDD_Err_Naive"].values, loss='squared')
    
    # 3C. Paired Comparative Tests vs Baseline (if applicable and available)
    baseline_comparison = None
    if args.v1_oof_path:
        base_oof_path = Path(args.v1_oof_path)
        baseline_label = "Family B Baseline" if "family_b" in str(base_oof_path).lower() else "V1.0 Baseline"
    else:
        if args.feature_set in ["family_b_ewma", "family_b_plus_ewma"]:
            base_oof_path = project_root / 'backend' / 'src' / 'models' / 'experiments' / 'family_b' / 'oof_predictions.csv'
            baseline_label = "Family B Baseline"
        else:
            base_oof_path = project_root / 'backend' / 'src' / 'models' / 'oof_predictions.csv'
            baseline_label = "V1.0 Baseline"
    
    base_oof_df = None
    if args.feature_set not in ["v1_baseline", "family_a"] and base_oof_path.exists():
        logger.info(f"\n--- PAIRED COMPARATIVE TESTS VS BASELINE: {baseline_label} ({base_oof_path}) ---")
        base_oof_df = pd.read_csv(base_oof_path)
        
        # Verify alignment
        assert len(oof_df) == len(base_oof_df), f"Row count mismatch: Current has {len(oof_df)}, Baseline has {len(base_oof_df)}"
        
        # Volatility: Observation-level DM
        dm_vol_base_abs = diebold_mariano_test(oof_df["Vol_Err_ML"].values, base_oof_df["Vol_Err_ML"].values, loss='absolute')
        dm_vol_base_sq = diebold_mariano_test(oof_df["Vol_Err_ML"].values, base_oof_df["Vol_Err_ML"].values, loss='squared')
        
        # MaxDD: Observation-level DM
        dm_mdd_base_abs = diebold_mariano_test(oof_df["MaxDD_Err_ML"].values, base_oof_df["MaxDD_Err_ML"].values, loss='absolute')
        dm_mdd_base_sq = diebold_mariano_test(oof_df["MaxDD_Err_ML"].values, base_oof_df["MaxDD_Err_ML"].values, loss='squared')
        
        # Date-level Panel-Aware HAC DM tests
        date_dm_vol_abs = compute_date_level_dm_test(oof_df, base_oof_df, target="Vol", loss="absolute")
        date_dm_vol_sq = compute_date_level_dm_test(oof_df, base_oof_df, target="Vol", loss="squared")
        date_dm_mdd_abs = compute_date_level_dm_test(oof_df, base_oof_df, target="MaxDD", loss="absolute")
        date_dm_mdd_sq = compute_date_level_dm_test(oof_df, base_oof_df, target="MaxDD", loss="squared")
        
        # Win rate
        curr_vol_wins_base = float((oof_df["Abs_Vol_Err_ML"] < base_oof_df["Abs_Vol_Err_ML"]).mean() * 100)
        curr_mdd_wins_base = float((oof_df["Abs_MaxDD_Err_ML"] < base_oof_df["Abs_MaxDD_Err_ML"]).mean() * 100)
        
        # Baseline overall metrics
        base_vol_metrics = compute_regression_metrics(base_oof_df["Realized_Vol"].values, base_oof_df["Pred_Vol_ML"].values)
        base_mdd_metrics = compute_regression_metrics(base_oof_df["Realized_MaxDD"].values, base_oof_df["Pred_MaxDD_ML"].values)
        
        baseline_comparison = {
            "baseline_label": baseline_label,
            "baseline_oof_path": str(base_oof_path),
            "baseline_oof_rows": len(base_oof_df),
            "common_rows": len(oof_df),
            "volatility": {
                "baseline_mae": base_vol_metrics["mae"],
                "baseline_rmse": base_vol_metrics["rmse"],
                "baseline_r2": base_vol_metrics["r2"],
                "baseline_pearson_r": base_vol_metrics["pearson_r"],
                "baseline_spearman_rho": base_vol_metrics["spearman_rho"],
                "current_mae": vol_metrics_ml["mae"],
                "current_rmse": vol_metrics_ml["rmse"],
                "current_r2": vol_metrics_ml["r2"],
                "current_pearson_r": vol_metrics_ml["pearson_r"],
                "current_spearman_rho": vol_metrics_ml["spearman_rho"],
                "delta_mae": vol_metrics_ml["mae"] - base_vol_metrics["mae"],
                "delta_rmse": vol_metrics_ml["rmse"] - base_vol_metrics["rmse"],
                "delta_r2": vol_metrics_ml["r2"] - base_vol_metrics["r2"],
                "current_win_rate_vs_baseline_pct": curr_vol_wins_base,
                "diebold_mariano_obs_abs": dm_vol_base_abs,
                "diebold_mariano_obs_sq": dm_vol_base_sq,
                "date_level_hac_dm_abs": date_dm_vol_abs,
                "date_level_hac_dm_sq": date_dm_vol_sq
            },
            "maximum_drawdown": {
                "baseline_mae": base_mdd_metrics["mae"],
                "baseline_rmse": base_mdd_metrics["rmse"],
                "baseline_r2": base_mdd_metrics["r2"],
                "baseline_pearson_r": base_mdd_metrics["pearson_r"],
                "baseline_spearman_rho": base_mdd_metrics["spearman_rho"],
                "current_mae": mdd_metrics_ml["mae"],
                "current_rmse": mdd_metrics_ml["rmse"],
                "current_r2": mdd_metrics_ml["r2"],
                "current_pearson_r": mdd_metrics_ml["pearson_r"],
                "current_spearman_rho": mdd_metrics_ml["spearman_rho"],
                "delta_mae": mdd_metrics_ml["mae"] - base_mdd_metrics["mae"],
                "delta_rmse": mdd_metrics_ml["rmse"] - base_mdd_metrics["rmse"],
                "delta_r2": mdd_metrics_ml["r2"] - base_mdd_metrics["r2"],
                "current_win_rate_vs_baseline_pct": curr_mdd_wins_base,
                "diebold_mariano_obs_abs": dm_mdd_base_abs,
                "diebold_mariano_obs_sq": dm_mdd_base_sq,
                "date_level_hac_dm_abs": date_dm_mdd_abs,
                "date_level_hac_dm_sq": date_dm_mdd_sq
            }
        }

    # 3D. Fold-by-Fold Breakdown
    logger.info("\n--- FOLD-BY-FOLD BREAKDOWN ---")
    fold_breakdowns = []
    for f_id in sorted(oof_df["Fold"].unique()):
        f_sub = oof_df[oof_df["Fold"] == f_id]
        f_vol = compute_regression_metrics(f_sub["Realized_Vol"].values, f_sub["Pred_Vol_ML"].values)
        f_mdd = compute_regression_metrics(f_sub["Realized_MaxDD"].values, f_sub["Pred_MaxDD_ML"].values)
        f_ewma = compute_regression_metrics(f_sub["Realized_Vol"].values, f_sub["Pred_Vol_EWMA"].values)
        f_naive = compute_regression_metrics(f_sub["Realized_MaxDD"].values, f_sub["Pred_MaxDD_Naive"].values)
        
        entry = {
            "fold": int(f_id),
            "n_obs": len(f_sub),
            "date_range": [str(f_sub["Date_t"].min()), str(f_sub["Date_t"].max())],
            "volatility": {
                "ml": f_vol,
                "ewma": f_ewma,
                "lift_mae": f_ewma["mae"] - f_vol["mae"],
                "win_rate_pct": float(f_sub["ML_Vol_Wins"].mean() * 100)
            },
            "maximum_drawdown": {
                "ml": f_mdd,
                "naive": f_naive,
                "lift_mae": f_naive["mae"] - f_mdd["mae"],
                "win_rate_pct": float(f_sub["ML_MaxDD_Wins"].mean() * 100)
            }
        }
        
        if baseline_comparison and base_oof_df is not None:
            f_base = base_oof_df[base_oof_df["Fold"] == f_id]
            f_base_vol = compute_regression_metrics(f_base["Realized_Vol"].values, f_base["Pred_Vol_ML"].values)
            f_base_mdd = compute_regression_metrics(f_base["Realized_MaxDD"].values, f_base["Pred_MaxDD_ML"].values)
            entry["baseline_comparison"] = {
                "volatility": {
                    "baseline_mae": f_base_vol["mae"],
                    "baseline_rmse": f_base_vol["rmse"],
                    "baseline_r2": f_base_vol["r2"],
                    "baseline_pearson_r": f_base_vol["pearson_r"],
                    "baseline_spearman_rho": f_base_vol["spearman_rho"],
                    "delta_mae": f_vol["mae"] - f_base_vol["mae"],
                    "delta_rmse": f_vol["rmse"] - f_base_vol["rmse"],
                    "delta_r2": f_vol["r2"] - f_base_vol["r2"],
                    "win_rate_vs_base_pct": float((f_sub["Abs_Vol_Err_ML"] < f_base["Abs_Vol_Err_ML"]).mean() * 100)
                },
                "maximum_drawdown": {
                    "baseline_mae": f_base_mdd["mae"],
                    "baseline_rmse": f_base_mdd["rmse"],
                    "baseline_r2": f_base_mdd["r2"],
                    "baseline_pearson_r": f_base_mdd["pearson_r"],
                    "baseline_spearman_rho": f_base_mdd["spearman_rho"],
                    "delta_mae": f_mdd["mae"] - f_base_mdd["mae"],
                    "delta_rmse": f_mdd["rmse"] - f_base_mdd["rmse"],
                    "delta_r2": f_mdd["r2"] - f_base_mdd["r2"],
                    "win_rate_vs_base_pct": float((f_sub["Abs_MaxDD_Err_ML"] < f_base["Abs_MaxDD_Err_ML"]).mean() * 100)
                }
            }
        fold_breakdowns.append(entry)
    
    # 4. Bias Analysis
    logger.info("\n--- PREDICTION BIAS ANALYSIS ---")
    bias_results = analyze_prediction_bias(oof_df)
    
    # 5. Realized-Vol Tercile Analysis
    logger.info("\n--- REALIZED-VOLATILITY TERCILE ANALYSIS ---")
    tercile_results = compute_tercile_analysis(oof_df)
    
    # 6. Tail Error Analysis
    logger.info("\n--- TAIL ERROR ANALYSIS (TOP 5% WORST PREDICTIONS) ---")
    tail_results = analyze_tail_errors(oof_df, tail_quantile=0.95)
    
    # 7. Portfolio Breakdown
    logger.info("\n--- PORTFOLIO ERROR BREAKDOWN ---")
    port_df = analyze_portfolio_breakdown(oof_df)
    port_csv_path = output_dir / 'portfolio_error_breakdown.csv'
    port_df.to_csv(port_csv_path, index=False)
    logger.info(f"Saved portfolio breakdown to: {port_csv_path}")
    
    # 8. Chronological Fold & Target Overlap Verification
    logger.info("\n--- CHRONOLOGICAL FOLD & OVERLAP VERIFICATION ---")
    cv = PurgedGroupTimeSeriesSplit(n_splits=5, embargo_days=21)
    fold_verification = verify_chronological_folds(df, cv)
    
    # 9. Rolling MAE Over Time
    logger.info("\n--- ROLLING MAE OVER TIME ---")
    rolling_df = compute_rolling_mae(oof_df, window_steps=12)
    rolling_csv_path = output_dir / 'rolling_mae_timeline.csv'
    rolling_df.to_csv(rolling_csv_path, index=False)
    logger.info(f"Saved rolling MAE timeline to: {rolling_csv_path}")
    
    # Compile Full JSON Diagnostic Report
    full_report = {
        "feature_set": args.feature_set,
        "dataset_summary": {
            "total_raw_rows": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "date_range": [str(df["Date_t"].min()), str(df["Date_t"].max())],
            "evaluated_oof_rows": len(oof_df)
        },
        "overall_metrics": {
            "volatility": {
                "ml": vol_metrics_ml,
                "ewma_benchmark": vol_metrics_ewma,
                "lift_mae": vol_metrics_ewma["mae"] - vol_metrics_ml["mae"],
                "lift_rmse": vol_metrics_ewma["rmse"] - vol_metrics_ml["rmse"],
                "ml_win_rate_pct": ml_vol_win_rate
            },
            "maximum_drawdown": {
                "ml": mdd_metrics_ml,
                "naive_mean_benchmark": mdd_metrics_naive,
                "lift_mae": mdd_metrics_naive["mae"] - mdd_metrics_ml["mae"],
                "lift_rmse": mdd_metrics_naive["rmse"] - mdd_metrics_ml["rmse"],
                "ml_win_rate_pct": ml_mdd_win_rate
            }
        },
        "diebold_mariano_tests_vs_benchmarks": {
            "volatility_ml_vs_ewma": {
                "absolute_loss": dm_vol_abs,
                "squared_loss": dm_vol_sq
            },
            "maxdd_ml_vs_naive": {
                "absolute_loss": dm_mdd_abs,
                "squared_loss": dm_mdd_sq
            }
        },
        "comparison_vs_baseline": baseline_comparison,
        "prediction_bias": bias_results,
        "tercile_analysis": tercile_results,
        "tail_errors": tail_results,
        "portfolio_summary": {
            "top_ml_wins": port_df.head(5).to_dict(orient="records"),
            "worst_ml_loses": port_df.tail(5).to_dict(orient="records")
        },
        "fold_metrics": fold_breakdowns,
        "fold_verification": fold_verification
    }
    
    report_path = output_dir / 'diagnostic_report.json'
    with open(report_path, 'w') as f:
        json.dump(full_report, f, indent=2)
    logger.info(f"Saved complete diagnostic report to: {report_path}")
    
    # Print clean summary to terminal
    print("\n" + "="*70)
    print(f"DIAGNOSTIC SUMMARY [FEATURE SET: {args.feature_set}]")
    print("="*70)
    print(f"Total Evaluated OOF Observations: {len(oof_df)}")
    print(f"\n1. FORWARD VOLATILITY FORECAST (ML vs EWMA):")
    print(f"   ML MAE:        {vol_metrics_ml['mae']:.4f} | RMSE: {vol_metrics_ml['rmse']:.4f} | MedAE: {vol_metrics_ml['medae']:.4f} | R: {vol_metrics_ml['pearson_r']:.4f} | Rho: {vol_metrics_ml['spearman_rho']:.4f} | R2: {vol_metrics_ml['r2']:.4f}")
    print(f"   EWMA MAE:      {vol_metrics_ewma['mae']:.4f} | RMSE: {vol_metrics_ewma['rmse']:.4f} | MedAE: {vol_metrics_ewma['medae']:.4f} | R: {vol_metrics_ewma['pearson_r']:.4f} | Rho: {vol_metrics_ewma['spearman_rho']:.4f} | R2: {vol_metrics_ewma['r2']:.4f}")
    print(f"   Lift vs EWMA:  {vol_metrics_ewma['mae'] - vol_metrics_ml['mae']:+.4f} MAE (positive = ML beats EWMA)")
    print(f"   ML Win Rate:   {ml_vol_win_rate:.2f}% of observations")
    print(f"   [Diebold-Mariano Test (Absolute Loss / MAE)]:")
    print(f"     Mean diff d_bar: {dm_vol_abs['d_bar']:+.6f} | DM Stat: {dm_vol_abs['dm_hln_stat']:+.4f} | p-value: {dm_vol_abs['p_value']:.4f}")
    print(f"     Interpretation:  {dm_vol_abs['interpretation']}")
    
    print(f"\n2. FORWARD MAXIMUM DRAWDOWN FORECAST (ML vs Naive Historical Mean):")
    print(f"   ML MAE:        {mdd_metrics_ml['mae']:.4f} | RMSE: {mdd_metrics_ml['rmse']:.4f} | MedAE: {mdd_metrics_ml['medae']:.4f} | R: {mdd_metrics_ml['pearson_r']:.4f} | Rho: {mdd_metrics_ml['spearman_rho']:.4f} | R2: {mdd_metrics_ml['r2']:.4f}")
    print(f"   Naive Mean:    {mdd_metrics_naive['mae']:.4f} | RMSE: {mdd_metrics_naive['rmse']:.4f} | MedAE: {mdd_metrics_naive['medae']:.4f} | R: {mdd_metrics_naive['pearson_r']:.4f}")
    print(f"   Lift vs Naive: {mdd_metrics_naive['mae'] - mdd_metrics_ml['mae']:+.4f} MAE (positive = ML beats naive)")
    print(f"   ML Win Rate:   {ml_mdd_win_rate:.2f}% of observations")
    print(f"   [Diebold-Mariano Test (Absolute Loss / MAE)]:")
    print(f"     Mean diff d_bar: {dm_mdd_abs['d_bar']:+.6f} | DM Stat: {dm_mdd_abs['dm_hln_stat']:+.4f} | p-value: {dm_mdd_abs['p_value']:.4e}")
    print(f"     Interpretation:  {dm_mdd_abs['interpretation']}")
    
    if baseline_comparison:
        b_label = baseline_comparison["baseline_label"]
        print(f"\n3. PAIRED COMPARISON: CURRENT ({args.feature_set}) vs {b_label}:")
        vol_c = baseline_comparison["volatility"]
        mdd_c = baseline_comparison["maximum_drawdown"]
        print(f"   Volatility:")
        print(f"     Baseline MAE: {vol_c['baseline_mae']:.4f} -> Current MAE: {vol_c['current_mae']:.4f} (Delta: {vol_c['delta_mae']:+.4f})")
        print(f"     Baseline RMSE: {vol_c['baseline_rmse']:.4f} -> Current RMSE: {vol_c['current_rmse']:.4f} (Delta: {vol_c['delta_rmse']:+.4f})")
        print(f"     Baseline R2:   {vol_c['baseline_r2']:.4f} -> Current R2:   {vol_c['current_r2']:.4f} (Delta: {vol_c['delta_r2']:+.4f})")
        print(f"     Baseline Pearson r: {vol_c['baseline_pearson_r']:.4f} -> Current: {vol_c['current_pearson_r']:.4f}")
        print(f"     Baseline Spearman rho: {vol_c['baseline_spearman_rho']:.4f} -> Current: {vol_c['current_spearman_rho']:.4f}")
        print(f"     Current Win Rate vs Baseline: {vol_c['current_win_rate_vs_baseline_pct']:.2f}%")
        print(f"     Observation-Level DM Stat (Abs Loss): {vol_c['diebold_mariano_obs_abs']['dm_hln_stat']:+.4f} (p={vol_c['diebold_mariano_obs_abs']['p_value']:.4f})")
        print(f"     Date-Level Panel-Aware HAC DM (T={vol_c['date_level_hac_dm_abs']['T_dates']}): d_bar={vol_c['date_level_hac_dm_abs']['d_bar']:+.6f}, DM-stat={vol_c['date_level_hac_dm_abs']['dm_stat']:+.4f}, p={vol_c['date_level_hac_dm_abs']['p_value']:.4f} -> {vol_c['date_level_hac_dm_abs']['interpretation']}")
        print(f"   Maximum Drawdown:")
        print(f"     Baseline MAE: {mdd_c['baseline_mae']:.4f} -> Current MAE: {mdd_c['current_mae']:.4f} (Delta: {mdd_c['delta_mae']:+.4f})")
        print(f"     Baseline RMSE: {mdd_c['baseline_rmse']:.4f} -> Current RMSE: {mdd_c['current_rmse']:.4f} (Delta: {mdd_c['delta_rmse']:+.4f})")
        print(f"     Baseline R2:   {mdd_c['baseline_r2']:.4f} -> Current R2:   {mdd_c['current_r2']:.4f} (Delta: {mdd_c['delta_r2']:+.4f})")
        print(f"     Baseline Pearson r: {mdd_c['baseline_pearson_r']:.4f} -> Current: {mdd_c['current_pearson_r']:.4f}")
        print(f"     Baseline Spearman rho: {mdd_c['baseline_spearman_rho']:.4f} -> Current: {mdd_c['current_spearman_rho']:.4f}")
        print(f"     Current Win Rate vs Baseline: {mdd_c['current_win_rate_vs_baseline_pct']:.2f}%")
        print(f"     Observation-Level DM Stat (Abs Loss): {mdd_c['diebold_mariano_obs_abs']['dm_hln_stat']:+.4f} (p={mdd_c['diebold_mariano_obs_abs']['p_value']:.4e})")
        print(f"     Date-Level Panel-Aware HAC DM (T={mdd_c['date_level_hac_dm_abs']['T_dates']}): d_bar={mdd_c['date_level_hac_dm_abs']['d_bar']:+.6f}, DM-stat={mdd_c['date_level_hac_dm_abs']['dm_stat']:+.4f}, p={mdd_c['date_level_hac_dm_abs']['p_value']:.4f} -> {mdd_c['date_level_hac_dm_abs']['interpretation']}")
        
    print(f"\n3B. FOLD-BY-FOLD DETAILED RESULTS:")
    for fb in fold_breakdowns:
        f_num = fb["fold"]
        v_ml = fb["volatility"]["ml"]
        v_ewma = fb["volatility"]["ewma"]
        m_ml = fb["maximum_drawdown"]["ml"]
        print(f"   --- Fold {f_num} (n={fb['n_obs']}, {fb['date_range'][0]} to {fb['date_range'][1]}):")
        print(f"       Vol ML:   MAE={v_ml['mae']:.4f} | RMSE={v_ml['rmse']:.4f} | R={v_ml['pearson_r']:.4f} | Rho={v_ml['spearman_rho']:.4f} | R2={v_ml['r2']:.4f}")
        print(f"       Vol EWMA: MAE={v_ewma['mae']:.4f} | RMSE={v_ewma['rmse']:.4f} | R={v_ewma['pearson_r']:.4f} | Rho={v_ewma['spearman_rho']:.4f} | R2={v_ewma['r2']:.4f}")
        if "baseline_comparison" in fb and baseline_comparison:
            bc_v = fb["baseline_comparison"]["volatility"]
            bc_m = fb["baseline_comparison"]["maximum_drawdown"]
            print(f"       Vol vs {baseline_comparison['baseline_label']}: dMAE={bc_v['delta_mae']:+.4f} | dRMSE={bc_v['delta_rmse']:+.4f} | Base R2={bc_v['baseline_r2']:.4f} -> ML R2={v_ml['r2']:.4f} (dR2={bc_v['delta_r2']:+.4f}) | WinRate={bc_v['win_rate_vs_base_pct']:.1f}%")
            print(f"       MaxDD vs {baseline_comparison['baseline_label']}: dMAE={bc_m['delta_mae']:+.4f} | dRMSE={bc_m['delta_rmse']:+.4f} | Base R2={bc_m['baseline_r2']:.4f} -> ML R2={m_ml['r2']:.4f} (dR2={bc_m['delta_r2']:+.4f}) | WinRate={bc_m['win_rate_vs_base_pct']:.1f}%")
        
    print(f"\n4. PREDICTION BIAS (Mean Error = Pred - True):")
    for k, v in bias_results.items():
        bias_dir = "OVER-predicts" if v['mean_error_bias'] > 0 else "UNDER-predicts"
        print(f"   {k:12s}: Bias={v['mean_error_bias']:+.4f} ({bias_dir}), Med={v['median_error']:+.4f}, Over={v['pct_over_predicting']:.1f}%, p-val={v['p_value']:.4e}")
        
    print(f"\n5. VOLATILITY ERROR BY REALIZED-VOL TERCILES:")
    for t_name, t_val in tercile_results["breakdown"].items():
        print(f"   {t_name:20s} (n={t_val['n_obs']}): ML MAE={t_val['ml_mae']:.4f}, EWMA MAE={t_val['ewma_mae']:.4f}, Lift={t_val['lift_mae']:+.4f}, ML Bias={t_val['ml_bias']:+.4f}")
        
    print(f"\n6. CHRONOLOGICAL TARGET OVERLAP VERIFICATION:")
    print(f"   All Folds Verified Zero Overlap: {fold_verification['all_folds_leakage_free']}")
    for f in fold_verification["fold_details"]:
        print(f"   Fold {f['fold']}: Train [{f['train_date_t_range'][0]} to {f['train_date_t_range'][1]} | Target_End max={f['train_target_end_max']}] -> Gap: {f['calendar_gap_days']}d -> Test [{f['test_date_t_range'][0]} to {f['test_date_t_range'][1]} | Target_Start min={f['test_target_start_min']}]")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
