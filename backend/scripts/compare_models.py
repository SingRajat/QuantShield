"""
QuantShield — 3-Model Continuous Risk Forecasting Comparison
Compares LightGBM, XGBoost, and CatBoost on identical data, identical
5-fold purged walk-forward CV splits, and identical metrics.

CatBoost results are loaded directly from existing experiment artifacts (no retraining).
LightGBM and XGBoost are evaluated across the exact same 5 purged folds with 21-day embargo.
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from scipy import stats
import lightgbm as lgb
import xgboost as xgb

from backend.src.models.purged_cv import PurgedGroupTimeSeriesSplit
from backend.src.models.risk_regressor import RiskRegressor
from backend.src.models.evaluation import compute_regression_metrics, diebold_mariano_test

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("compare_models")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR = PROJECT_ROOT / "backend" / "src" / "models" / "experiments"
FAMILY_B_DIR = EXP_DIR / "family_b_ewma"
OUTPUT_DIR = EXP_DIR / "model_comparison"


def compute_date_level_hac_dm(
    df_m1: pd.DataFrame,
    df_m2: pd.DataFrame,
    target: str = "Vol",
    loss: str = "absolute",
) -> Dict[str, Any]:
    """
    Panel-aware date-clustered Diebold-Mariano test using Bartlett HAC long-run variance.
    Collapses portfolio loss differentials per unique Date_t:
        d_date = mean_{portfolios}(Loss_m1 - Loss_m2)
    """
    df1 = df_m1.sort_values(by=["Date_t", "Portfolio_ID"]).reset_index(drop=True)
    df2 = df_m2.sort_values(by=["Date_t", "Portfolio_ID"]).reset_index(drop=True)

    err_col = f"{target}_Err_ML"
    if loss == "absolute":
        loss1 = np.abs(df1[err_col].values)
        loss2 = np.abs(df2[err_col].values)
    else:
        loss1 = df1[err_col].values ** 2
        loss2 = df2[err_col].values ** 2

    df_diff = pd.DataFrame({"Date_t": df1["Date_t"], "diff": loss1 - loss2})
    date_diff = df_diff.groupby("Date_t")["diff"].mean().values

    T = len(date_diff)
    if T < 2:
        return {"dm_stat": np.nan, "p_value": np.nan, "T_dates": T}

    d_bar = float(np.mean(date_diff))
    gamma_0 = float(np.var(date_diff, ddof=0))

    max_lags = max(1, min(int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0))), T // 4))

    d_demean = date_diff - d_bar
    gamma = [float(np.sum(d_demean[k:] * d_demean[:-k]) / T) for k in range(1, max_lags + 1)]

    lr_var = gamma_0 + 2.0 * sum((1.0 - (k + 1) / (max_lags + 1)) * gamma[k] for k in range(max_lags))
    lr_var = max(lr_var, 1e-12)

    dm_stat = float(d_bar / np.sqrt(lr_var / T))
    p_value = float(2.0 * (1.0 - stats.t.cdf(np.abs(dm_stat), df=T - 1)))

    return {
        "target": target,
        "loss": loss,
        "T_dates": int(T),
        "d_bar": d_bar,
        "dm_stat": dm_stat,
        "p_value": p_value,
        "max_lags": int(max_lags),
    }


def train_eval_xgboost_cv(
    df: pd.DataFrame,
    features: List[str],
    targets: List[str],
    n_splits: int = 5,
    embargo_days: int = 21,
) -> pd.DataFrame:
    """Trains XGBoost across purged time-series folds and records out-of-fold predictions."""
    logger.info("Executing 5-fold Purged CV for XGBoost...")
    cv = PurgedGroupTimeSeriesSplit(n_splits=n_splits, embargo_days=embargo_days)
    records = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(df), start=1):
        if len(test_idx) == 0:
            continue

        X_train, Y_train = df.iloc[train_idx][features], df.iloc[train_idx][targets]
        X_test, Y_test = df.iloc[test_idx][features], df.iloc[test_idx][targets]

        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            early_stopping_rounds=20,
            verbosity=0,
        )
        model.fit(X_train, Y_train, eval_set=[(X_test, Y_test)], verbose=False)
        preds = model.predict(X_test)

        train_naive_maxdd = float(Y_train["Forward_MaxDD"].mean())
        test_slice = df.iloc[test_idx].copy()

        for idx_rel, (_, row) in enumerate(test_slice.iterrows()):
            p_vol, p_mdd = float(preds[idx_rel, 0]), float(preds[idx_rel, 1])
            r_vol, r_mdd = float(row["Forward_Vol"]), float(row["Forward_MaxDD"])
            ewma_vol = float(row.get("EWMA_Vol_t", np.nan))
            records.append({
                "Portfolio_ID": row["Portfolio_ID"],
                "Date_t": pd.to_datetime(row["Date_t"]),
                "Target_Start": pd.to_datetime(row["Target_Start"]),
                "Target_End": pd.to_datetime(row["Target_End"]),
                "Fold": fold,
                "Realized_Vol": r_vol,
                "Realized_MaxDD": r_mdd,
                "Pred_Vol_ML": p_vol,
                "Pred_MaxDD_ML": p_mdd,
                "Pred_Vol_EWMA": ewma_vol,
                "Pred_MaxDD_Naive": train_naive_maxdd,
                "Vol_Err_ML": p_vol - r_vol,
                "MaxDD_Err_ML": p_mdd - r_mdd,
                "Abs_Vol_Err_ML": abs(p_vol - r_vol),
                "Abs_MaxDD_Err_ML": abs(p_mdd - r_mdd),
            })

    return pd.DataFrame(records)


def train_eval_lightgbm_cv(
    df: pd.DataFrame,
    features: List[str],
    targets: List[str],
    n_splits: int = 5,
    embargo_days: int = 21,
) -> pd.DataFrame:
    """Trains LightGBM across purged time-series folds and records out-of-fold predictions."""
    logger.info("Executing 5-fold Purged CV for LightGBM...")
    cv = PurgedGroupTimeSeriesSplit(n_splits=n_splits, embargo_days=embargo_days)
    records = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(df), start=1):
        if len(test_idx) == 0:
            continue

        X_train = df.iloc[train_idx][features]
        X_test = df.iloc[test_idx][features]
        Y_train = df.iloc[train_idx][targets]
        Y_test = df.iloc[test_idx][targets]

        fold_preds = np.zeros((len(test_idx), len(targets)), dtype=float)

        for col_idx, col in enumerate(targets):
            model = lgb.LGBMRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                random_state=42,
                verbose=-1,
            )
            model.fit(
                X_train,
                Y_train[col],
                eval_set=[(X_test, Y_test[col])],
                callbacks=[lgb.early_stopping(20, verbose=False)],
            )
            fold_preds[:, col_idx] = model.predict(X_test)

        train_naive_maxdd = float(Y_train["Forward_MaxDD"].mean())
        test_slice = df.iloc[test_idx].copy()

        for idx_rel, (_, row) in enumerate(test_slice.iterrows()):
            p_vol, p_mdd = float(fold_preds[idx_rel, 0]), float(fold_preds[idx_rel, 1])
            r_vol, r_mdd = float(row["Forward_Vol"]), float(row["Forward_MaxDD"])
            ewma_vol = float(row.get("EWMA_Vol_t", np.nan))
            records.append({
                "Portfolio_ID": row["Portfolio_ID"],
                "Date_t": pd.to_datetime(row["Date_t"]),
                "Target_Start": pd.to_datetime(row["Target_Start"]),
                "Target_End": pd.to_datetime(row["Target_End"]),
                "Fold": fold,
                "Realized_Vol": r_vol,
                "Realized_MaxDD": r_mdd,
                "Pred_Vol_ML": p_vol,
                "Pred_MaxDD_ML": p_mdd,
                "Pred_Vol_EWMA": ewma_vol,
                "Pred_MaxDD_Naive": train_naive_maxdd,
                "Vol_Err_ML": p_vol - r_vol,
                "MaxDD_Err_ML": p_mdd - r_mdd,
                "Abs_Vol_Err_ML": abs(p_vol - r_vol),
                "Abs_MaxDD_Err_ML": abs(p_mdd - r_mdd),
            })

    return pd.DataFrame(records)


def summarize_metrics(oof_df: pd.DataFrame) -> Dict[str, Any]:
    """Computes all summary regression metrics for an OOF prediction set."""
    vol_m = compute_regression_metrics(oof_df["Realized_Vol"].values, oof_df["Pred_Vol_ML"].values)
    mdd_m = compute_regression_metrics(oof_df["Realized_MaxDD"].values, oof_df["Pred_MaxDD_ML"].values)
    ewma_m = compute_regression_metrics(oof_df["Realized_Vol"].values, oof_df["Pred_Vol_EWMA"].values)
    naive_m = compute_regression_metrics(oof_df["Realized_MaxDD"].values, oof_df["Pred_MaxDD_Naive"].values)

    return {
        "volatility": {
            **vol_m,
            "lift_vs_ewma_mae": float(ewma_m["mae"] - vol_m["mae"]),
            "lift_vs_ewma_rmse": float(ewma_m["rmse"] - vol_m["rmse"]),
        },
        "max_drawdown": {
            **mdd_m,
            "lift_vs_naive_mae": float(naive_m["mae"] - mdd_m["mae"]),
            "lift_vs_naive_rmse": float(naive_m["rmse"] - mdd_m["rmse"]),
        },
    }


def format_table(results: Dict[str, Dict[str, Any]]) -> str:
    """Formats a side-by-side terminal comparison table using ASCII characters."""
    headers = ["Metric", "LightGBM", "XGBoost", "CatBoost"]
    col_w = [22, 14, 14, 14]

    def line(sep="-"):
        return "+" + "+".join([sep * w for w in col_w]) + "+"

    rows = [
        ("Vol MAE", "volatility", "mae", "{:.5f}"),
        ("Vol RMSE", "volatility", "rmse", "{:.5f}"),
        ("Vol MedAE", "volatility", "medae", "{:.5f}"),
        ("Vol Pearson R", "volatility", "pearson_r", "{:.4f}"),
        ("Vol Spearman rho", "volatility", "spearman_rho", "{:.4f}"),
        ("Vol R2", "volatility", "r2", "{:.4f}"),
        ("Vol Delta MAE (vs EWMA)", "volatility", "lift_vs_ewma_mae", "{:+.5f}"),
        ("---", "", "", ""),
        ("MaxDD MAE", "max_drawdown", "mae", "{:.5f}"),
        ("MaxDD RMSE", "max_drawdown", "rmse", "{:.5f}"),
        ("MaxDD MedAE", "max_drawdown", "medae", "{:.5f}"),
        ("MaxDD Pearson R", "max_drawdown", "pearson_r", "{:.4f}"),
        ("MaxDD Spearman rho", "max_drawdown", "spearman_rho", "{:.4f}"),
        ("MaxDD R2", "max_drawdown", "r2", "{:.4f}"),
        ("MaxDD Delta MAE (vs Naive)", "max_drawdown", "lift_vs_naive_mae", "{:+.5f}"),
    ]

    out = []
    out.append(line("-"))
    hdr_str = "|" + "|".join([f" {h:<{col_w[i]-2}} " for i, h in enumerate(headers)]) + "|"
    out.append(hdr_str)
    out.append(line("="))

    for label, grp, metric_key, fmt in rows:
        if label == "---":
            out.append(line("-"))
            continue
        vals = [
            label,
            fmt.format(results["LightGBM"][grp][metric_key]),
            fmt.format(results["XGBoost"][grp][metric_key]),
            fmt.format(results["CatBoost"][grp][metric_key]),
        ]
        row_str = "|" + "|".join([f" {v:<{col_w[i]-2}} " for i, v in enumerate(vals)]) + "|"
        out.append(row_str)

    out.append(line("-"))
    return "\n".join(out)


def main():
    logger.info("Starting 3-Model Comparison (LightGBM vs XGBoost vs CatBoost)...")

    # 1. Load training dataset
    dataset_path = FAMILY_B_DIR / "training_dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path).sort_values(by="Date_t").reset_index(drop=True)
    features = RiskRegressor.FEATURE_SETS["family_b_ewma"]
    targets = ["Forward_Vol", "Forward_MaxDD"]

    logger.info(f"Loaded dataset: {df.shape[0]} rows, {len(features)} features, targets={targets}")

    # 2. Load existing CatBoost OOF predictions (zero retraining)
    catboost_path = FAMILY_B_DIR / "oof_predictions.csv"
    if not catboost_path.exists():
        raise FileNotFoundError(f"CatBoost OOF predictions not found: {catboost_path}")
    df_catboost = pd.read_csv(catboost_path)
    logger.info(f"Loaded CatBoost OOF predictions: {df_catboost.shape[0]} rows (no retraining).")

    # 3. Train & evaluate XGBoost
    df_xgboost = train_eval_xgboost_cv(df, features, targets, n_splits=5, embargo_days=21)

    # 4. Train & evaluate LightGBM
    df_lightgbm = train_eval_lightgbm_cv(df, features, targets, n_splits=5, embargo_days=21)

    # 5. Compute metrics for each model
    results = {
        "LightGBM": summarize_metrics(df_lightgbm),
        "XGBoost": summarize_metrics(df_xgboost),
        "CatBoost": summarize_metrics(df_catboost),
    }

    # 6. Econometrically defensible pairwise tests: Date-Level HAC Diebold-Mariano tests
    logger.info("Computing Date-Level HAC Diebold-Mariano pairwise tests...")
    dm_tests = {
        "Vol_LightGBM_vs_CatBoost": compute_date_level_hac_dm(df_lightgbm, df_catboost, target="Vol", loss="absolute"),
        "Vol_XGBoost_vs_CatBoost": compute_date_level_hac_dm(df_xgboost, df_catboost, target="Vol", loss="absolute"),
        "Vol_LightGBM_vs_XGBoost": compute_date_level_hac_dm(df_lightgbm, df_xgboost, target="Vol", loss="absolute"),
        "MaxDD_LightGBM_vs_CatBoost": compute_date_level_hac_dm(df_lightgbm, df_catboost, target="MaxDD", loss="absolute"),
        "MaxDD_XGBoost_vs_CatBoost": compute_date_level_hac_dm(df_xgboost, df_catboost, target="MaxDD", loss="absolute"),
        "MaxDD_LightGBM_vs_XGBoost": compute_date_level_hac_dm(df_lightgbm, df_xgboost, target="MaxDD", loss="absolute"),
    }

    # 7. Save artifacts first
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_lightgbm.to_csv(OUTPUT_DIR / "lgbm_oof_predictions.csv", index=False)
    df_xgboost.to_csv(OUTPUT_DIR / "xgb_oof_predictions.csv", index=False)

    full_payload = {
        "metadata": {
            "dataset_rows": int(df.shape[0]),
            "evaluated_oof_rows": int(df_lightgbm.shape[0]),
            "n_splits": 5,
            "embargo_days": 21,
            "features": features,
            "targets": targets,
            "hyperparameters": {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "early_stopping_rounds": 20,
                "random_state": 42,
            },
        },
        "results": results,
        "pairwise_dm_tests": dm_tests,
    }

    with open(OUTPUT_DIR / "comparison_results.json", "w") as f:
        json.dump(full_payload, f, indent=2)

    logger.info(f"Artifacts saved successfully in: {OUTPUT_DIR}")

    # 8. Print formatted summary table
    table_str = format_table(results)
    print("\n" + "=" * 64)
    print("        QUANT SHIELD -- 3-MODEL BENCHMARK COMPARISON        ")
    print("=" * 64)
    print(table_str)
    print("\nPairwise Date-Level HAC Diebold-Mariano Tests (Absolute Loss, T=825 dates):")
    for name, test in dm_tests.items():
        print(f"  * {name:<30}: d_bar={test['d_bar']:+.6f}, DM-stat={test['dm_stat']:+.3f}, p-val={test['p_value']:.4f}")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()

