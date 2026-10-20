import os
import sys
import json
import logging
from pathlib import Path
import pandas as pd
import yfinance as yf

# Add project root and backend to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'backend'))

from backend.src.data.etf_ingestion import ETFDataFetcher
from backend.src.features.portfolio_builder import PortfolioBuilder
from backend.src.features.dataset_builder import DatasetBuilder
from backend.src.models.risk_regressor import RiskRegressor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mock portfolios using liquid, historic Indian equities listed pre-2005/2010
MOCK_PORTFOLIOS = [
    {"etf_name": "NIFTY_IT_ETF", "holdings": [{"ticker": "TCS.NS", "weight": 0.3}, {"ticker": "INFY.NS", "weight": 0.3}, {"ticker": "HCLTECH.NS", "weight": 0.2}, {"ticker": "WIPRO.NS", "weight": 0.2}]},
    {"etf_name": "NIFTY_BANK_ETF", "holdings": [{"ticker": "HDFCBANK.NS", "weight": 0.3}, {"ticker": "ICICIBANK.NS", "weight": 0.3}, {"ticker": "SBIN.NS", "weight": 0.2}, {"ticker": "AXISBANK.NS", "weight": 0.2}]},
    {"etf_name": "NIFTY_AUTO_ETF", "holdings": [{"ticker": "TATAMOTORS.NS", "weight": 0.3}, {"ticker": "M&M.NS", "weight": 0.2}, {"ticker": "MARUTI.NS", "weight": 0.3}, {"ticker": "ASHOKLEY.NS", "weight": 0.2}]},
    {"etf_name": "NIFTY_PHARMA_ETF", "holdings": [{"ticker": "SUNPHARMA.NS", "weight": 0.3}, {"ticker": "CIPLA.NS", "weight": 0.3}, {"ticker": "DRREDDY.NS", "weight": 0.2}, {"ticker": "DIVISLAB.NS", "weight": 0.2}]},
    {"etf_name": "NIFTY_FMCG_ETF", "holdings": [{"ticker": "ITC.NS", "weight": 0.4}, {"ticker": "HINDUNILVR.NS", "weight": 0.3}, {"ticker": "BRITANNIA.NS", "weight": 0.2}, {"ticker": "TATACONSUM.NS", "weight": 0.1}]},
    {"etf_name": "NIFTY_METAL_ETF", "holdings": [{"ticker": "TATASTEEL.NS", "weight": 0.3}, {"ticker": "HINDALCO.NS", "weight": 0.3}, {"ticker": "JSWSTEEL.NS", "weight": 0.2}, {"ticker": "SAIL.NS", "weight": 0.2}]},
    {"etf_name": "NIFTY_ENERGY_ETF", "holdings": [{"ticker": "RELIANCE.NS", "weight": 0.4}, {"ticker": "NTPC.NS", "weight": 0.2}, {"ticker": "ONGC.NS", "weight": 0.2}, {"ticker": "POWERGRID.NS", "weight": 0.2}]},
    {"etf_name": "CONSERVATIVE_DEBT_PROXY_ETF", "holdings": [{"ticker": "HDFCBANK.NS", "weight": 0.4}, {"ticker": "ITC.NS", "weight": 0.3}, {"ticker": "INFY.NS", "weight": 0.3}]},
    {"etf_name": "AGGRESSIVE_GROWTH_ETF", "holdings": [{"ticker": "TITAN.NS", "weight": 0.3}, {"ticker": "ASIANPAINT.NS", "weight": 0.3}, {"ticker": "BAJFINANCE.NS", "weight": 0.2}, {"ticker": "EICHERMOT.NS", "weight": 0.2}]},
    {"etf_name": "HIGH_DIVIDEND_YIELD_ETF", "holdings": [{"ticker": "ONGC.NS", "weight": 0.3}, {"ticker": "PFC.NS", "weight": 0.2}, {"ticker": "RECLTD.NS", "weight": 0.2}, {"ticker": "BPCL.NS", "weight": 0.3}]},
    {"etf_name": "NIFTY_FINANCIAL_SERVICES_ETF", "holdings": [{"ticker": "BAJFINANCE.NS", "weight": 0.2}, {"ticker": "CHOLAFIN.NS", "weight": 0.2}, {"ticker": "HDFCBANK.NS", "weight": 0.3}, {"ticker": "ICICIBANK.NS", "weight": 0.3}]},
    {"etf_name": "REALTY_INFRA_ETF", "holdings": [{"ticker": "ULTRACEMCO.NS", "weight": 0.3}, {"ticker": "L&T.NS", "weight": 0.4}, {"ticker": "AMBUJACEM.NS", "weight": 0.15}, {"ticker": "SHREECEM.NS", "weight": 0.15}]},
    {"etf_name": "MID_CAP_BLEND_ETF", "holdings": [{"ticker": "TVSMOTOR.NS", "weight": 0.3}, {"ticker": "CUMMINSIND.NS", "weight": 0.3}, {"ticker": "FEDERALBNK.NS", "weight": 0.2}, {"ticker": "TATACHEM.NS", "weight": 0.2}]},
    {"etf_name": "LARGECAP_50_MOCK_ETF", "holdings": [{"ticker": "HDFCBANK.NS", "weight": 0.2}, {"ticker": "RELIANCE.NS", "weight": 0.2}, {"ticker": "ICICIBANK.NS", "weight": 0.2}, {"ticker": "INFY.NS", "weight": 0.2}, {"ticker": "L&T.NS", "weight": 0.2}]},
    {"etf_name": "TATA_CONGLOMERATE_ETF", "holdings": [{"ticker": "TCS.NS", "weight": 0.3}, {"ticker": "TATAMOTORS.NS", "weight": 0.3}, {"ticker": "TATASTEEL.NS", "weight": 0.2}, {"ticker": "TITAN.NS", "weight": 0.2}]},
    {"etf_name": "NIFTY_MNC_PROXY", "holdings": [{"ticker": "NESTLEIND.NS", "weight": 0.3}, {"ticker": "MARUTI.NS", "weight": 0.3}, {"ticker": "COLPAL.NS", "weight": 0.2}, {"ticker": "BATAINDIA.NS", "weight": 0.2}]},
    {"etf_name": "NIFTY_CPSE_PROXY", "holdings": [{"ticker": "NTPC.NS", "weight": 0.3}, {"ticker": "ONGC.NS", "weight": 0.3}, {"ticker": "POWERGRID.NS", "weight": 0.2}, {"ticker": "BEL.NS", "weight": 0.2}]},
    {"etf_name": "NIFTY_CONSUMPTION_PROXY", "holdings": [{"ticker": "TITAN.NS", "weight": 0.3}, {"ticker": "ASIANPAINT.NS", "weight": 0.3}, {"ticker": "ITC.NS", "weight": 0.2}, {"ticker": "GODREJCP.NS", "weight": 0.2}]},
    {"etf_name": "NIFTY_COMMODITIES_PROXY", "holdings": [{"ticker": "ULTRACEMCO.NS", "weight": 0.3}, {"ticker": "GRASIM.NS", "weight": 0.3}, {"ticker": "UPL.NS", "weight": 0.2}, {"ticker": "PIDILITIND.NS", "weight": 0.2}]},
    {"etf_name": "PSU_BANK_PROXY", "holdings": [{"ticker": "SBIN.NS", "weight": 0.4}, {"ticker": "PNB.NS", "weight": 0.2}, {"ticker": "BANKBARODA.NS", "weight": 0.2}, {"ticker": "CANBK.NS", "weight": 0.2}]},
    {"etf_name": "PRIVATE_BANK_PROXY", "holdings": [{"ticker": "HDFCBANK.NS", "weight": 0.3}, {"ticker": "ICICIBANK.NS", "weight": 0.3}, {"ticker": "KOTAKBANK.NS", "weight": 0.2}, {"ticker": "INDUSINDBK.NS", "weight": 0.2}]},
    {"etf_name": "CAPITAL_GOODS_PROXY", "holdings": [{"ticker": "L&T.NS", "weight": 0.4}, {"ticker": "BHARATFORG.NS", "weight": 0.2}, {"ticker": "SIEMENS.NS", "weight": 0.2}, {"ticker": "THERMAX.NS", "weight": 0.2}]},
    {"etf_name": "HEALTHCARE_PROXY", "holdings": [{"ticker": "APOLLOHOSP.NS", "weight": 0.3}, {"ticker": "BIOCON.NS", "weight": 0.3}, {"ticker": "PEL.NS", "weight": 0.2}, {"ticker": "GLENMARK.NS", "weight": 0.2}]},
    {"etf_name": "CONSUMER_DURABLES_PROXY", "holdings": [{"ticker": "VOLTAS.NS", "weight": 0.3}, {"ticker": "HAVELLS.NS", "weight": 0.3}, {"ticker": "WHIRLPOOL.NS", "weight": 0.2}, {"ticker": "BLUESTARCO.NS", "weight": 0.2}]},
    {"etf_name": "OIL_GAS_PROXY", "holdings": [{"ticker": "RELIANCE.NS", "weight": 0.4}, {"ticker": "GAIL.NS", "weight": 0.3}, {"ticker": "IGL.NS", "weight": 0.2}, {"ticker": "PETRONET.NS", "weight": 0.1}]},
    {"etf_name": "MEDIA_ENTERTAINMENT_PROXY", "holdings": [{"ticker": "SUNTV.NS", "weight": 0.4}, {"ticker": "ZEEL.NS", "weight": 0.4}, {"ticker": "TV18BRDCST.NS", "weight": 0.2}]},
    {"etf_name": "ESG_LEADERS_PROXY", "holdings": [{"ticker": "TCS.NS", "weight": 0.3}, {"ticker": "INFY.NS", "weight": 0.3}, {"ticker": "WIPRO.NS", "weight": 0.2}, {"ticker": "KOTAKBANK.NS", "weight": 0.2}]},
    {"etf_name": "ADANI_GROUP_CONGLOMERATE", "holdings": [{"ticker": "ADANIENT.NS", "weight": 0.4}, {"ticker": "AMBUJACEM.NS", "weight": 0.3}, {"ticker": "ACC.NS", "weight": 0.3}]},
    {"etf_name": "MURUGAPPA_TVS_PROXY", "holdings": [{"ticker": "CHOLAFIN.NS", "weight": 0.3}, {"ticker": "TVSMOTOR.NS", "weight": 0.3}, {"ticker": "COROMANDEL.NS", "weight": 0.2}, {"ticker": "CARBORUNUNIV.NS", "weight": 0.2}]},
    {"etf_name": "BROKING_FINANCIALS_PROXY", "holdings": [{"ticker": "MOTILALOFS.NS", "weight": 0.3}, {"ticker": "EDELWEISS.NS", "weight": 0.3}, {"ticker": "JMFINANCIL.NS", "weight": 0.2}, {"ticker": "GEOJITFSL.NS", "weight": 0.2}]}
]

for port in MOCK_PORTFOLIOS:
    port["reporting_date"] = "2023-10-31"

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="QuantShield Risk Forecasting Pipeline")
    parser.add_argument(
        "--feature-set",
        type=str,
        default="v1_baseline",
        choices=["v1_baseline", "family_a", "family_b", "v1_1", "family_b_ewma", "family_b_plus_ewma"],
        help="Feature set configuration ('v1_baseline' / 'family_a', 'family_b' / 'v1_1', or 'family_b_ewma')"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save artifacts. Defaults to backend/src/models/experiments/{feature_set}"
    )
    parser.add_argument(
        "--from-base-dataset",
        type=str,
        default=None,
        help="Path to existing training_dataset.csv to derive features from without refetching from yfinance"
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        default=False,
        help="Atomically promote validated model and calibration bundle to production (backend/src/models/)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info(f"Starting QuantShield Continuous Risk Forecasting Pipeline [Feature Set: {args.feature_set}]...")

    # Determine output directory (QS-R1 Fix: Always isolate experiment artifacts; never default or write directly into production)
    prod_dir = (project_root / 'backend' / 'src' / 'models').resolve()
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
        if output_dir == prod_dir:
            raise ValueError(
                "Direct write to production directory (backend/src/models) via --output-dir is prohibited. "
                "Train into an experiment or staging directory, and pass --promote after audit validation passes."
            )
    elif args.feature_set in ["family_b", "v1_1"]:
        output_dir = project_root / 'backend' / 'src' / 'models' / 'experiments' / 'family_b'
    elif args.feature_set in ["family_b_ewma", "family_b_plus_ewma"]:
        output_dir = project_root / 'backend' / 'src' / 'models' / 'experiments' / 'family_b_ewma'
    elif args.feature_set in ["v1_baseline", "family_a"]:
        output_dir = project_root / 'backend' / 'src' / 'models' / 'experiments' / 'v1_baseline'
    else:
        output_dir = project_root / 'backend' / 'src' / 'models' / 'experiments' / args.feature_set

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Artifacts will be saved to: {output_dir}")

    # Dataset Construction
    if args.from_base_dataset:
        base_path = Path(args.from_base_dataset)
        logger.info(f"Loading base dataset from: {base_path}")
        if not base_path.exists():
            raise FileNotFoundError(f"Base dataset not found at {base_path}")
        base_df = pd.read_csv(base_path)
        base_df["Date_t"] = pd.to_datetime(base_df["Date_t"])
        base_df["Target_Start"] = pd.to_datetime(base_df["Target_Start"])
        base_df["Target_End"] = pd.to_datetime(base_df["Target_End"])

        if args.feature_set in ["family_b", "v1_1", "family_b_ewma", "family_b_plus_ewma"]:
            if "Delta_Vol_5" not in base_df.columns:
                panel_df = DatasetBuilder.add_family_b_features(base_df)
            else:
                panel_df = base_df
        else:
            panel_df = base_df
    else:
        # 1. Ingestion Phase - 20 YEARS
        fetcher = ETFDataFetcher(years=20)
        portfolios = {}
        component_returns_dict = {}
        weights_dict = {}
        
        for port_def in MOCK_PORTFOLIOS:
            etf_name = port_def["etf_name"]
            logger.info(f"Fetching data for portfolio: {etf_name}")
            try:
                output = fetcher.fetch_data(port_def)
                price_data = output["price_data"]
                weights = output["weights"]
                
                # 2. Portfolio Building Phase
                builder = PortfolioBuilder(price_data=price_data)
                portfolio_df = builder.build_portfolio(weights)
                portfolios[etf_name] = portfolio_df
                component_returns_dict[etf_name] = builder.daily_returns
                weights_dict[etf_name] = weights
            except Exception as e:
                logger.error(f"Failed to process portfolio {etf_name}: {e}")

        if not portfolios:
            logger.error("No valid portfolios were processed. Exiting.")
            return

        # 3. Build Panel Dataset
        logger.info(f"Initializing DatasetBuilder (feature_set={args.feature_set})...")
        dataset_builder = DatasetBuilder(portfolios=portfolios, feature_set=args.feature_set)
        panel_df = dataset_builder.build_panel_dataset(drop_incomplete=True, feature_set=args.feature_set)

    d_min = panel_df['Date_t'].min()
    d_max = panel_df['Date_t'].max()
    d_min_str = d_min.strftime('%Y-%m-%d') if hasattr(d_min, 'strftime') else str(d_min)[:10]
    d_max_str = d_max.strftime('%Y-%m-%d') if hasattr(d_max, 'strftime') else str(d_max)[:10]
    logger.info(f"Date range: {d_min_str} to {d_max_str}")

    # Save the dataset to the designated output directory
    dataset_path = output_dir / 'training_dataset.csv'
    panel_df.to_csv(dataset_path, index=False)
    logger.info(f"Saved dataset to: {dataset_path}")

    # 4. Multi-Output Continuous Risk Regressor Training with Purged Walk-Forward CV
    logger.info(f"Initializing Multi-Output RiskRegressor (feature_set={args.feature_set})...")
    regressor = RiskRegressor(
        iterations=300, 
        depth=6, 
        learning_rate=0.05, 
        random_seed=42, 
        feature_set=args.feature_set
    )
    
    # 5 splits, 21d embargo after max forward target end date
    eval_results = regressor.train_and_evaluate(panel_df=panel_df, n_splits=5, embargo_days=21)
    
    vol_res = eval_results["vol_forecast"]
    mdd_res = eval_results["maxdd_forecast"]
    
    logger.info(
        f"\n=======================================================\n"
        f"Purged Walk-Forward Continuous CV Results ({eval_results['total_evaluated_observations']} evaluated observations):\n"
        f"  Feature Set:                    {args.feature_set} ({len(regressor.features)} features)\n"
        f"  ML Vol Forecast OOF MAE:        {vol_res['oof_mae']:.4f}\n"
        f"  ML Vol Forecast OOF RMSE:       {vol_res['oof_rmse']:.4f}\n"
        f"  ML Vol Forecast Correlation:    {vol_res['oof_correlation']:.4f}\n"
        f"  EWMA Benchmark OOF MAE:         {vol_res.get('ewma_baseline_mae', float('nan')):.4f}\n"
        f"  EWMA Benchmark OOF RMSE:        {vol_res.get('ewma_baseline_rmse', float('nan')):.4f}\n"
        f"  EWMA Benchmark Correlation:     {vol_res.get('ewma_baseline_correlation', float('nan')):.4f}\n"
        f"  Vol MAE Lift vs EWMA:           {vol_res.get('lift_vs_ewma_mae', float('nan')):+.4f}\n"
        f"-------------------------------------------------------\n"
        f"  ML MaxDD Forecast OOF MAE:      {mdd_res['oof_mae']:.4f}\n"
        f"  ML MaxDD Forecast OOF RMSE:     {mdd_res['oof_rmse']:.4f}\n"
        f"  ML MaxDD Forecast Correlation:  {mdd_res['oof_correlation']:.4f}\n"
        f"======================================================="
    )

    # 5. Serialization Integration
    model_path = output_dir / 'saved_model.cbm'
    regressor.model.save_model(str(model_path))
    logger.info(f"Serialized trained CatBoost MultiRMSE model to: {model_path}")

    # QS-R8: Generate immutable artifact ID from model binary hash
    import hashlib
    with open(model_path, 'rb') as f:
        artifact_id = hashlib.sha256(f.read()).hexdigest()[:12]

    # Save experiment config
    config_payload = {
        "experiment_name": f"Experiment_{args.feature_set}",
        "feature_set": args.feature_set,
        "artifact_id": artifact_id,
        "feature_count": len(regressor.features),
        "features": regressor.features,
        "targets": RiskRegressor.TARGETS,
        "forecast_horizon_days": DatasetBuilder.FORECAST_HORIZON,
        "iterations": 300,
        "depth": 6,
        "learning_rate": 0.05,
        "early_stopping_rounds": 20,
        "n_splits": 5,
        "embargo_days": 21,
        "random_seed": 42
    }
    config_path = output_dir / 'experiment_config.json'
    with open(config_path, 'w') as f:
        json.dump(config_payload, f, indent=2)
    logger.info(f"Saved experiment configuration to: {config_path}")

    # Save model transparency metadata for API & UI consumption
    metadata_path = output_dir / 'model_metadata.json'
    meta_payload = {
        "artifact_id": artifact_id,
        "build_timestamp": pd.Timestamp.now().isoformat(),
        "model_version": "V1.2",
        "evaluation_protocol": "5-fold chronological walk-forward out-of-fold cross-validation",
        "features": regressor.features,
        "targets": RiskRegressor.TARGETS,
        "forecast_horizon_days": DatasetBuilder.FORECAST_HORIZON,
        "total_evaluated_observations": eval_results["total_evaluated_observations"],
        "vol_forecast": vol_res,
        "maxdd_forecast": mdd_res,
        "folds": eval_results["folds"]
    }
    with open(metadata_path, 'w') as f:
        json.dump(meta_payload, f, indent=2)
    logger.info(f"Saved continuous model evaluation metadata to: {metadata_path}")

    # 6. Fit & Export Calibration Artifacts (Phase 4.1 & Walk-Forward OOF Validation)
    if "oof_predictions" in eval_results:
        from sklearn.linear_model import LogisticRegression
        import numpy as np

        oof_data = eval_results["oof_predictions"]
        true_vol = np.array(oof_data["true_vol"])
        pred_vol = np.array(oof_data["pred_vol"])
        true_maxdd = np.array(oof_data["true_maxdd"])
        pred_maxdd = np.array(oof_data["pred_maxdd"])
        fold_ids = np.array(oof_data.get("fold_id", []))

        # QS-R3: Chronological Walk-Forward Final-Fold Out-of-Fold Evaluation Partition
        # Development folds: Folds 1 to K-1 (e.g. Folds 1-4) used for fitting calibration parameters
        # Final evaluation fold: Final Fold K (e.g. Fold 5) reserved for walk-forward OOF evaluation
        unique_folds = np.unique(fold_ids) if len(fold_ids) > 0 else np.array([])
        if len(unique_folds) > 1:
            eval_fold = unique_folds[-1]
            dev_mask = fold_ids != eval_fold
            eval_mask = fold_ids == eval_fold
            logger.info(f"Calibration partition: Folds {unique_folds[:-1].tolist()} used for fitting (N={np.sum(dev_mask)}), Fold {eval_fold} reserved for walk-forward OOF evaluation (N={np.sum(eval_mask)})")
        else:
            # Fallback to 80/20 chronological split if fold_ids not present
            n_samples = len(true_vol)
            split_idx = int(n_samples * 0.8)
            dev_mask = np.zeros(n_samples, dtype=bool)
            dev_mask[:split_idx] = True
            eval_mask = ~dev_mask
            logger.info(f"Calibration partition: Chronological 80/20 split used (dev N={np.sum(dev_mask)}, eval N={np.sum(eval_mask)})")

        # Fit empirical quantiles strictly on development fold residuals
        dev_err_vol = pred_vol[dev_mask] - true_vol[dev_mask]
        dev_err_maxdd = pred_maxdd[dev_mask] - true_maxdd[dev_mask]

        vol_q05 = float(np.percentile(dev_err_vol, 5))
        vol_q95 = float(np.percentile(dev_err_vol, 95))
        maxdd_q05 = float(np.percentile(dev_err_maxdd, 5))
        maxdd_q95 = float(np.percentile(dev_err_maxdd, 95))

        # Fit Platt scaling (multinomial logistic regression) strictly on development fold predictions
        dev_regimes = np.where(true_vol[dev_mask] < 0.16, 0, np.where(true_vol[dev_mask] < 0.25, 1, 2))
        
        # QS-R7 Fix: Verify all 3 regimes exist in development folds before multinomial fitting
        unique_dev_regimes = np.unique(dev_regimes)
        if len(unique_dev_regimes) < 3:
            raise ValueError(
                f"Calibration error: Development folds contain only {len(unique_dev_regimes)} regime class(es) "
                f"({unique_dev_regimes.tolist()}), but 3 are required for 3-class multinomial calibration."
            )

        clf = LogisticRegression(solver="lbfgs", max_iter=1000)
        clf.fit(pred_vol[dev_mask].reshape(-1, 1), dev_regimes)

        classes = ["Low", "Medium", "High"]
        calib_coeffs = {}
        for i, c in enumerate(classes):
            calib_coeffs[c] = {
                "intercept": float(clf.intercept_[i]),
                "slope": float(clf.coef_[i][0])
            }

        # Evaluate empirical coverage strictly on the walk-forward evaluation fold
        h_true_vol = true_vol[eval_mask]
        h_pred_vol = pred_vol[eval_mask]
        h_true_maxdd = true_maxdd[eval_mask]
        h_pred_maxdd = pred_maxdd[eval_mask]

        vol_cov_oof = float(np.mean((h_true_vol >= (h_pred_vol - vol_q95)) & (h_true_vol <= (h_pred_vol - vol_q05))))
        mdd_cov_oof = float(np.mean((h_true_maxdd >= (h_pred_maxdd - maxdd_q95)) & (h_true_maxdd <= (h_pred_maxdd - maxdd_q05))))

        # Evaluate Brier Score, Multiclass Log Loss, and ECE on evaluation fold
        h_regimes = np.where(h_true_vol < 0.16, 0, np.where(h_true_vol < 0.25, 1, 2))
        h_probs = clf.predict_proba(h_pred_vol.reshape(-1, 1))

        n_h = len(h_regimes)
        y_one_hot = np.zeros((n_h, 3))
        for i, r in enumerate(h_regimes):
            y_one_hot[i, r] = 1.0

        brier_score = float(np.mean(np.sum((h_probs - y_one_hot) ** 2, axis=1)))

        dev_priors = np.bincount(dev_regimes, minlength=3) / len(dev_regimes)
        heuristic_probs = np.tile(dev_priors, (n_h, 1))
        heuristic_brier = float(np.mean(np.sum((heuristic_probs - y_one_hot) ** 2, axis=1)))
        brier_lift = brier_score - heuristic_brier

        eps = 1e-15
        h_probs_clipped = np.clip(h_probs, eps, 1.0 - eps)
        log_loss_score = float(-np.mean(np.sum(y_one_hot * np.log(h_probs_clipped), axis=1)))
        heuristic_probs_clipped = np.clip(heuristic_probs, eps, 1.0 - eps)
        heuristic_log_loss = float(-np.mean(np.sum(y_one_hot * np.log(heuristic_probs_clipped), axis=1)))
        log_loss_lift = log_loss_score - heuristic_log_loss

        # Expected Calibration Error (ECE) across 10 confidence bins
        confidences = np.max(h_probs, axis=1)
        predictions = np.argmax(h_probs, axis=1)
        accuracies = (predictions == h_regimes)
        bins = np.linspace(0, 1, 11)
        ece = 0.0
        for b_idx in range(len(bins) - 1):
            bin_mask = (confidences > bins[b_idx]) & (confidences <= bins[b_idx + 1])
            if np.any(bin_mask):
                bin_acc = np.mean(accuracies[bin_mask])
                bin_conf = np.mean(confidences[bin_mask])
                ece += (np.sum(bin_mask) / n_h) * abs(bin_acc - bin_conf)
        ece = float(ece)

        # QS-R4 Fix: Enforce multi-criteria audit gate:
        # 1. Empirical coverage for both Volatility and Max Drawdown within [0.85, 0.95]
        # 2. Probability calibration Brier score improves upon naive baseline (brier_lift <= 0.0)
        # 3. Expected Calibration Error (ECE) is contained within acceptable tolerance (ECE <= 0.15)
        coverage_passed = (0.85 <= vol_cov_oof <= 0.95) and (0.85 <= mdd_cov_oof <= 0.95)
        calibration_passed = (brier_lift <= 0.0) and (ece <= 0.15)
        validation_status = "AUDIT_PASSED" if (coverage_passed and calibration_passed) else "REVIEW_NEEDED"

        calib_payload = {
            "method": "5-Fold Walk-Forward Out-Of-Fold Softmax Calibration & Inverted Residual Corridor",
            "artifact_id": artifact_id,
            "coefficients": calib_coeffs,
            "residual_quantiles": {
                "vol_err_q05": vol_q05,
                "vol_err_q95": vol_q95,
                "maxdd_err_q05": maxdd_q05,
                "maxdd_err_q95": maxdd_q95
            },
            "metrics": {
                "sample_size_total": len(true_vol),
                "sample_size_dev": int(np.sum(dev_mask)),
                "sample_size_oof_eval": int(n_h),
                "vol_empirical_coverage": vol_cov_oof,
                "maxdd_empirical_coverage": mdd_cov_oof,
                "target_coverage": 0.90,
                "acceptable_corridor": [0.85, 0.95],
                "brier_score": round(brier_score, 4),
                "heuristic_brier_score": round(heuristic_brier, 4),
                "brier_improvement_absolute": round(brier_lift, 4),
                "log_loss": round(log_loss_score, 4),
                "heuristic_log_loss": round(heuristic_log_loss, 4),
                "log_loss_improvement_absolute": round(log_loss_lift, 4),
                "expected_calibration_error": round(ece, 4),
                "evaluation_protocol": "Chronological Walk-Forward Final-Fold OOF Validation",
                "validation_status": validation_status
            }
        }

        calib_path = output_dir / 'calibration.json'
        with open(calib_path, 'w') as f:
            json.dump(calib_payload, f, indent=2)
        logger.info(f"Saved calibration artifact to: {calib_path}")

        # QS-R1 & QS-R5 Fix: Guarded promotion to production root with versioned bundle & active manifest
        prod_dir = (project_root / 'backend' / 'src' / 'models').resolve()
        if args.promote and prod_dir.exists() and output_dir != prod_dir:
            if validation_status != "AUDIT_PASSED":
                logger.warning(
                    f"Production promotion blocked: validation_status is '{validation_status}', not 'AUDIT_PASSED'. "
                    f"Metrics: Vol_Cov={vol_cov_oof:.4f}, MaxDD_Cov={mdd_cov_oof:.4f}, Brier_Lift={brier_lift:.4f}, ECE={ece:.4f}."
                )
            else:
                import shutil
                # 1. Archive into immutable versioned bundle directory
                timestamp_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                version_dir = prod_dir / 'versions' / f"{timestamp_str}_{artifact_id}"
                version_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(model_path, version_dir / 'saved_model.cbm')
                shutil.copy2(calib_path, version_dir / 'calibration.json')
                shutil.copy2(metadata_path, version_dir / 'model_metadata.json')

                # 2. Stage active files and manifest into production root
                staging_dir = prod_dir / '.staging_bundle'
                staging_dir.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(model_path, staging_dir / 'saved_model.cbm')
                    shutil.copy2(calib_path, staging_dir / 'calibration.json')
                    shutil.copy2(metadata_path, staging_dir / 'model_metadata.json')
                    manifest_payload = {
                        "active_version": version_dir.name,
                        "artifact_id": artifact_id,
                        "promoted_at": pd.Timestamp.now().isoformat(),
                        "feature_set": args.feature_set,
                        "vol_empirical_coverage": vol_cov_oof,
                        "maxdd_empirical_coverage": mdd_cov_oof,
                        "brier_improvement_absolute": round(brier_lift, 4),
                        "expected_calibration_error": round(ece, 4),
                        "validation_status": validation_status
                    }
                    with open(staging_dir / 'active_manifest.json', 'w') as mf:
                        json.dump(manifest_payload, mf, indent=2)

                    for item in ['saved_model.cbm', 'calibration.json', 'model_metadata.json', 'active_manifest.json']:
                        (staging_dir / item).replace(prod_dir / item)
                    logger.info(f"Promoted versioned bundle {version_dir.name} and updated active manifest in {prod_dir}")
                finally:
                    if staging_dir.exists():
                        shutil.rmtree(staging_dir, ignore_errors=True)
        elif not args.promote and output_dir != prod_dir:
            logger.info("Artifacts preserved in experiment directory. Specify --promote to promote to production.")

if __name__ == "__main__":
    main()
