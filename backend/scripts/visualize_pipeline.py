"""
QuantShield Pipeline Visualizer
================================
Visually see how data flows through each stage of the ML pipeline.

"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'backend'))

import pandas as pd
import numpy as np
from backend.src.data.etf_ingestion import ETFDataFetcher
from backend.src.features.portfolio_builder import PortfolioBuilder
from backend.src.features.risk_metrics import RiskFeatureEngineer
from backend.src.features.dataset_builder import DatasetBuilder

# Use a single clean portfolio for demo
DEMO_PORTFOLIO = {
    "etf_name": "DEMO_NIFTY_BANK",
    "reporting_date": "2023-10-31",
    "holdings": [
        {"ticker": "HDFCBANK.NS", "weight": 0.40},
        {"ticker": "ICICIBANK.NS", "weight": 0.30},
        {"ticker": "SBIN.NS", "weight": 0.20},
        {"ticker": "AXISBANK.NS", "weight": 0.10}
    ]
}

def print_header(step_num, title):
    print(f"\n{'='*70}")
    print(f"  STEP {step_num}: {title}")
    print(f"{'='*70}\n")

def main():
    # =========================================================================
    # STEP 1: RAW DATA INGESTION
    # =========================================================================
    print_header(1, "DATA INGESTION (ETFDataFetcher)")
    print("  Input: Portfolio definition with tickers and weights")
    print(f"  Tickers: {[h['ticker'] for h in DEMO_PORTFOLIO['holdings']]}")
    print(f"  Weights: {[h['weight'] for h in DEMO_PORTFOLIO['holdings']]}")
    print(f"  Fetching 5 years of daily price data from Yahoo Finance...\n")

    fetcher = ETFDataFetcher(years=5)
    output = fetcher.fetch_data(DEMO_PORTFOLIO)
    price_data = output["price_data"]
    weights = output["weights"]

    print(f"  Output: Price DataFrame")
    print(f"  Shape: {price_data.shape[0]} trading days × {price_data.shape[1]} stocks")
    print(f"  Date Range: {price_data.index[0].date()} to {price_data.index[-1].date()}")
    print(f"\n  First 5 rows of RAW PRICE DATA:")
    print(f"  {'-'*60}")
    print(price_data.head().to_string(float_format="Rs. {:,.2f}".format))

    # =========================================================================
    # STEP 2: PORTFOLIO CONSTRUCTION
    # =========================================================================
    print_header(2, "PORTFOLIO CONSTRUCTION (PortfolioBuilder)")
    print("  Input: Raw prices + Weights")
    print("  Process: Convert prices -> daily returns -> weight them -> sum into one portfolio return\n")

    builder = PortfolioBuilder(price_data=price_data)
    portfolio_df = builder.build_portfolio(weights)

    print(f"  Individual Stock Daily Returns (first 5 days):")
    print(f"  {'-'*60}")
    print(builder.daily_returns.head().to_string(float_format="{:.4%}".format))

    print(f"\n  Combined Portfolio Daily Returns (first 10 days):")
    print(f"  {'-'*60}")
    print(portfolio_df[['Daily_Return']].head(10).to_string(float_format="{:.4%}".format))
    print(f"\n  Total trading days with returns: {len(portfolio_df['Daily_Return'].dropna())}")

    # =========================================================================
    # STEP 3: ROLLING WINDOW SLICING
    # =========================================================================
    print_header(3, "ROLLING WINDOW SLICING (DatasetBuilder Logic)")
    daily_returns = portfolio_df['Daily_Return'].dropna()
    n_days = len(daily_returns)
    window_len = 126
    step_size = 21
    num_windows = (n_days - window_len) // step_size + 1

    print(f"  Total trading days available: {n_days}")
    print(f"  Window length: {window_len} days (~6 months)")
    print(f"  Step size: {step_size} days (~1 month)")
    print(f"  Number of windows generated: {num_windows}")
    print(f"\n  Visual representation of first 5 windows:")
    print(f"  {'-'*60}")

    for i in range(min(5, num_windows)):
        start = i * step_size
        end = start + window_len
        w_start = daily_returns.index[start].date()
        w_end = daily_returns.index[end - 1].date()
        bar = "#" * 30
        gap = " " * (i * 3)
        print(f"  Window {i+1}: {gap}{bar}  [{w_start} -> {w_end}]")

    print(f"  ...")
    last_start = (num_windows - 1) * step_size
    last_end = last_start + window_len
    w_start = daily_returns.index[last_start].date()
    w_end = daily_returns.index[last_end - 1].date()
    print(f"  Window {num_windows}: {' ' * 15}{'#' * 30}  [{w_start} -> {w_end}]")

    # =========================================================================
    # STEP 4: FEATURE ENGINEERING (one sample window)
    # =========================================================================
    print_header(4, "RISK FEATURE ENGINEERING (RiskFeatureEngineer)")
    print(f"  Computing 11 statistical metrics for Window 1...")
    print(f"  Window 1: {daily_returns.index[0].date()} -> {daily_returns.index[125].date()}\n")

    window_returns = daily_returns.iloc[0:126]
    comp_returns = builder.daily_returns.iloc[0:126]

    engineer = RiskFeatureEngineer(
        portfolio_returns=window_returns,
        component_returns=comp_returns,
        weights=weights,
        market_returns=window_returns  # self-benchmark for demo
    )
    features = engineer.compute_all_features()

    print(f"  {'Metric':<30} {'Value':>15}  {'Meaning'}")
    print(f"  {'-'*75}")
    print(f"  {'Annualized Volatility':<30} {features['Annualized_Volatility']:>14.2%}   Annual price swing")
    print(f"  {'Historical VaR (95%)':<30} {features['Historical_VaR_95']:>14.2%}   Worst daily loss (95% conf)")
    print(f"  {'Maximum Drawdown':<30} {features['Maximum_Drawdown']:>14.2%}   Largest peak-to-trough drop")
    print(f"  {'Diversification Ratio':<30} {features['Diversification_Ratio']:>14.2f}    >1 = diversification helps")
    print(f"  {'Skewness':<30} {features['Skewness']:>14.2f}    Negative = more crashes")
    print(f"  {'Kurtosis':<30} {features['Kurtosis']:>14.2f}    High = fat tails")
    print(f"  {'Rolling Vol (20d)':<30} {features['RollingVol20']:>14.2%}   Recent short-term risk")
    print(f"  {'Rolling Vol (60d)':<30} {features['RollingVol60']:>14.2%}   Recent medium-term risk")
    print(f"  {'Sharpe Ratio':<30} {features['Sharpe']:>14.2f}    Return per unit risk")
    print(f"  {'Sortino Ratio':<30} {features['Sortino']:>14.2f}    Return per downside risk")
    print(f"  {'Beta':<30} {features['Beta']:>14.2f}    Market sensitivity")

    # =========================================================================
    # STEP 5: FORWARD RISK TARGET ENGINEERING (Horizon h = 21 Days)
    # =========================================================================
    print_header(5, "FORWARD RISK TARGET ENGINEERING (DatasetBuilder.compute_forward_target)")
    
    future_21d = daily_returns.iloc[126:147]
    fwd_vol, fwd_maxdd, label = DatasetBuilder.compute_forward_target(future_21d)

    print(f"  Future Window [t+1 : t+21]: {future_21d.index[0].date()} -> {future_21d.index[-1].date()}")
    print(f"  Realized Forward Annualized Volatility: {fwd_vol:>14.2%}")
    print(f"  Realized Forward Maximum Drawdown:      {fwd_maxdd:>14.2%}")
    print(f"\n  Forward Regime Rules:")
    print(f"    High   -> Vol >= 22% OR MaxDD >= 6.5%")
    print(f"    Low    -> Vol < 15% AND MaxDD < 4.5%")
    print(f"    Medium -> Otherwise")
    print(f"\n  +---------------------------------------------------------+")
    print(f"  |  FORWARD REALIZED RISK TARGET (Y_t+21): {label:>12}     |")
    print(f"  +---------------------------------------------------------+")

    # =========================================================================
    # STEP 6: SUMMARY
    # =========================================================================
    print_header(6, "FINAL DATASET SAMPLE (One Row in training_dataset.csv)")
    print(f"  Portfolio_ID:  DEMO_NIFTY_BANK")
    print(f"  Input Window:  {daily_returns.index[0].date()} -> {daily_returns.index[125].date()} (X_t)")
    print(f"  Target Window: {future_21d.index[0].date()} -> {future_21d.index[-1].date()} (Y_t+21)")
    print(f"  Vol:           {features['Annualized_Volatility']:.4f}")
    print(f"  VaR95:         {features['Historical_VaR_95']:.4f}")
    print(f"  MaxDD:         {features['Maximum_Drawdown']:.4f}")
    print(f"  DivRatio:      {features['Diversification_Ratio']:.4f}")
    print(f"  Skewness:      {features['Skewness']:.4f}")
    print(f"  Kurtosis:      {features['Kurtosis']:.4f}")
    print(f"  RollingVol20:  {features['RollingVol20']:.4f}")
    print(f"  RollingVol60:  {features['RollingVol60']:.4f}")
    print(f"  Sharpe:        {features['Sharpe']:.4f}")
    print(f"  Sortino:       {features['Sortino']:.4f}")
    print(f"  Beta:          {features['Beta']:.4f}")
    print(f"  Forward_Vol:   {fwd_vol:.4f}")
    print(f"  Forward_MaxDD: {fwd_maxdd:.4f}")
    print(f"  Label:         {label}")
    print(f"\n  -> Fed into CatBoostClassifier with PurgedGroupTimeSeriesSplit (30d embargo)")
    print(f"  -> Benchmarked against RiskMetrics EWMA statistical baseline")
    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE - Zero lookahead leakage.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
