import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
import yfinance as yf
import json
import logging

from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables (.env)
load_dotenv()

# Add project root and backend to sys.path to resolve module imports
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'backend'))

from backend.src.data.etf_ingestion import ETFDataFetcher
from backend.src.features.portfolio_builder import PortfolioBuilder
from backend.src.features.dataset_builder import DatasetBuilder
from backend.src.features.risk_metrics import RiskFeatureEngineer
from backend.src.models.risk_regressor import RiskRegressor
from backend.src.models.risk_calibration import calibrate_risk_probabilities, get_calibration_metadata
from backend.src.models.llm_agent import MockLLMAgent
from prometheus_fastapi_instrumentator import Instrumentator

logger = logging.getLogger(__name__)

app = FastAPI(
    title="QuantShield Continuous Risk Forecasting API",
    description="Multi-output continuous risk forecasting engine (V1.2: 16-Feature Multi-Horizon Dynamics + EWMA).",
    version="1.2.0"
)

# Enable CORS for all origins (Streamlit Cloud, local, mobile, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instrument Prometheus metrics endpoint (/metrics)
Instrumentator().instrument(app).expose(app)

class Holding(BaseModel):
    ticker: str
    weight: float

class PortfolioRequest(BaseModel):
    etf_name: str
    reporting_date: str
    holdings: List[Holding]
    benchmark: str = "^NSEI"

# Preload V1.2 candidate/production model and metadata at startup
model_path = project_root / 'backend' / 'src' / 'models' / 'saved_model.cbm'
if not model_path.exists():
    model_path = project_root / 'backend' / 'src' / 'models' / 'experiments' / 'family_b_ewma' / 'saved_model.cbm'

metadata_path = project_root / 'backend' / 'src' / 'models' / 'model_metadata.json'
if not metadata_path.exists():
    metadata_path = project_root / 'backend' / 'src' / 'models' / 'experiments' / 'family_b_ewma' / 'model_metadata.json'

sklearn_model = None
model_metadata = {}

try:
    sklearn_model = CatBoostRegressor()
    sklearn_model.load_model(str(model_path))
    logger.info(f"Loaded CatBoost MultiRMSE model from {model_path}")
    print(f"Loaded CatBoost MultiRMSE model from {model_path}")
except Exception as e:
    logger.error(f"Failed to load CatBoost model: {e}")
    print(f"Failed to load CatBoost model from {model_path}: {e}")

if metadata_path.exists():
    try:
        with open(metadata_path, 'r') as f:
            model_metadata = json.load(f)
        print(f"Loaded model metadata from {metadata_path}")
    except Exception as e:
        print(f"Failed to load model metadata: {e}")

FEATURE_NAME_MAP = {
    "Vol": "Annualized_Volatility",
    "VaR95": "Historical_VaR_95",
    "MaxDD": "Maximum_Drawdown",
    "DivRatio": "Diversification_Ratio",
    "Skewness": "Skewness",
    "Kurtosis": "Kurtosis",
    "RollingVol20": "RollingVol20",
    "RollingVol60": "RollingVol60",
    "Sharpe": "Sharpe",
    "Sortino": "Sortino",
    "Beta": "Beta"
}

@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "QuantShield Continuous Risk Forecasting API",
        "model_version": "V1.2 (16-Feature Multi-Horizon Dynamics + EWMA)",
        "docs_url": "/docs",
        "health_check": "/api/v1/risk/health"
    }

@app.get("/api/v1/risk/health")
async def health_check():
    return {
        "status": "ok", 
        "model_loaded": sklearn_model is not None,
        "feature_count": len(sklearn_model.feature_names_) if sklearn_model else 0,
        "features": list(sklearn_model.feature_names_) if sklearn_model else []
    }

@app.post("/api/v1/risk/predict")
def predict_risk(request: PortfolioRequest):
    if not sklearn_model:
        raise HTTPException(status_code=500, detail="RiskRegressor model is not loaded.")
        
    try:
        # 1. Validate Weights 
        total_weight = sum([h.weight for h in request.holdings])
        if not (0.95 <= total_weight <= 1.05):
            raise HTTPException(
                status_code=400, 
                detail=f"Portfolio weights must sum to ~1.0. Provided sum: {total_weight:.4f}"
            )
            
        # 2. Transform request into ingestion format
        holdings_input = {
            "etf_name": request.etf_name,
            "reporting_date": request.reporting_date,
            "holdings": [{"ticker": h.ticker, "weight": h.weight} for h in request.holdings]
        }
        
        # 3. Ingestion pipeline (fetch data with sufficient lookback for lags)
        try:
            fetcher = ETFDataFetcher(years=2)
            output = fetcher.fetch_data(holdings_input)
            price_data = output["price_data"]
            weights = output["weights"]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception:
            fetcher = ETFDataFetcher(years=5)
            output = fetcher.fetch_data(holdings_input)
            price_data = output["price_data"]
            weights = output["weights"]
            
        # 4. Reconstruct daily portfolio returns
        builder = PortfolioBuilder(price_data=price_data)
        portfolio_df = builder.build_portfolio(weights)
        daily_returns = portfolio_df['Daily_Return'].dropna()
        
        min_required = DatasetBuilder.WINDOW_LENGTH # 252 days
        if len(daily_returns) < min_required:
            raise ValueError(
                f"Insufficient trading history: {len(daily_returns)} days available, "
                f"at least {min_required} days required for rolling risk metrics."
            )
            
        # 5. Extract exact 16 features for V1.2 live inference (Family B + EWMA)
        inference_payload = DatasetBuilder.compute_inference_features(daily_returns, feature_set="family_b_ewma")
        features_df = inference_payload["features_df"]
        ewma_vol = inference_payload["ewma_vol"]
        feature_values = inference_payload["feature_values"]
        
        # 6. Multi-Output Continuous Risk Forecast: [Forward_Vol, Forward_MaxDD]
        raw_preds = sklearn_model.predict(features_df)
        if hasattr(raw_preds, 'ndim') and raw_preds.ndim == 2:
            pred_fwd_vol = float(raw_preds[0, 0])
            pred_fwd_maxdd = float(raw_preds[0, 1])
        else:
            pred_fwd_vol = float(raw_preds[0])
            pred_fwd_maxdd = float(raw_preds[1])
            
        # 7. Spread vs Benchmark
        vol_spread_vs_ewma = pred_fwd_vol - ewma_vol
        
        # 8. Risk Regime and Threat Classification
        if pred_fwd_vol < 0.16:
            risk_class = "Low"
            vol_regime = "Subdued Volatility Corridor"
        elif pred_fwd_vol < 0.25:
            risk_class = "Medium"
            vol_regime = "Equilibrium / Range-Bound Volatility"
        else:
            risk_class = "High"
            vol_regime = "Elevated / Stochastic Volatility Surge"
            
        if pred_fwd_maxdd < 0.05:
            dd_threat = "Low Capital Impairment Risk"
        elif pred_fwd_maxdd < 0.10:
            dd_threat = "Moderate Drawdown Exposure"
        else:
            dd_threat = "Severe Tail Risk & Drawdown Exposure"
            
        # Continuous composite synthetic vulnerability score (0.0 to 10.0 scale)
        risk_score = float(np.clip(round((pred_fwd_vol / 0.35) * 6.0 + (pred_fwd_maxdd / 0.12) * 4.0, 2), 0.5, 10.0))
        
        # Option A: Empirical Out-Of-Fold (OOF) Risk Calibration (N = 4,917)
        prob_dict = calibrate_risk_probabilities(pred_fwd_vol)
        
        # 9. Slice recent 252-day window for supplementary metrics and charts
        inference_returns = daily_returns.iloc[-252:]
        start_date = inference_returns.index[0]
        end_date = inference_returns.index[-1]
        component_returns = builder.daily_returns.loc[start_date:end_date]
        
        # Benchmark comparison series
        benchmark_ticker = request.benchmark
        benchmark_cum = [0.0] * len(inference_returns)
        market_returns = None
        try:
            bm_raw = yf.download(
                benchmark_ticker,
                start=start_date,
                end=end_date + pd.Timedelta(days=1),
                progress=False
            )
            if not bm_raw.empty:
                if isinstance(bm_raw.columns, pd.MultiIndex):
                    price_levels = bm_raw.columns.get_level_values(0).unique()
                    if 'Adj Close' in price_levels:
                        bm_prices = bm_raw.xs('Adj Close', axis=1, level=0)
                    else:
                        bm_prices = bm_raw.xs('Close', axis=1, level=0)
                else:
                    if 'Adj Close' in bm_raw.columns:
                        bm_prices = bm_raw[['Adj Close']]
                    else:
                        bm_prices = bm_raw[['Close']]
                
                if isinstance(bm_prices, pd.DataFrame):
                    bm_prices = bm_prices.iloc[:, 0]
                bm_daily = bm_prices.pct_change()
                market_returns = bm_daily.reindex(inference_returns.index).ffill(limit=5)
                aligned_bm_returns = market_returns.fillna(0.0)
                benchmark_cum = ((1 + aligned_bm_returns).cumprod() - 1).tolist()
        except Exception as e:
            logger.warning(f"Error fetching benchmark {benchmark_ticker}: {e}")
            
        # Traditional portfolio metrics for UI tooltips
        engineer = RiskFeatureEngineer(
            portfolio_returns=inference_returns,
            component_returns=component_returns,
            weights=weights,
            market_returns=market_returns
        )
        features_dict = engineer.compute_all_features()
        
        # Portfolio cumulative returns series
        portfolio_cum = ((1 + inference_returns).cumprod() - 1).tolist()
        returns_data = component_returns.to_dict(orient="list")
        
        # LLM Explanation Layer
        agent = MockLLMAgent()
        dashboard_explanations = agent.generate_dashboard_explanations(
            risk_class, features_dict, portfolio_cum[-1], benchmark_cum[-1]
        )
        
        return {
            "model_version": "V1.2 (Family B + EWMA)",
            "forecast_horizon_days": 21,
            "forward_volatility": round(pred_fwd_vol, 4),
            "forward_max_drawdown": round(pred_fwd_maxdd, 4),
            "predictions": {
                "forward_volatility": round(pred_fwd_vol, 4),
                "forward_max_drawdown": round(pred_fwd_maxdd, 4),
                "annualized_vol_pct": round(pred_fwd_vol * 100, 2),
                "max_drawdown_pct": round(pred_fwd_maxdd * 100, 2),
                "risk_score": risk_score,
                "vol_regime": vol_regime,
                "drawdown_threat": dd_threat
            },
            "benchmark": {
                "model": "RiskMetrics EWMA (λ=0.94)",
                "ewma_volatility": round(ewma_vol, 4),
                "vol_spread_vs_ewma": round(vol_spread_vs_ewma, 4),
                "signal": "Volatility Premium" if vol_spread_vs_ewma > 0 else "Volatility Discount"
            },
            "features_15": feature_values,
            "features_16": feature_values,
            "feature_values": feature_values,
            "features_list": list(features_df.columns),
            # Backward-compatible fields
            "risk_class": risk_class,
            "risk_score": risk_score,
            "probabilities": prob_dict,
            "risk_calibration": get_calibration_metadata(),
            "baseline_forecast": {
                "model": "RiskMetrics EWMA (λ=0.94)",
                "forecast_volatility": round(float(ewma_vol), 4),
                "predicted_regime": "Low" if ewma_vol < 0.16 else ("Medium" if ewma_vol < 0.25 else "High")
            },
            "transparency_report": model_metadata,
            "metrics": features_dict,
            "dashboard_explanations": dashboard_explanations,
            "portfolio_returns": portfolio_cum,
            "benchmark_returns": benchmark_cum,
            "benchmark_name": benchmark_ticker,
            "returns_data": returns_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
