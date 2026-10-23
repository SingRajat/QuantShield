import os
import sys
import math
import re
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Dict, Any, Optional, Tuple
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

# Add project root and backend to sys.path
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

logger = logging.getLogger(__name__)

def _sanitize_floats(obj: Any) -> Any:
    """Recursively replaces NaN and Inf with None for RFC 7159 compliant JSON."""
    if isinstance(obj, (float, np.floating)):
        return None if (math.isnan(obj) or math.isinf(obj)) else float(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize_floats(obj.tolist())
    if isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_floats(v) for v in obj]
    return obj

app = FastAPI(
    title="QuantShield Continuous Risk Forecasting API",
    description="Multi-output continuous risk forecasting engine (V1.2: 16-Feature Multi-Horizon Dynamics + EWMA).",
    version="1.2.0"
)

# Fail-safe CORS: reject wildcard with credentials
raw_origins = os.environ.get("CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501")
allowed_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]
if "*" in allowed_origins:
    raise RuntimeError("Insecure CORS configuration: Wildcard '*' cannot be used with credentials.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-Gateway-Secret"],
)

@app.middleware("http")
async def enforce_gateway_boundary(request: Request, call_next):
    # Only enforce gateway boundary in production mode
    if os.environ.get("ENVIRONMENT") == "production":
        expected_secret = os.environ.get("GATEWAY_SECRET")
        if not expected_secret or request.headers.get("X-Gateway-Secret") != expected_secret:
            return JSONResponse(
                status_code=403, 
                content={"detail": "Direct unauthenticated access blocked. Requests must transit through verified API Gateway."}
            )
    return await call_next(request)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    msg = "; ".join([str(e.get("msg", "")) for e in errors])
    if "weight" in msg.lower():
        return JSONResponse(status_code=400, content={"detail": f"Portfolio weights must sum to ~1.0: {msg}"})
    return JSONResponse(status_code=400, content={"detail": msg})

class Holding(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=20)
    weight: float = Field(..., ge=0.0, le=1.0)

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v):
        v = str(v).strip().upper()
        if not re.match(r'^[A-Z0-9.\-_^]{1,20}$', v):
            raise ValueError(f"Invalid ticker format: {v}")
        return v

class PortfolioRequest(BaseModel):
    etf_name: str = Field(..., max_length=100)
    reporting_date: Optional[str] = None
    holdings: List[Holding] = Field(..., min_length=1, max_length=50)
    benchmark: str = Field(default="^NSEI", max_length=20)

    @field_validator("reporting_date")
    @classmethod
    def validate_reporting_date(cls, v):
        if v is not None and str(v).strip():
            v_str = str(v).strip()
            try:
                dt = datetime.strptime(v_str, "%Y-%m-%d").date()
            except ValueError:
                raise ValueError(f"Invalid reporting_date format: {v_str}. Expected YYYY-MM-DD.")
            today_dt = datetime.today().date()
            if dt > today_dt:
                raise ValueError(f"reporting_date cannot be in the future: {dt} > {today_dt}.")
            return v_str
        return None

    @field_validator("benchmark")
    @classmethod
    def validate_benchmark(cls, v):
        v = str(v).strip().upper()
        if not re.match(r'^[A-Z0-9.\-_^]{1,20}$', v):
            raise ValueError(f"Invalid benchmark ticker: {v}")
        return v

    @model_validator(mode="after")
    def validate_portfolio(self):
        tickers = [h.ticker for h in self.holdings]
        if len(tickers) != len(set(tickers)):
            raise ValueError("Duplicate tickers detected in holdings.")
        total_weight = sum(h.weight for h in self.holdings)
        if not (0.90 <= total_weight <= 1.10):
            raise ValueError(f"Holdings weights must sum to ~1.0, got {total_weight:.4f}")
        for h in self.holdings:
            h.weight = float(h.weight / total_weight)
        return self

# Preload candidate/production model and metadata at startup
model_path = project_root / 'backend' / 'src' / 'models' / 'saved_model.cbm'
if not model_path.exists():
    model_path = project_root / 'backend' / 'src' / 'models' / 'experiments' / 'family_b_ewma' / 'saved_model.cbm'

metadata_path = project_root / 'backend' / 'src' / 'models' / 'model_metadata.json'
if not metadata_path.exists():
    metadata_path = project_root / 'backend' / 'src' / 'models' / 'experiments' / 'family_b_ewma' / 'model_metadata.json'

sklearn_model = None
model_metadata = {}

try:
    _candidate = CatBoostRegressor()
    _candidate.load_model(str(model_path))
    sklearn_model = _candidate
    logger.info(f"Loaded CatBoost MultiRMSE model from {model_path}")
except Exception as e:
    sklearn_model = None
    logger.error(f"Failed to load CatBoost model from {model_path}: {e}")

if metadata_path.exists():
    try:
        with open(metadata_path, 'r') as f:
            model_metadata = json.load(f)
        logger.info(f"Loaded model metadata from {metadata_path}")
    except Exception as e:
        logger.error(f"Failed to load model metadata: {e}")

# Retained for backward compatibility with test_api.py import contract
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

def _fetch_and_align_benchmark(
    benchmark_ticker: str, 
    start_date: pd.Timestamp, 
    end_date: pd.Timestamp, 
    inference_index: pd.Index
) -> Tuple[Optional[pd.Series], List[float], List[float]]:
    """
    Fetches benchmark prices and derives:
    1. market_returns: daily returns with NaNs preserved for RiskFeatureEngineer Beta calculation.
    2. benchmark_cum: cumulative returns filled with 0.0 for UI charting.
    3. benchmark_daily: daily return list filled with 0.0 for JSON delivery.
    """
    benchmark_cum = [0.0] * len(inference_index)
    benchmark_daily = [0.0] * len(inference_index)
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
                levels = bm_raw.columns.get_level_values(0).unique()
                bm_prices = bm_raw.xs('Adj Close', axis=1, level=0) if 'Adj Close' in levels else bm_raw.xs('Close', axis=1, level=0)
            else:
                bm_prices = bm_raw[['Adj Close']] if 'Adj Close' in bm_raw.columns else bm_raw[['Close']]
            
            if isinstance(bm_prices, pd.DataFrame):
                bm_prices = bm_prices.iloc[:, 0]
            bm_daily = bm_prices.pct_change()
            market_returns = bm_daily.reindex(inference_index).ffill(limit=5)
            aligned_bm_returns = market_returns.fillna(0.0)
            benchmark_cum = ((1 + aligned_bm_returns).cumprod() - 1).tolist()
            benchmark_daily = aligned_bm_returns.tolist()
    except Exception as e:
        logger.warning(f"Error fetching benchmark {benchmark_ticker}: {e}")
    return market_returns, benchmark_cum, benchmark_daily

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
    if sklearn_model is None:
        return {"status": "degraded", "model_loaded": False, "feature_count": 0, "features": []}
    return {
        "status": "ok", 
        "model_loaded": True,
        "feature_count": len(sklearn_model.feature_names_),
        "features": list(sklearn_model.feature_names_)
    }

@app.post("/api/v1/risk/predict")
def predict_risk(request: PortfolioRequest):
    if sklearn_model is None:
        raise HTTPException(
            status_code=503, 
            detail="RiskRegressor model is not loaded or service is degraded.",
            headers={"Retry-After": "30"}
        )
        
    try:
        # Transform request into ingestion format (weights already validated & normalized by Pydantic)
        holdings_input = {
            "etf_name": request.etf_name,
            "reporting_date": request.reporting_date,
            "holdings": [{"ticker": h.ticker, "weight": h.weight} for h in request.holdings]
        }
        
        # Ingestion pipeline: 2-year lookback with 5-year fallback for non-ValueError issues
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
            
        # Reconstruct daily portfolio returns
        builder = PortfolioBuilder(price_data=price_data)
        portfolio_df = builder.build_portfolio(weights)
        daily_returns = portfolio_df['Daily_Return'].dropna()
        
        min_required = DatasetBuilder.WINDOW_LENGTH # 252 days
        if len(daily_returns) < min_required:
            raise ValueError(
                f"Insufficient trading history: {len(daily_returns)} days available, "
                f"at least {min_required} days required for rolling risk metrics."
            )
            
        # Extract exact 16 features for V1.2 live inference (Family B + EWMA)
        inference_payload = DatasetBuilder.compute_inference_features(daily_returns, feature_set="family_b_ewma")
        features_df = inference_payload["features_df"]
        ewma_vol = inference_payload["ewma_vol"]
        feature_values = inference_payload["feature_values"]
        
        # Multi-Output Continuous Risk Forecast: [Forward_Vol, Forward_MaxDD]
        raw_preds = sklearn_model.predict(features_df)
        if hasattr(raw_preds, 'ndim') and raw_preds.ndim == 2:
            pred_fwd_vol = float(raw_preds[0, 0])
            pred_fwd_maxdd = float(raw_preds[0, 1])
        else:
            pred_fwd_vol = float(raw_preds[0])
            pred_fwd_maxdd = float(raw_preds[1])
            
        # Spread vs Benchmark
        vol_spread_vs_ewma = pred_fwd_vol - ewma_vol
        
        # Risk Regime and Threat Classification
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
        
        # Empirical Out-Of-Fold (OOF) Risk Calibration (N = 4,917)
        prob_dict = calibrate_risk_probabilities(pred_fwd_vol)
        
        # Slice recent 252-day window for supplementary metrics and charts
        inference_returns = daily_returns.iloc[-252:]
        start_date = inference_returns.index[0]
        end_date = inference_returns.index[-1]
        component_returns = builder.daily_returns.loc[start_date:end_date]
        
        # Benchmark comparison series (strictly maintaining separate market and zero-filled series)
        benchmark_ticker = request.benchmark
        market_returns, benchmark_cum, benchmark_daily = _fetch_and_align_benchmark(
            benchmark_ticker, start_date, end_date, inference_returns.index
        )
            
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
        
        return _sanitize_floats({
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
            "benchmark_daily_returns": benchmark_daily,
            "benchmark_name": benchmark_ticker,
            "returns_data": returns_data,
            "artifact_id": model_metadata.get("artifact_id", "v1.2-prod")
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
