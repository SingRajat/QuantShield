import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import joblib
import numpy as np
import pandas as pd

# Add project root and backend to sys.path to resolve module imports
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'backend'))

from backend.src.data.etf_ingestion import ETFDataFetcher
from backend.src.features.portfolio_builder import PortfolioBuilder
from backend.src.features.risk_metrics import RiskFeatureEngineer
from backend.src.models.llm_agent import MockLLMAgent
from backend.src.models.risk_classifier import RiskClassifier

app = FastAPI(title="QuantShield Risk API")

class Holding(BaseModel):
    ticker: str
    weight: float

class PortfolioRequest(BaseModel):
    etf_name: str
    reporting_date: str
    holdings: List[Holding]
    benchmark: str = "^NSEI"

# Preload model at startup
model_path = project_root / 'backend' / 'src' / 'models' / 'saved_model.pkl'
sklearn_model = None
try:
    sklearn_model = joblib.load(model_path)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Failed to load SKLearn model from {model_path}: {e}")

@app.post("/api/v1/risk/predict")
async def predict_risk(request: PortfolioRequest):
    if not sklearn_model:
        raise HTTPException(status_code=500, detail="RiskClassifier model is not loaded.")
        
    try:
        # 1. Validate Weights 
        total_weight = sum([h.weight for h in request.holdings])
        if not (0.95 <= total_weight <= 1.05):
            raise HTTPException(status_code=400, detail=f"Portfolio weights must sum to ~1.0. Provided sum: {total_weight:.4f}")
            
        # 2. Transform request into ingestion format
        holdings_input = {
            "etf_name": request.etf_name,
            "reporting_date": request.reporting_date,
            "holdings": [{"ticker": h.ticker, "weight": h.weight} for h in request.holdings],
            "benchmark": request.benchmark
        }
        
        # 3. Ingestion pipeline (fetch 5 years of data for consistency with training horizon)
        fetcher = ETFDataFetcher(years=5)
        output = fetcher.fetch_data(holdings_input)
        price_data = output["price_data"]
        weights = output["weights"]
        
        # 3. Portfolio Builder
        builder = PortfolioBuilder(price_data=price_data)
        portfolio_df = builder.build_portfolio(weights)
        
        # 4. Slice inference window (e.g. recent 6 months ~ 126 trading days)
        daily_returns = portfolio_df['Daily_Return'].dropna()
        if len(daily_returns) < 126:
             raise ValueError(f"Insufficient data: {len(daily_returns)} days fetched, required 126.")
             
        inference_returns = daily_returns.iloc[-126:]
        
        # Slice component returns for diversification score
        start_date = inference_returns.index[0]
        end_date = inference_returns.index[-1]
        component_returns = builder.daily_returns.loc[start_date:end_date]
        
        # 5. Statistical computation
        engineer = RiskFeatureEngineer(
            portfolio_returns=inference_returns,
            component_returns=component_returns,
            weights=weights
        )
        
        features_dict = engineer.compute_all_features()
        
        # Map stats exact to ML input signature
        vol = features_dict.get("Annualized_Volatility", np.nan)
        var95 = features_dict.get("Historical_VaR_95", np.nan)
        max_dd = features_dict.get("Maximum_Drawdown", np.nan)
        div_ratio = features_dict.get("Diversification_Ratio", 1.0)
        
        ml_features = pd.DataFrame([{
            "Vol": vol,
            "VaR95": var95,
            "MaxDD": max_dd,
            "DivRatio": div_ratio
        }])
        
        # 6. Inference Layer (use sklearn model directly, respecting ordered FEATURES)
        features_order = ["Vol", "VaR95", "MaxDD", "DivRatio"]
        X_infer = ml_features[features_order]
        prediction = sklearn_model.predict(X_infer)[0]
        
        proba = sklearn_model.predict_proba(X_infer)[0]
        risk_score = float(max(proba)) * 10
        
        # Calculate returns for charts
        portfolio_cum = ((1 + inference_returns).cumprod() - 1).tolist()
        
        benchmark_prices = output["benchmark_data"].reindex(inference_returns.index).ffill().bfill()
        benchmark_daily = benchmark_prices.pct_change().fillna(0)
        benchmark_cum = ((1 + benchmark_daily).cumprod() - 1).iloc[:, 0].tolist()
        
        returns_data = component_returns.to_dict(orient="list")
        
        # 7. LLM Explanation Layer
        agent = MockLLMAgent()
        explanation = agent.generate_explanation(prediction, features_dict)
        
        return {
            "risk_class": prediction,
            "metrics": features_dict,
            "explanation": explanation,
            "portfolio_returns": portfolio_cum,
            "benchmark_returns": benchmark_cum,
            "benchmark_name": output["benchmark_name"],
            "returns_data": returns_data,
            "risk_score": risk_score
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
