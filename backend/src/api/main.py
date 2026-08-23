import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
import yfinance as yf

from dotenv import load_dotenv

# Load environment variables (.env)
load_dotenv()

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
model_path = project_root / 'backend' / 'src' / 'models' / 'saved_model.cbm'
sklearn_model = None
try:
    sklearn_model = CatBoostClassifier()
    sklearn_model.load_model(str(model_path))
    print(f"CatBoost model loaded successfully from {model_path}")
except Exception as e:
    print(f"Failed to load CatBoost model from {model_path}: {e}")

@app.get("/api/v1/risk/health")
async def health_check():
    return {"status": "ok", "model_loaded": sklearn_model is not None}

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
            "holdings": [{"ticker": h.ticker, "weight": h.weight} for h in request.holdings]
        }
        
        # 3. Ingestion pipeline (fetch 5 years of data for consistency with training horizon)
        try:
            fetcher = ETFDataFetcher(years=5)
            output = fetcher.fetch_data(holdings_input)
            price_data = output["price_data"]
            weights = output["weights"]
        except ValueError as e:
            # specifically catch ValueError from ingestion (like missing tickers)
            raise HTTPException(status_code=400, detail=str(e))
        
        # 3. Portfolio Builder
        builder = PortfolioBuilder(price_data=price_data)
        portfolio_df = builder.build_portfolio(weights)
        
        # 4. Slice inference window (e.g. recent 1 year ~ 252 trading days)
        daily_returns = portfolio_df['Daily_Return'].dropna()
        if len(daily_returns) < 252:
             raise ValueError(f"Insufficient data: {len(daily_returns)} days fetched, required 252.")
             
        inference_returns = daily_returns.iloc[-252:]
        
        # Slice component returns for diversification score
        start_date = inference_returns.index[0]
        end_date = inference_returns.index[-1]
        component_returns = builder.daily_returns.loc[start_date:end_date]
        
        # Fetch Benchmark data for Beta calculation and comparison chart
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
                
                bm_prices = bm_prices.reindex(inference_returns.index).ffill().bfill()
                bm_daily = bm_prices.pct_change().fillna(0)
                market_returns = bm_daily.iloc[:, 0]
                benchmark_cum = ((1 + bm_daily).cumprod() - 1).iloc[:, 0].tolist()
        except Exception as e:
            print(f"Error fetching benchmark: {e}")
            pass
        
        # 5. Statistical computation
        engineer = RiskFeatureEngineer(
            portfolio_returns=inference_returns,
            component_returns=component_returns,
            weights=weights,
            market_returns=market_returns
        )
        
        features_dict = engineer.compute_all_features()
        
        # Map stats exact to ML input signature
        vol = features_dict.get("Annualized_Volatility", np.nan)
        var95 = features_dict.get("Historical_VaR_95", np.nan)
        max_dd = features_dict.get("Maximum_Drawdown", np.nan)
        div_ratio = features_dict.get("Diversification_Ratio", 1.0)
        
        skewness = features_dict.get("Skewness", np.nan)
        kurtosis = features_dict.get("Kurtosis", np.nan)
        rolling_vol_20 = features_dict.get("RollingVol20", np.nan)
        rolling_vol_60 = features_dict.get("RollingVol60", np.nan)
        sharpe = features_dict.get("Sharpe", np.nan)
        sortino = features_dict.get("Sortino", np.nan)
        beta = features_dict.get("Beta", np.nan)
        
        ml_features = pd.DataFrame([{
            "Vol": vol,
            "VaR95": var95,
            "MaxDD": max_dd,
            "DivRatio": div_ratio,
            "Skewness": skewness,
            "Kurtosis": kurtosis,
            "RollingVol20": rolling_vol_20,
            "RollingVol60": rolling_vol_60,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Beta": beta
        }])
        
        # 6. Inference Layer (use sklearn model directly, respecting ordered FEATURES)
        features_order = [
            "Vol", "VaR95", "MaxDD", "DivRatio", 
            "Skewness", "Kurtosis", "RollingVol20", "RollingVol60", 
            "Sharpe", "Sortino", "Beta"
        ]
        
        X_infer = ml_features[features_order]
        prediction = str(sklearn_model.predict(X_infer).ravel()[0])
        
        # 7. Risk Score — composite score used to position within CatBoost's tier band
        vol = float(features_dict.get("Annualized_Volatility", 0))
        var95 = float(features_dict.get("Historical_VaR_95", 0))
        max_dd = float(features_dict.get("Maximum_Drawdown", 0))
        div_ratio = float(features_dict.get("Diversification_Ratio", 1.0))
        skewness = float(features_dict.get("Skewness", 0) or 0)
        kurtosis_val = float(features_dict.get("Kurtosis", 0) or 0)
        beta_val = float(features_dict.get("Beta", 1.0) or 1.0)
        sortino_val = float(features_dict.get("Sortino", 0) or 0)
        
        norm_vol = min(vol / 0.25, 1.0)
        norm_var = min(var95 / 0.05, 1.0)
        norm_dd = min(max_dd / 0.30, 1.0)
        norm_div_penalty = 1.0 - min(max(div_ratio - 1.0, 0), 1.0)
        norm_skew_penalty = min(abs(min(skewness, 0)) / 2.0, 1.0)
        norm_kurt_penalty = min(max(kurtosis_val, 0) / 5.0, 1.0)
        norm_beta_penalty = min(max(beta_val - 1.0, 0) / 0.5, 1.0)
        norm_sortino_penalty = 1.0 - min(max(sortino_val, 0) / 2.0, 1.0)
        
        core_score = (0.25 * norm_vol) + (0.35 * norm_var) + (0.15 * norm_dd)
        tail_score = (0.10 * norm_skew_penalty) + (0.05 * norm_kurt_penalty) + (0.05 * norm_beta_penalty) + (0.05 * norm_sortino_penalty)
        composite = min(1.0, (core_score + tail_score) * (1.0 + (0.15 * norm_div_penalty)))
        
        # Map composite (0-1) into the CatBoost tier's band for gauge alignment
        if prediction == "Low":
            risk_score = round(0.0 + 3.33 * composite, 2)       # 0.00 – 3.33
        elif prediction == "Medium":
            risk_score = round(3.34 + 3.32 * composite, 2)      # 3.34 – 6.66
        else:
            risk_score = round(6.67 + 3.33 * composite, 2)      # 6.67 – 10.0
        risk_score = min(10.0, risk_score)
        
        # 8. Chart data — Portfolio cumulative returns
        portfolio_cum = ((1 + inference_returns).cumprod() - 1).tolist()
        
        # 9. Benchmark data was fetched earlier in Step 4.5
        
        # 10. Component returns for correlation heatmap
        returns_data = component_returns.to_dict(orient="list")
        
        # 11. LLM Explanation Layer
        agent = MockLLMAgent()
        dashboard_explanations = agent.generate_dashboard_explanations(
            prediction, features_dict, portfolio_cum[-1], benchmark_cum[-1]
        )
        
        return {
            "risk_class": prediction,
            "metrics": features_dict,
            "dashboard_explanations": dashboard_explanations,
            "portfolio_returns": portfolio_cum,
            "benchmark_returns": benchmark_cum,
            "benchmark_name": benchmark_ticker,
            "returns_data": returns_data,
            "risk_score": risk_score
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
