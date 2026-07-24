# System Integration & API Layer Implementation Plan

We will proceed with Phase 2 of the Risk Monitoring System. This plan adheres strictly to all architectural and business constraints (identical pipelines, no forecasting, no unapproved metrics).

## 1. End-to-End Pipeline Validation
**Goal**: Validate the training pipeline and serialize the final [RiskClassifier](file:///c:/Users/rajat/OneDrive/Desktop/QuantShield/backend/src/models/risk_classifier.py#11-137) model.

### [NEW] `backend/scripts/train_model.py`
- We will create a script that strings together the existing layers: `Ingestion` -> [PortfolioBuilder](file:///c:/Users/rajat/OneDrive/Desktop/QuantShield/backend/src/features/portfolio_builder.py#8-74) -> [DatasetBuilder](file:///c:/Users/rajat/OneDrive/Desktop/QuantShield/backend/src/features/dataset_builder.py#9-121) -> [RiskClassifier](file:///c:/Users/rajat/OneDrive/Desktop/QuantShield/backend/src/models/risk_classifier.py#11-137).
- It will define 3 mock target portfolios (e.g., IT, Bank, Auto sectors) using predefined weights of Indian ETFs (`.NS`).
- It will execute the full training loop with `TimeSeriesSplit` and save the trained model using `joblib` to `backend/src/models/saved_model.pkl`.

## 2. API Integration (FastAPI)
**Goal**: Expose the identical inference pipeline via a REST API.

### [NEW] `backend/src/api/main.py`
- Setup a FastAPI application.
- Define a `POST /api/v1/risk/predict` endpoint.
- **Inference Pipeline**: 
  1. Receive `holdings` (tickers and weights).
  2. Call [ETFDataFetcher](file:///c:/Users/rajat/OneDrive/Desktop/QuantShield/backend/src/data/etf_ingestion.py#11-132) to get the latest slice of data.
  3. Call [PortfolioBuilder](file:///c:/Users/rajat/OneDrive/Desktop/QuantShield/backend/src/features/portfolio_builder.py#8-74) to reconstruct returns.
  4. Call `RiskFeatureEngineer` to compute the EXACT 4 approved features (`Annualized_Volatility`, `Historical_VaR_95`, `Maximum_Drawdown`, `Diversification_Ratio`).
  5. Load [RiskClassifier](file:///c:/Users/rajat/OneDrive/Desktop/QuantShield/backend/src/models/risk_classifier.py#11-137) and predict the Risk Class (`Low`, `Medium`, `High`).

## 3. Mocked LLM Explanation Agent
**Goal**: Provide a natural language explanation of the computed risk metrics.

### [NEW] `backend/src/models/llm_agent.py`
- Build a `MockLLMAgent` class (capable of being swapped for OpenAI/Anthropic later).
- Accepts the 4 financial metrics and Risk Class.
- Returns a deterministic textual explanation (No forecasting, purely backward-looking descriptive analysis).

## 4. Streamlit Frontend Integration
**Goal**: Provide an interactive UI for the Risk Monitoring System.

### [NEW] `frontend/app.py`
- A Streamlit app allowing users to input portfolio components and weights.
- Communicates with the FastAPI backend.
- Displays risk classification, individual metric cards, and the LLM explanation.

## Verification Plan

### Automated Tests
- Run `python -m backend.scripts.train_model` to verify data flow, no leakage, and model serialization.
- Start FastAPI with `uvicorn src.api.main:app --reload` and send a test POST request using Python's `requests` or `curl`.

### Manual Verification
- We will start the Streamlit frontend with `streamlit run frontend/app.py`.
- We will manually input a test portfolio (e.g., 50% `TCS.NS`, 50% `INFY.NS`) and verify the UI successfully displays the final Risk Classification and explanations without errors.
