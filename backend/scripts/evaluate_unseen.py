import os
import sys
import argparse
import joblib
import logging
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score

# Add project root and backend to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'backend'))

from backend.src.data.etf_ingestion import ETFDataFetcher
from backend.src.features.portfolio_builder import PortfolioBuilder
from backend.src.features.dataset_builder import DatasetBuilder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Very different from the top Large/Midcap Indian ETFs the model was trained on
UNSEEN_PORTFOLIOS = [
    {
        "etf_name": "UNSEEN_SMALLCAP_ETF",
        "reporting_date": "2023-10-31",
        "holdings": [
            {"ticker": "SUZLON.NS", "weight": 0.20},
            {"ticker": "IRFC.NS", "weight": 0.20},
            {"ticker": "RVNL.NS", "weight": 0.20},
            {"ticker": "BSE.NS", "weight": 0.20},
            {"ticker": "CDSL.NS", "weight": 0.20}
        ]
    },
    {
        "etf_name": "UNSEEN_TECH_DEFENSIVE_ETF",
        "reporting_date": "2023-10-31",
        "holdings": [
            {"ticker": "KPITTECH.NS", "weight": 0.40},
            {"ticker": "TATAELXSI.NS", "weight": 0.30},
            {"ticker": "PERSISTENT.NS", "weight": 0.30}
        ]
    },
    {
        "etf_name": "UNSEEN_PSU_BANK_ETF",
        "reporting_date": "2023-10-31",
        "holdings": [
            {"ticker": "PNB.NS", "weight": 0.30},
            {"ticker": "BOB.NS", "weight": 0.30},
            {"ticker": "CANBK.NS", "weight": 0.20},
            {"ticker": "UNIONBANK.NS", "weight": 0.20}
        ]
    }
]

def main():
    logger.info("Starting Evaluation on Unseen Test Dataset...")

    # 1. Load the Pre-Trained Model
    model_path = project_root / 'backend' / 'src' / 'models' / 'saved_model.pkl'
    if not model_path.exists():
        logger.error(f"Cannot find the trained model at: {model_path}")
        return
        
    classifier_model = joblib.load(model_path)
    logger.info(f"Loaded trained SKLearn model from: {model_path}")

    # 2. Ingestion Phase for Unseen Data
    fetcher = ETFDataFetcher(years=5)
    
    portfolios = {}
    component_returns_dict = {}
    weights_dict = {}
    
    for port_def in UNSEEN_PORTFOLIOS:
        etf_name = port_def["etf_name"]
        logger.info(f"Fetching data for UNSEEN portfolio: {etf_name}")
        
        try:
            output = fetcher.fetch_data(port_def)
            price_data = output["price_data"]
            weights = output["weights"]
            
            # Portfolio Building Phase
            builder = PortfolioBuilder(price_data=price_data)
            portfolio_df = builder.build_portfolio(weights)
            
            portfolios[etf_name] = portfolio_df
            component_returns_dict[etf_name] = builder.daily_returns
            weights_dict[etf_name] = weights
            
        except Exception as e:
            logger.error(f"Failed to process portfolio {etf_name}: {e}")

    if not portfolios:
        logger.error("No valid unseen portfolios were processed. Exiting.")
        return

    # 3. Dataset Building (Using exactly the same rules as training)
    logger.info("Initializing DatasetBuilder for unseen data...")
    dataset_builder = DatasetBuilder(
        portfolios=portfolios,
        component_returns_dict=component_returns_dict,
        weights_dict=weights_dict
    )
    
    unseen_df = dataset_builder.build_panel_dataset()
    logger.info(f"Successfully built Unseen Panel Dataset. Shape: {unseen_df.shape}")
    logger.info(f"Actual distribution (calculated by rules):\n{unseen_df['Label'].value_counts()}")

    # 4. Model Evaluation on Unseen Data
    features = [
        "Vol",
        "VaR95",
        "MaxDD",
        "DivRatio",
        "Skewness",
        "Kurtosis",
        "RollingVol20",
        "RollingVol60",
        "Sharpe",
        "Sortino",
        "Beta"
    ]
    X_unseen = unseen_df[features]
    y_actual = unseen_df['Label']
    
    # Predict without seeing the actual labels!
    y_pred = classifier_model.predict(X_unseen)
    
    accuracy = accuracy_score(y_actual, y_pred)
    report = classification_report(y_actual, y_pred)
    
    logger.info(f"\n=======================================================\n")
    logger.info(f"GENERALIZATION ACCURACY ON UNSEEN ETFs: {accuracy:.4f}")
    logger.info(f"\nClassification Report for Unseen Data:\n{report}")
    logger.info(f"=======================================================\n")

    # Optional: Save Unseen Dataset
    unseen_path = project_root / 'backend' / 'src' / 'models' / 'unseen_evaluation_dataset.csv'
    unseen_df['Predicted_Label'] = y_pred
    unseen_df.to_csv(unseen_path, index=False)
    logger.info(f"Saved unseen evaluation predictions to: {unseen_path}")


if __name__ == "__main__":
    main()
