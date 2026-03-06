import os
import sys
import argparse
import joblib
import logging
from pathlib import Path

# Add project root and backend to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'backend'))

from backend.src.data.etf_ingestion import ETFDataFetcher
from backend.src.features.portfolio_builder import PortfolioBuilder
from backend.src.features.dataset_builder import DatasetBuilder
from backend.src.models.risk_classifier import RiskClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MOCK_PORTFOLIOS = [
    {
        "etf_name": "NIFTY_IT_ETF",
        "reporting_date": "2023-10-31",
        "holdings": [
            {"ticker": "TCS.NS", "weight": 0.28},
            {"ticker": "INFY.NS", "weight": 0.28},
            {"ticker": "HCLTECH.NS", "weight": 0.12},
            {"ticker": "WIPRO.NS", "weight": 0.12},
            {"ticker": "TECHM.NS", "weight": 0.10},
            {"ticker": "LTIM.NS", "weight": 0.10}
        ]
    },
    {
        "etf_name": "NIFTY_BANK_ETF",
        "reporting_date": "2023-10-31",
        "holdings": [
            {"ticker": "HDFCBANK.NS", "weight": 0.35},
            {"ticker": "ICICIBANK.NS", "weight": 0.25},
            {"ticker": "SBIN.NS", "weight": 0.15},
            {"ticker": "KOTAKBANK.NS", "weight": 0.15},
            {"ticker": "AXISBANK.NS", "weight": 0.10}
        ]
    },
    {
        "etf_name": "NIFTY_AUTO_ETF",
        "reporting_date": "2023-10-31",
        "holdings": [
            {"ticker": "TATAMOTORS.NS", "weight": 0.25},
            {"ticker": "M&M.NS", "weight": 0.25},
            {"ticker": "MARUTI.NS", "weight": 0.20},
            {"ticker": "BAJAJ-AUTO.NS", "weight": 0.15},
            {"ticker": "EICHERMOT.NS", "weight": 0.15}
        ]
    },
    {
        "etf_name": "NIFTY_PHARMA_ETF",
        "reporting_date": "2023-10-31",
        "holdings": [
            {"ticker": "SUNPHARMA.NS", "weight": 0.25},
            {"ticker": "CIPLA.NS", "weight": 0.20},
            {"ᚑticker": "DRREDDY.NS", "weight": 0.20},
            {"ticker": "DIVISLAB.NS", "weight": 0.20},
            {"ticker": "TORNTPHARM.NS", "weight": 0.15}
        ]
    },
    {
        "etf_name": "NIFTY_FMCG_ETF",
        "reporting_date": "2023-10-31",
        "holdings": [
            {"ticker": "ITC.NS", "weight": 0.35},
            {"ticker": "HUL.NS", "weight": 0.25},
            {"ticker": "BRITANNIA.NS", "weight": 0.15},
            {"ticker": "NESTLEIND.NS", "weight": 0.15},
            {"ticker": "TATACONSUM.NS", "weight": 0.10}
        ]
    },
    {
        "etf_name": "NIFTY_METAL_ETF",
        "reporting_date": "2023-10-31",
        "holdings": [
            {"ticker": "TATASTEEL.NS", "weight": 0.25},
            {"ticker": "HINDALCO.NS", "weight": 0.25},
            {"ticker": "JSWSTEEL.NS", "weight": 0.20},
            {"ticker": "ADANIENT.NS", "weight": 0.15},
            {"ticker": "COALINDIA.NS", "weight": 0.15}
        ]
    },
    {
        "etf_name": "NIFTY_ENERGY_ETF",
        "reporting_date": "2023-10-31",
        "holdings": [
            {"ticker": "RELIANCE.NS", "weight": 0.35},
            {"ticker": "NTPC.NS", "weight": 0.20},
            {"ticker": "ONGC.NS", "weight": 0.15},
            {"ticker": "POWERGRID.NS", "weight": 0.15},
            {"ticker": "IOC.NS", "weight": 0.15}
        ]
    },
    {
        "etf_name": "CONSERVATIVE_DEBT_PROXY_ETF",
        "reporting_date": "2023-10-31",
        "holdings": [
            {"ticker": "LIQUIDBEES.NS", "weight": 0.60},
            {"ticker": "HDFCBANK.NS", "weight": 0.15},
            {"ticker": "ITC.NS", "weight": 0.15},
            {"ticker": "INFY.NS", "weight": 0.10}
        ]
    },
    {
        "etf_name": "AGGRESSIVE_GROWTH_ETF",
        "reporting_date": "2023-10-31",
        "holdings": [
            {"ticker": "ZOMATO.NS", "weight": 0.20},
            {"ticker": "PAYTM.NS", "weight": 0.20},
            {"ticker": "POLICYBZR.NS", "weight": 0.20},
            {"ticker": "NYKAA.NS", "weight": 0.20},
            {"ticker": "DELHIVERY.NS", "weight": 0.20}
        ]
    },
    {
        "etf_name": "HIGH_DIVIDEND_YIELD_ETF",
        "reporting_date": "2023-10-31",
        "holdings": [
            {"ticker": "COALINDIA.NS", "weight": 0.25},
            {"ticker": "ONGC.NS", "weight": 0.25},
            {"ticker": "PFC.NS", "weight": 0.20},
            {"ticker": "RECLTD.NS", "weight": 0.20},
            {"ticker": "ITC.NS", "weight": 0.10}
        ]
    },
    {
        "etf_name": "NIFTY_FINANCIAL_SERVICES_ETF",
        "reporting_date": "2023-10-31",
        "holdings": [
            {"ticker": "BAJFINANCE.NS", "weight": 0.25},
            {"ticker": "CHOLAFIN.NS", "weight": 0.20},
            {"ticker": "MUTHOOTFIN.NS", "weight": 0.20},
            {"ticker": "BAJAJFINSV.NS", "weight": 0.20},
            {"ticker": "SRF.NS", "weight": 0.15}
        ]
    },
    {
        "etf_name": "REALTY_INFRA_ETF",
        "reporting_date": "2023-10-31",
        "holdings": [
            {"ticker": "DLF.NS", "weight": 0.30},
            {"ticker": "L&T.NS", "weight": 0.30},
            {"ticker": "GODREJPROP.NS", "weight": 0.15},
            {"ticker": "MACROTECH.NS", "weight": 0.15},
            {"ticker": "OBEROIRLTY.NS", "weight": 0.10}
        ]
    },
    {
        "etf_name": "MID_CAP_BLEND_ETF",
        "reporting_date": "2023-10-31",
        "holdings": [
            {"ticker": "TRENT.NS", "weight": 0.20},
            {"ticker": "TVSMOTOR.NS", "weight": 0.20},
            {"ticker": "CGPOWER.NS", "weight": 0.20},
            {"ticker": "CUMMINSIND.NS", "weight": 0.20},
            {"ticker": "POLYCAB.NS", "weight": 0.20}
        ]
    },
    {
        "etf_name": "LARGECAP_50_MOCK_ETF",
        "reporting_date": "2023-10-31",
        "holdings": [
            {"ticker": "HDFCBANK.NS", "weight": 0.15},
            {"ticker": "RELIANCE.NS", "weight": 0.15},
            {"ticker": "ICICIBANK.NS", "weight": 0.10},
            {"ticker": "INFY.NS", "weight": 0.10},
            {"ticker": "TCS.NS", "weight": 0.10},
            {"ticker": "ITC.NS", "weight": 0.10},
            {"ticker": "L&T.NS", "weight": 0.10},
            {"ticker": "SBIN.NS", "weight": 0.10},
            {"ticker": "BHARTIARTL.NS", "weight": 0.10}
        ]
    },
    {
        "etf_name": "TATA_CONGLOMERATE_ETF",
        "reporting_date": "2023-10-31",
        "holdings": [
            {"ticker": "TCS.NS", "weight": 0.25},
            {"ticker": "TATAMOTORS.NS", "weight": 0.20},
            {"ticker": "TATASTEEL.NS", "weight": 0.20},
            {"ticker": "TITAN.NS", "weight": 0.20},
            {"ticker": "TATAPOWER.NS", "weight": 0.15}
        ]
    }
]

def main():
    logger.info("Starting End-to-End Pipeline Validation...")

    # 1. Ingestion Phase
    fetcher = ETFDataFetcher(years=5)
    
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

    # 3. Dataset Building (Statistical Feature Engineering & Target Assignment)
    logger.info("Initializing DatasetBuilder...")
    dataset_builder = DatasetBuilder(
        portfolios=portfolios,
        component_returns_dict=component_returns_dict,
        weights_dict=weights_dict
    )
    
    panel_df = dataset_builder.build_panel_dataset()
    logger.info(f"Successfully built panel dataset. Shape: {panel_df.shape}")
    logger.info(f"Class distribution:\n{panel_df['Label'].value_counts()}")

    # Save the generated dataset to a CSV file
    dataset_path = project_root / 'backend' / 'src' / 'models' / 'training_dataset.csv'
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    panel_df.to_csv(dataset_path, index=False)
    logger.info(f"Saved generated training dataset to: {dataset_path}")

    # 4. Training Model (ML Classifier)
    logger.info("Initializing RiskClassifier...")
    classifier = RiskClassifier()
    
    eval_results = classifier.train_and_evaluate(panel_dataset=panel_df, n_splits=5)
    
    logger.info(f"Training completed successfully. Avg Accuracy: {eval_results['average_accuracy']:.2f}")

    # 5. Serialization Integration
    model_dir = project_root / 'backend' / 'src' / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / 'saved_model.pkl'
    joblib.dump(classifier.model, model_path)
    logger.info(f"Serialized trained SKLearn model to: {model_path}")
    
    logger.info("End-to-End Pipeline Validation was completed successfully.")

if __name__ == "__main__":
    main()
