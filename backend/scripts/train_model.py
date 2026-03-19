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

# To guarantee 20 years of data without `dropna()` truncating the series to recently listed stocks,
# all mock portfolios strictly use high-liquidity, historic Indian equities listed pre-2005/2010.
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

# Provide reporting dates globally, as the fetcher expects it in the dictionary
for port in MOCK_PORTFOLIOS:
    port["reporting_date"] = "2023-10-31"

def main():
    logger.info("Starting End-to-End Pipeline Validation...")

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

    # 3. Dataset Building (Statistical Feature Engineering & Target Assignment)
    # This implicitly respects WINDOW_LENGTH=126 and STEP_SIZE=21 per your original pipeline
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
