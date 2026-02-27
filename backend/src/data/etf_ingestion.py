import yfinance as yf
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ETFDataFetcher:
    """
    Fetches historical price data for the UNDERLYING stocks of an ETF portfolio.
    Follows a Holdings-Based Portfolio Reconstruction approach.
    """
    
    def __init__(self, years: int = 5):
        self.years = years
        
    def fetch_data(self, holdings_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accepts ETF holdings structure, extracts underlying tickers,
        and fetches 'Adj Close' prices for the last `self.years` years.
        
        Args:
            holdings_input (Dict): Expected format:
            {
                "etf_name": "...",
                "reporting_date": "...",
                "holdings": [
                    {"ticker": "...", "weight": ...},
                    ...
                ]
            }
            
        Returns:
            Dict: Output format:
            {
                "etf_name": str,
                "weights": Dict[str, float],
                "price_data": pd.DataFrame
            }
        """
        if "etf_name" not in holdings_input or "holdings" not in holdings_input:
            raise ValueError("Input must contain 'etf_name' and 'holdings' keys.")
            
        etf_name = holdings_input["etf_name"]
        holdings_list = holdings_input["holdings"]
        
        if not holdings_list:
            raise ValueError("The 'holdings' list cannot be empty.")
            
        # Extract weights and list of tickers to fetch
        weights = {}
        tickers_to_fetch = []
        
        for item in holdings_list:
            if "ticker" not in item or "weight" not in item:
                raise ValueError("Each holding must contain a 'ticker' and 'weight'.")
            ticker = item["ticker"]
            weight = float(item["weight"])
            
            weights[ticker] = weight
            tickers_to_fetch.append(ticker)
            
        # Verify weights approximately sum to 1.0 (or 100 if percentages, assuming normalized to 1)
        total_weight = sum(weights.values())
        if not (0.95 <= total_weight <= 1.05):
            logger.warning(f"Total weights sum to {total_weight}. Ensure weights are normalized to 1.0.")

        # Fetch Data
        end_date = datetime.today()
        start_date = end_date - relativedelta(years=self.years)
        
        logger.info(f"Fetching data for {len(tickers_to_fetch)} underlying stocks from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        try:
            # yfinance returns a MultiIndex DataFrame if multiple tickers are provided
            data = yf.download(tickers_to_fetch, start=start_date, end=end_date, progress=False)
            
            # Error checking: yfinance might download empty data if no tickers are valid
            if data.empty:
                raise ValueError("Downloaded data is completely empty. Please verify tickers.")
            
            # Extract 'Adj Close'
            if isinstance(data.columns, pd.MultiIndex):
                # yfinance >= 0.2.x returns MultiIndex
                price_level = data.columns.get_level_values(0).unique()
                if 'Adj Close' in price_level:
                    df = data.xs('Adj Close', axis=1, level=0)
                elif 'Close' in price_level:
                    df = data.xs('Close', axis=1, level=0)
                    logger.warning("Falling back to 'Close'. 'Adj Close' not found.")
                else:
                    raise ValueError("Neither 'Adj Close' nor 'Close' found in downloaded data.")
            else:
                # Fallback for single ticker
                if 'Adj Close' in data.columns:
                    df = data['Adj Close'].to_frame(name=tickers_to_fetch[0])
                elif 'Close' in data.columns:
                    df = data['Close'].to_frame(name=tickers_to_fetch[0])
                    logger.warning("Falling back to 'Close'. 'Adj Close' not found.")
                else:
                    raise ValueError("Neither 'Adj Close' nor 'Close' found in downloaded data.")
            
            # Validate all requested tickers returned columns
            missing_tickers = [t for t in tickers_to_fetch if t not in df.columns]
            if missing_tickers:
                 raise ValueError(f"Failed to fetch data for these expected tickers: {missing_tickers}")

            # Data Integrity: Forward fill holidays/weekends, backward fill missing starting prices
            df = df.ffill().bfill()
            
            # Second pass: check if any columns remain all NaN
            nan_cols = df.columns[df.isna().all()].tolist()
            if nan_cols:
                raise ValueError(f"These validly fetched tickers contain only NaN values over the requested timeframe: {nan_cols}")

            logger.info(f"Successfully fetched underlying data. Shape: {df.shape}")
            
            result = {
                "etf_name": etf_name,
                "weights": weights,
                "price_data": df
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching underlying stock data: {str(e)}")
            raise

if __name__ == "__main__":
    # Example input
    mock_input = {
        "etf_name": "NIFTY_IT",
        "reporting_date": "2023-10-31",
        "holdings": [
            {"ticker": "TCS.NS", "weight": 0.26},
            {"ticker": "INFY.NS", "weight": 0.25},
            {"ticker": "HCLTECH.NS", "weight": 0.10},
            {"ticker": "WIPRO.NS", "weight": 0.08}
        ]
    }
    
    fetcher = ETFDataFetcher()
    output = fetcher.fetch_data(mock_input)
    print(f"ETF: {output['etf_name']}")
    print(f"Weights: {output['weights']}")
    print(output['price_data'].head())
    print("...")
    print(output['price_data'].tail())
