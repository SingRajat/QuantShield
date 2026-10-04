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
                raise ValueError(f"Downloaded data is completely empty. Please verify tickers: {tickers_to_fetch}")
            
            # Extract 'Adj Close' or 'Close' robustly
            if isinstance(data.columns, pd.MultiIndex):
                df_adj = data.xs('Adj Close', axis=1, level=0) if 'Adj Close' in data.columns.get_level_values(0) else pd.DataFrame()
                df_close = data.xs('Close', axis=1, level=0) if 'Close' in data.columns.get_level_values(0) else pd.DataFrame()
                
                adj_valid_cols = df_adj.dropna(axis=1, how='all').shape[1] if not df_adj.empty else 0
                close_valid_cols = df_close.dropna(axis=1, how='all').shape[1] if not df_close.empty else 0
                
                if adj_valid_cols >= close_valid_cols and adj_valid_cols > 0:
                    df = df_adj
                elif close_valid_cols > 0:
                    df = df_close
                    logger.warning("Using 'Close' instead of 'Adj Close' due to missing data.")
                else:
                    raise ValueError("Neither 'Adj Close' nor 'Close' returned valid data.")
            else:
                # Fallback for single ticker
                if 'Adj Close' in data.columns:
                    df = data['Adj Close'].to_frame(name=tickers_to_fetch[0])
                elif 'Close' in data.columns:
                    df = data['Close'].to_frame(name=tickers_to_fetch[0])
                    logger.warning("Falling back to 'Close'. 'Adj Close' not found.")
                else:
                    raise ValueError("Neither 'Adj Close' nor 'Close' found in downloaded data.")
            
            # Handle missing or delisted tickers gracefully
            missing_tickers = [t for t in tickers_to_fetch if t not in df.columns or df[t].isna().all()]
            if missing_tickers:
                 logger.warning(f"Failed to fetch data for {missing_tickers}. Dropping them and re-normalizing weights.")
                 df = df.drop(columns=[t for t in missing_tickers if t in df.columns])
                 for t in missing_tickers:
                     if t in weights:
                         del weights[t]
                 if not weights:
                     raise ValueError("All requested tickers failed to download or are delisted.")
                 
                 # Re-normalize weights
                 total_w = sum(weights.values())
                 weights = {t: w / total_w for t, w in weights.items()}

            # Data Integrity: Forward fill short gaps (holidays/weekends) up to 5 days.
            # Do NOT bfill(): leaving leading missing values as NaN ensures pre-IPO/late-listing periods
            # are not masked with zero-volatility flat prices.
            df = df.ffill(limit=5)
            
            # Second pass: check if any columns remain all NaN
            nan_cols = [c for c in df.columns if df[c].dropna().empty]
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
