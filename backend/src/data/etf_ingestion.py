import yfinance as yf
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CACHE = {}

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
        and fetches 'Adj Close' prices. Uses in-memory CACHE.
        """
        if "etf_name" not in holdings_input or "holdings" not in holdings_input:
            raise ValueError("Input must contain 'etf_name' and 'holdings' keys.")
            
        etf_name = holdings_input["etf_name"]
        holdings_list = holdings_input["holdings"]
        benchmark = holdings_input.get("benchmark", "^NSEI")
        
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
            
        # Verify weights approximately sum to 1.0
        total_weight = sum(weights.values())
        if not (0.95 <= total_weight <= 1.05):
            logger.warning(f"Total weights sum to {total_weight}. Ensure weights are normalized to 1.0.")

        # Fetch Data
        end_date = datetime.today()
        start_date = end_date - relativedelta(years=self.years)
        
        all_tickers = tickers_to_fetch + [benchmark]
        logger.info(f"Fetching data for {len(all_tickers)} tickers from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        global CACHE
        dfs = []
        
        for ticker in all_tickers:
            if ticker in CACHE:
                dfs.append(CACHE[ticker])
            else:
                try:
                    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    if df.empty:
                        raise ValueError(f"Downloaded data is completely empty for {ticker}.")
                    
                    
                    price_level = df.columns.get_level_values(0).unique()
                    if 'Adj Close' in price_level:
                        df_close = df.xs('Adj Close', axis=1, level=0)
                    elif 'Close' in price_level:
                        df_close = df.xs('Close', axis=1, level=0)
                        logger.warning(f"Falling back to 'Close' for {ticker}.")
                    else:
                        raise ValueError(f"Neither 'Adj Close' nor 'Close' found for {ticker}.")
                        
                    # Rename the single column back to the ticker name since yfinance might name it the ticker anyway
                    df_close.columns = [ticker]
                    
                    df_close = df_close.ffill().bfill()
                    if df_close.isna().all().any():
                        raise ValueError(f"Ticker {ticker} contains only NaN values.")
                        
                    CACHE[ticker] = df_close
                    dfs.append(df_close)
                except Exception as e:
                    if ticker == benchmark and ticker != "^NSEI":
                        logger.warning(f"Failed to fetch custom benchmark {ticker}. Falling back to ^NSEI.")
                        try:
                            if "^NSEI" in CACHE:
                                dfs.append(CACHE["^NSEI"])
                            else:
                                df_bench = yf.download("^NSEI", start=start_date, end=end_date, progress=False)
                                df_close = df_bench['Adj Close'].to_frame(name="^NSEI")
                                df_close = df_close.ffill().bfill()
                                CACHE["^NSEI"] = df_close
                                dfs.append(df_close)
                            benchmark = "^NSEI"
                        except Exception as fallback_e:
                            raise ValueError(f"Failed benchmark fallback: {fallback_e}")
                    else:
                        raise ValueError(f"Failed to fetch data for {ticker}: {e}")

        logger.info(f"Successfully fetched underlying data.")
        
        combined_df = pd.concat(dfs, axis=1)
        price_data = combined_df[tickers_to_fetch]
        benchmark_data = combined_df[[benchmark]]
        
        result = {
            "etf_name": etf_name,
            "weights": weights,
            "price_data": price_data,
            "benchmark_data": benchmark_data,
            "benchmark_name": benchmark
        }
        return result
        
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
