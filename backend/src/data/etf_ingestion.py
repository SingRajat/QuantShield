import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
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

        # Fetch Data - honor reporting_date if provided, else default to today
        today_dt = datetime.today().date()
        rep_date_str = holdings_input.get("reporting_date")
        if rep_date_str:
            try:
                as_of_dt = datetime.strptime(str(rep_date_str).strip(), "%Y-%m-%d").date()
            except ValueError:
                raise ValueError(f"Invalid reporting_date format: {rep_date_str}. Expected YYYY-MM-DD.")
            if as_of_dt > today_dt:
                raise ValueError(f"reporting_date cannot be in the future: {as_of_dt} > {today_dt}.")
        else:
            as_of_dt = today_dt

        start_date = as_of_dt - relativedelta(years=self.years)
        end_date = as_of_dt + timedelta(days=1)  # Yahoo Finance end parameter is exclusive
        
        logger.info(f"Fetching data for {len(tickers_to_fetch)} underlying stocks from {start_date} to {end_date} (as-of {as_of_dt})")
        
        try:
            # yfinance returns a MultiIndex DataFrame if multiple tickers are provided
            data = yf.download(
                tickers_to_fetch, 
                start=start_date.strftime("%Y-%m-%d"), 
                end=end_date.strftime("%Y-%m-%d"), 
                progress=False
            )
            
            # Error checking: yfinance might download empty data if no tickers are valid
            if data.empty:
                raise ValueError(f"Downloaded data is completely empty between {start_date} and {as_of_dt}. Please verify tickers.")
            
            # Defensive guard: crop any timestamps strictly greater than as_of_dt
            data = data[data.index.date <= as_of_dt]
            
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
            
            # Handle missing or delisted tickers strictly (fail closed per QS-06)
            missing_tickers = [t for t in tickers_to_fetch if t not in df.columns or df[t].isna().all()]
            if missing_tickers:
                raise ValueError(
                    f"Failed to fetch market data for required tickers: {missing_tickers}. "
                    f"Verify ticker symbols on Yahoo Finance or replace them."
                )

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
