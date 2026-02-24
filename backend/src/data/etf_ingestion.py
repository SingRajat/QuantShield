import yfinance as yf
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ETFDataFetcher:
    """
    Fetches historical price data for a specified list of ETFs.
    """
    
    DEFAULT_TICKERS = [
        "SETFNIF50.NS", "HDFCSENSEX.NS",  # Broad Market
        "SETFNIFBK.NS", "ITBEES.NS", "NETFPHARMA.NS",  # Sector
        "JUNIORBEES.NS", "MOM50.NS", "CPSEETF.NS"  # Higher Beta
    ]
    
    def __init__(self, tickers=None, years=5):
        self.tickers = tickers if tickers else self.DEFAULT_TICKERS
        self.years = years
        
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetches 'Adj Close' prices for the tickers spanning the last `self.years` years.
        Handles missing values via forward fill and then backward fill.
        """
        end_date = datetime.today()
        start_date = end_date - relativedelta(years=self.years)
        
        logger.info(f"Fetching data for {len(self.tickers)} tickers from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        try:
            # yfinance returns a MultiIndex DataFrame if multiple tickers are provided
            data = yf.download(self.tickers, start=start_date, end=end_date, progress=False)
            
            # Extract 'Adj Close' or 'Close' using cross-section or checking levels
            if isinstance(data.columns, pd.MultiIndex):
                # yfinance >= 0.2.x returns MultiIndex
                price_level = data.columns.get_level_values(0).unique()
                if 'Adj Close' in price_level:
                    df = data.xs('Adj Close', axis=1, level=0)
                elif 'Close' in price_level:
                    df = data.xs('Close', axis=1, level=0)
                else:
                    raise ValueError("Neither 'Adj Close' nor 'Close' found in downloaded data.")
            else:
                # Fallback for single ticker or older yfinance
                if 'Adj Close' in data.columns:
                    df = data['Adj Close']
                elif 'Close' in data.columns:
                    df = data['Close']
                else:
                    raise ValueError("Neither 'Adj Close' nor 'Close' found in downloaded data.")
                
                if isinstance(df, pd.Series):
                    df = df.to_frame(name=self.tickers[0])
                
            # Handle missing values
            # Forward fill to carry over previous day's price for holidays
            df = df.ffill()
            # Backward fill to handle missing starting prices
            df = df.bfill()
            
            logger.info(f"Successfully fetched data. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching ETF data: {str(e)}")
            raise

if __name__ == "__main__":
    fetcher = ETFDataFetcher()
    df = fetcher.fetch_data()
    print(df.head())
    print(df.tail())
    print(df.info())
