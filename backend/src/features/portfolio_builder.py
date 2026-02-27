import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class PortfolioBuilder:
    """
    Constructs portfolio returns from individual asset returns and weights.
    Also provides functionality to generate random portfolios for training data.
    """
    
    def __init__(self, price_data: pd.DataFrame):
        """
        Initializes the PortfolioBuilder with historical price data.
        
        Args:
            price_data (pd.DataFrame): DataFrame containing daily prices, where columns are ETF tickers
                                     and the index represents dates.
        """
        if price_data.empty:
            raise ValueError("Provided price_data is empty.")
            
        self.price_data = price_data
        # Calculate daily returns
        self.daily_returns = self.price_data.pct_change().dropna()
        self.available_tickers = list(self.daily_returns.columns)
        logger.info(f"Initialized PortfolioBuilder with {len(self.available_tickers)} tickers.")

    def build_portfolio(self, weights: Dict[str, float]) -> pd.DataFrame:
        """
        Calculates the portfolio's aggregated daily and cumulative returns for a given configuration of weights.
        
        Args:
            weights (Dict[str, float]): Dictionary mapping tickers to their target weights (summing to ~1.0).
                                        Example: {"SETFNIF50.NS": 0.6, "ITBEES.NS": 0.4}
                                        
        Returns:
            pd.DataFrame: DataFrame containing 'Daily_Return' and 'Cumulative_Return'
        """
        self._validate_weights(weights)
        
        # Align weights with the DataFrame columns
        aligned_weights = np.array([weights.get(ticker, 0.0) for ticker in self.available_tickers])
        
        # Calculate portfolio daily returns: sum(weight_i * return_i) for each day
        portfolio_daily_returns = self.daily_returns.dot(aligned_weights)
        
        # Calculate cumulative returns
        portfolio_cumulative_returns = (1 + portfolio_daily_returns).cumprod() - 1
        
        result = pd.DataFrame({
            'Daily_Return': portfolio_daily_returns,
            'Cumulative_Return': portfolio_cumulative_returns
        })
        
        return result

    def _validate_weights(self, weights: Dict[str, float]):
        """
        Validates the provided weight dictionary.
        """
        if not weights:
            raise ValueError("Weights dictionary cannot be empty.")
            
        for ticker in weights.keys():
            if ticker not in self.available_tickers:
                raise ValueError(f"Ticker '{ticker}' not found in available price data.")
                
        total_weight = sum(weights.values())
        if not np.isclose(total_weight, 1.0, rtol=1e-5):
            raise ValueError(f"Portfolio weights must sum to 1.0. Current sum: {total_weight}")
