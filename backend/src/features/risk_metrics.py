import pandas as pd
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class RiskFeatureEngineer:
    """
    Computes specific Risk Metrics from daily portfolio returns.
    Metrics allowed:
    1. Annualized Volatility
    2. Historical VaR (95%)
    3. Maximum Drawdown
    4. Diversification Ratio
    """
    
    TRADING_DAYS_PER_YEAR = 252

    def __init__(self, portfolio_returns: pd.Series, component_returns: pd.DataFrame = None, weights: Dict[str, float] = None):
        """
        Initializes the engineer.
        
        Args:
            portfolio_returns (pd.Series): The daily returns of the portfolio.
            component_returns (pd.DataFrame): Daily returns of the individual assets (needed for Diversification Ratio).
            weights (Dict[str, float]): The normalized weights of the assets in the portfolio (needed for Diversification Ratio).
        """
        if portfolio_returns.empty:
            raise ValueError("Provided portfolio_returns is empty.")
            
        self.portfolio_returns = portfolio_returns
        self.component_returns = component_returns
        self.weights = weights
        
    def compute_annualized_volatility(self) -> float:
        """
        Calculates the annualized volatility of the portfolio based on daily returns.
        """
        daily_vol = self.portfolio_returns.std()
        annualized_vol = daily_vol * np.sqrt(self.TRADING_DAYS_PER_YEAR)
        return float(annualized_vol)

    def compute_historical_var_95(self) -> float:
        """
        Calculates the Historical Value at Risk (VaR) at 95% confidence.
        Represents the minimum expected loss over the next day in the worst 5% of cases.
        Output is expressed as a positive number (a loss amount).
        """
        # 5th percentile of returns represents the threshold for the worst 5% of days
        var_95 = np.percentile(self.portfolio_returns, 5)
        # Returns generally are negative for losses, VaR is typically expressed as a positive maximum loss
        return float(abs(var_95))

    def compute_max_drawdown(self) -> float:
        """
        Calculates the Maximum Drawdown.
        Represents the maximum observed loss linearly from a historical peak.
        """
        cumulative_returns = (1 + self.portfolio_returns).cumprod()
        rolling_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        return float(abs(max_drawdown))

    def compute_diversification_ratio(self) -> float:
        """
        Calculates the Diversification Ratio of the portfolio.
        Ratio = (Weighted average of individual asset volatilities) / (Portfolio volatility)
        Requires component_returns and weights to be provided during initialization.
        """
        if self.component_returns is None or self.weights is None:
            logger.warning("Component returns or weights not provided. Cannot compute Diversification Ratio.")
            return np.nan
            
        # 1. Compute individual daily volatilities
        individual_vols = self.component_returns.std()
        
        # 2. Compute the weighted average of individual volatilities
        weighted_avg_vol = 0.0
        for ticker, weight in self.weights.items():
            if ticker in individual_vols:
               weighted_avg_vol += weight * individual_vols[ticker]
               
        # 3. Get portfolio volatility (daily)
        portfolio_vol = self.portfolio_returns.std()
        
        if np.isclose(portfolio_vol, 0.0):
            return 1.0 # If risk is 0, return 1 to avoid ZeroDivisionError
            
        diversification_ratio = weighted_avg_vol / portfolio_vol
        return float(diversification_ratio)

    def compute_all_features(self) -> Dict[str, float]:
        """
        Computes all 4 allowed risk features and returns them as a dictionary.
        """
        features = {
            "Annualized_Volatility": self.compute_annualized_volatility(),
            "Historical_VaR_95": self.compute_historical_var_95(),
            "Maximum_Drawdown": self.compute_max_drawdown()
        }
        
        # Only add Diversification Ratio if dependencies are met
        if self.component_returns is not None and self.weights is not None:
             features["Diversification_Ratio"] = self.compute_diversification_ratio()
             
        return features
