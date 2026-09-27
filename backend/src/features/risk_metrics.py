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

    def __init__(self, portfolio_returns: pd.Series, component_returns: pd.DataFrame = None, weights: Dict[str, float] = None, market_returns: pd.Series = None):
        """
        Initializes the engineer.
        
        Args:
            portfolio_returns (pd.Series): The daily returns of the portfolio.
            component_returns (pd.DataFrame): Daily returns of the individual assets (needed for Diversification Ratio).
            weights (Dict[str, float]): The normalized weights of the assets in the portfolio (needed for Diversification Ratio).
            market_returns (pd.Series): The daily returns of the market benchmark (needed for Beta).
        """
        if portfolio_returns.empty:
            raise ValueError("Provided portfolio_returns is empty.")
            
        self.portfolio_returns = portfolio_returns
        self.component_returns = component_returns
        self.weights = weights
        self.market_returns = market_returns
        
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
        Returns np.nan if components are missing or incomplete.
        """
        if self.component_returns is None or self.weights is None:
            logger.warning("Component returns or weights not provided. Cannot compute Diversification Ratio.")
            return np.nan
            
        # 1. Check that all portfolio assets exist in component_returns
        missing_tickers = [t for t in self.weights.keys() if t not in self.component_returns.columns]
        if missing_tickers:
            logger.warning(f"Tickers missing from component returns: {missing_tickers}")
            return np.nan

        # 2. Compute individual daily volatilities
        individual_vols = self.component_returns.std()
        if individual_vols.isna().any():
            return np.nan
        
        # 3. Compute the weighted average of individual volatilities
        weighted_avg_vol = 0.0
        for ticker, weight in self.weights.items():
            weighted_avg_vol += weight * individual_vols[ticker]
               
        # 4. Get portfolio volatility (daily)
        portfolio_vol = self.portfolio_returns.std()
        
        if pd.isna(portfolio_vol):
            return np.nan
        if np.isclose(portfolio_vol, 0.0):
            return 1.0 # If risk is 0, return 1 to avoid ZeroDivisionError
            
        diversification_ratio = weighted_avg_vol / portfolio_vol
        return float(diversification_ratio)

    def compute_skewness(self) -> float:
        return float(self.portfolio_returns.skew())

    def compute_kurtosis(self) -> float:
        return float(self.portfolio_returns.kurtosis())

    def compute_rolling_vol(self, window: int) -> float:
        if len(self.portfolio_returns) < window:
            return np.nan
        daily_vol = self.portfolio_returns.tail(window).std()
        return float(daily_vol * np.sqrt(self.TRADING_DAYS_PER_YEAR))

    def compute_sharpe_ratio(self) -> float:
        vol = self.portfolio_returns.std()
        if vol == 0:
            return 0.0
        return float((self.portfolio_returns.mean() / vol) * np.sqrt(self.TRADING_DAYS_PER_YEAR))

    def compute_sortino_ratio(self) -> float:
        downside_returns = self.portfolio_returns[self.portfolio_returns < 0]
        downside_vol = downside_returns.std()
        if pd.isna(downside_vol) or downside_vol == 0:
            return 0.0
        return float((self.portfolio_returns.mean() / downside_vol) * np.sqrt(self.TRADING_DAYS_PER_YEAR))

    def compute_beta(self, min_periods: int = 20) -> float:
        """
        Calculates Beta against market returns after aligning strictly on index dates.
        Returns np.nan if market returns are missing or overlap is insufficient.
        """
        if self.market_returns is None or self.portfolio_returns is None:
            return np.nan
        
        aligned = pd.concat([self.portfolio_returns.rename("port"), self.market_returns.rename("mkt")], axis=1).dropna()
        if len(aligned) < min_periods:
            return np.nan
            
        cov = aligned["port"].cov(aligned["mkt"])
        var_market = aligned["mkt"].var()
        if pd.isna(var_market) or np.isclose(var_market, 0.0):
            return np.nan
        return float(cov / var_market)

    def compute_all_features(self) -> Dict[str, float]:
        """
        Computes all risk features and returns them as a dictionary.
        """
        features = {
            "Annualized_Volatility": self.compute_annualized_volatility(),
            "Historical_VaR_95": self.compute_historical_var_95(),
            "Maximum_Drawdown": self.compute_max_drawdown(),
            "Diversification_Ratio": self.compute_diversification_ratio(),
            "Skewness": self.compute_skewness(),
            "Kurtosis": self.compute_kurtosis(),
            "RollingVol20": self.compute_rolling_vol(20),
            "RollingVol60": self.compute_rolling_vol(60),
            "Sharpe": self.compute_sharpe_ratio(),
            "Sortino": self.compute_sortino_ratio(),
            "Beta": self.compute_beta()
        }
        return features
