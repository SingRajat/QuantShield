import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple
from src.features.risk_metrics import RiskFeatureEngineer

logger = logging.getLogger(__name__)

class DatasetBuilder:
    """
    Builds a Panel Dataset using rolling windows for ML classification.
    Operates on reconstructed portfolio returns to maintain architectural consistency
    between training and inference pipelines.
    """
    
    WINDOW_LENGTH = 126  # ~6 months (trading days)
    STEP_SIZE = 21       # ~1 month (trading days)
    
    def __init__(self, portfolios: Dict[str, pd.DataFrame], component_returns_dict: Dict[str, pd.DataFrame] = None, weights_dict: Dict[str, Dict[str, float]] = None):
        """
        Initializes the DatasetBuilder with reconstructed portfolio returns.
        
        Args:
            portfolios (Dict[str, pd.DataFrame]): A mapping from a portfolio identifier (e.g., ETF ticker or Portfolio ID)
                                                  to its reconstructed returns DataFrame (must contain 'Daily_Return').
            component_returns_dict (Dict[str, pd.DataFrame]): Mapping from portfolio ID to its component returns DataFrame.
                                                              Required for Diversification Ratio.
            weights_dict (Dict[str, Dict[str, float]]): Mapping from portfolio ID to its asset weights.
                                                        Required for Diversification Ratio.
        """
        if not portfolios:
            raise ValueError("Provided portfolios dictionary is empty.")
            
        self.portfolios = portfolios
        self.component_returns_dict = component_returns_dict or {}
        self.weights_dict = weights_dict or {}
        
    def _assign_risk_label(self, vol: float, var95: float, max_dd: float) -> str:
        """
        Assigns a predefined Risk Class (Low, Medium, High) based on computed metrics.
        This rule-based label provides the target for the ML classification model.
        """
        if vol < 0.12 and max_dd < 0.15:
            return "Low"
        elif vol > 0.20 or max_dd > 0.25 or var95 > 0.03:
            return "High"
        else:
            return "Medium"

    def build_panel_dataset(self) -> pd.DataFrame:
        """
        Applies rolling windows to each reconstructed portfolio and computes exactly 4 approved features.
        
        Returns:
            pd.DataFrame: Panel Dataset structured as:
                          Portfolio_ID | Window_Start | Window_End | Vol | VaR95 | MaxDD | DivRatio | Label
        """
        rows = []
        
        for portfolio_id, portfolio_df in self.portfolios.items():
            if 'Daily_Return' not in portfolio_df.columns:
                logger.warning(f"Portfolio {portfolio_id} is missing 'Daily_Return' column. Skipping.")
                continue
                
            daily_returns = portfolio_df['Daily_Return'].dropna()
            n_days = len(daily_returns)
            
            if n_days < self.WINDOW_LENGTH:
                logger.warning(f"Not enough data for {portfolio_id}. Required: {self.WINDOW_LENGTH}, Available: {n_days}. Skipping.")
                continue
                
            component_returns = self.component_returns_dict.get(portfolio_id)
            weights = self.weights_dict.get(portfolio_id)
                
            # Apply rolling windows
            for start_idx in range(0, n_days - self.WINDOW_LENGTH + 1, self.STEP_SIZE):
                end_idx = start_idx + self.WINDOW_LENGTH
                window_returns = daily_returns.iloc[start_idx:end_idx]
                
                window_start = window_returns.index[0]
                window_end = window_returns.index[-1]
                
                # Slice component returns if available
                window_component_returns = None
                if component_returns is not None:
                     # Attempt to slice component returns to match the window
                     try:
                         # Ensure alignment by index limits
                         window_component_returns = component_returns.loc[window_start:window_end]
                     except Exception as e:
                         logger.warning(f"Could not slice component returns for {portfolio_id}: {e}")
                
                engineer = RiskFeatureEngineer(
                    portfolio_returns=window_returns,
                    component_returns=window_component_returns,
                    weights=weights
                )
                
                features = engineer.compute_all_features()
                
                vol = features.get("Annualized_Volatility", np.nan)
                var95 = features.get("Historical_VaR_95", np.nan)
                max_dd = features.get("Maximum_Drawdown", np.nan)
                div_ratio = features.get("Diversification_Ratio", 1.0) # Default to 1.0 if untrackable
                
                label = self._assign_risk_label(vol, var95, max_dd)
                
                rows.append({
                    "Portfolio_ID": portfolio_id,
                    "Window_Start": window_start,
                    "Window_End": window_end,
                    "Vol": vol,
                    "VaR95": var95,
                    "MaxDD": max_dd,
                    "DivRatio": div_ratio,
                    "Label": label
                })
                
        panel_df = pd.DataFrame(rows)
        return panel_df
