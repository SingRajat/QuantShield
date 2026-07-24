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
        
        # Use first portfolio's Daily_Return as a market proxy for Beta calculation
        first_portfolio = list(self.portfolios.keys())[0]
        self.market_proxy = self.portfolios[first_portfolio]['Daily_Return'].dropna()
        
    def _assign_risk_label(self, vol: float, var95: float, max_dd: float, div_ratio: float = 1.0, 
                           skewness: float = 0.0, kurtosis: float = 0.0, 
                           rolling_vol_20: float = 0.0, rolling_vol_60: float = 0.0,
                           sharpe: float = 0.0, sortino: float = 0.0, beta: float = 1.0) -> str:
        """
        Assigns a Risk Class (Low, Medium, High) using a continuous composite score.
        Introduces fuzzy boundaries to prevent perfect rule reconstruction by ML models.
        """
        # Normalize original metrics
        norm_vol = min(vol / 0.25, 1.0)        # Assume 25% vol is extreme
        norm_var = min(var95 / 0.05, 1.0)      # Assume 5% daily VaR is extreme
        norm_dd = min(max_dd / 0.30, 1.0)      # Assume 30% drawdown is extreme
        # Invert div_ratio (higher ratio = lower risk)
        norm_div_penalty = 1.0 - min(max(div_ratio - 1.0, 0), 1.0) 

        # Normalize new metrics safely with fallbacks if nan
        safe_skew = 0.0 if pd.isna(skewness) else skewness
        norm_skew_penalty = min(abs(min(safe_skew, 0)) / 2.0, 1.0) # 0 to 1

        safe_kurt = 0.0 if pd.isna(kurtosis) else kurtosis
        norm_kurt_penalty = min(max(safe_kurt, 0) / 5.0, 1.0)

        safe_beta = 1.0 if pd.isna(beta) else beta
        norm_beta_penalty = min(max(safe_beta - 1.0, 0) / 0.5, 1.0)

        safe_sortino = 0.0 if pd.isna(sortino) else sortino
        norm_sortino_penalty = 1.0 - min(max(safe_sortino, 0) / 2.0, 1.0)

        # Create a non-linear composite risk score (0 to 1)
        # Weights emphasize severe downside (MaxDD and VaR) over pure Volatility
        core_score = (0.25 * norm_vol) + (0.35 * norm_var) + (0.15 * norm_dd)
        
        # New tail/exposure factors (25% of weight)
        tail_score = (0.10 * norm_skew_penalty) + (0.05 * norm_kurt_penalty) + (0.05 * norm_beta_penalty) + (0.05 * norm_sortino_penalty)
        
        composite_score = core_score + tail_score
        
        # Apply diversification penalty as a modifier
        composite_score = composite_score * (1.0 + (0.15 * norm_div_penalty))
        
        # Base logical cutoff thresholds
        low_threshold = 0.35
        high_threshold = 0.65

        # Introduce realistic "fuzziness" (overlap) at the boundaries 
        if composite_score < low_threshold:
            if max_dd > 0.15 and vol < 0.10: 
                return "Medium"
            if safe_beta > 1.5:
                return "Medium"
            return "Low"
            
        elif composite_score > high_threshold:
            if div_ratio > 1.8 and vol < 0.25:
                return "Medium"
            if safe_sortino > 2.0:
                return "Medium"
            return "High"
            
        else:
            if norm_var > 0.8: 
                 return "High"
            if composite_score < 0.45 and max_dd < 0.10: 
                 return "Low"
            if safe_kurt > 5.0 or safe_skew < -1.5:
                 return "High"
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
                
                # Slice market proxy for beta calculation
                window_market_returns = None
                if self.market_proxy is not None:
                     try:
                         window_market_returns = self.market_proxy.loc[window_start:window_end]
                     except Exception:
                         pass
                
                engineer = RiskFeatureEngineer(
                    portfolio_returns=window_returns,
                    component_returns=window_component_returns,
                    weights=weights,
                    market_returns=window_market_returns
                )
                
                features = engineer.compute_all_features()
                
                vol = features.get("Annualized_Volatility", np.nan)
                var95 = features.get("Historical_VaR_95", np.nan)
                max_dd = features.get("Maximum_Drawdown", np.nan)
                div_ratio = features.get("Diversification_Ratio", 1.0)
                skewness = features.get("Skewness", np.nan)
                kurtosis = features.get("Kurtosis", np.nan)
                rolling_vol_20 = features.get("RollingVol20", np.nan)
                rolling_vol_60 = features.get("RollingVol60", np.nan)
                sharpe = features.get("Sharpe", np.nan)
                sortino = features.get("Sortino", np.nan)
                beta = features.get("Beta", np.nan)
                
                label = self._assign_risk_label(vol, var95, max_dd, div_ratio, skewness, kurtosis, rolling_vol_20, rolling_vol_60, sharpe, sortino, beta)
                
                rows.append({
                    "Portfolio_ID": portfolio_id,
                    "Window_Start": window_start,
                    "Window_End": window_end,
                    "Vol": vol,
                    "VaR95": var95,
                    "MaxDD": max_dd,
                    "DivRatio": div_ratio,
                    "Skewness": skewness,
                    "Kurtosis": kurtosis,
                    "RollingVol20": rolling_vol_20,
                    "RollingVol60": rolling_vol_60,
                    "Sharpe": sharpe,
                    "Sortino": sortino,
                    "Beta": beta,
                    "Label": label
                })
                
        panel_df = pd.DataFrame(rows)
        return panel_df
