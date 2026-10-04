import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional, List, Any

logger = logging.getLogger(__name__)

class DatasetBuilder:
    """
    Builds a Panel Dataset using rolling windows for continuous risk forecasting.
    Produces temporal lagged features for Volatility and Maximum Drawdown.
    """
    
    WINDOW_LENGTH = 252   # ~1 year lookback (trading days)
    FORECAST_HORIZON = 21 # ~1 month forward horizon (trading days)
    STEP_SIZE = 21        # ~1 month rolling step
    LAGS = [0, 5, 21, 63] # Lags to compute features at (0 is t)
    
    FEATURE_SETS = {
        "v1_baseline": [
            "Vol_t", "Vol_t5", "Vol_t21", "Vol_t63",
            "MaxDD_t", "MaxDD_t5", "MaxDD_t21", "MaxDD_t63"
        ],
        "family_a": [
            "Vol_t", "Vol_t5", "Vol_t21", "Vol_t63",
            "MaxDD_t", "MaxDD_t5", "MaxDD_t21", "MaxDD_t63"
        ],
        "family_b": [
            "Vol_t", "Vol_t5", "Vol_t21", "Vol_t63",
            "MaxDD_t", "MaxDD_t5", "MaxDD_t21", "MaxDD_t63",
            "Delta_Vol_5", "Delta_Vol_21", "Delta_Vol_63",
            "Vol_Ratio_21_63",
            "Delta_MaxDD_5", "Delta_MaxDD_21", "Delta_MaxDD_63"
        ],
        "v1_1": [
            "Vol_t", "Vol_t5", "Vol_t21", "Vol_t63",
            "MaxDD_t", "MaxDD_t5", "MaxDD_t21", "MaxDD_t63",
            "Delta_Vol_5", "Delta_Vol_21", "Delta_Vol_63",
            "Vol_Ratio_21_63",
            "Delta_MaxDD_5", "Delta_MaxDD_21", "Delta_MaxDD_63"
        ],
        "family_b_ewma": [
            "Vol_t", "Vol_t5", "Vol_t21", "Vol_t63",
            "MaxDD_t", "MaxDD_t5", "MaxDD_t21", "MaxDD_t63",
            "Delta_Vol_5", "Delta_Vol_21", "Delta_Vol_63",
            "Vol_Ratio_21_63",
            "Delta_MaxDD_5", "Delta_MaxDD_21", "Delta_MaxDD_63",
            "EWMA_Vol_t"
        ],
        "family_b_plus_ewma": [
            "Vol_t", "Vol_t5", "Vol_t21", "Vol_t63",
            "MaxDD_t", "MaxDD_t5", "MaxDD_t21", "MaxDD_t63",
            "Delta_Vol_5", "Delta_Vol_21", "Delta_Vol_63",
            "Vol_Ratio_21_63",
            "Delta_MaxDD_5", "Delta_MaxDD_21", "Delta_MaxDD_63",
            "EWMA_Vol_t"
        ]
    }

    def __init__(
        self, 
        portfolios: Dict[str, pd.DataFrame], 
        feature_set: str = "v1_baseline",
        **kwargs # Accept legacy kwargs to avoid breaking existing imports/calls during transition
    ):
        """
        Initializes the DatasetBuilder with reconstructed portfolio returns.
        
        Args:
            portfolios (Dict[str, pd.DataFrame]): A mapping from a portfolio identifier (e.g., ETF ticker)
                                                  to its reconstructed returns DataFrame (must contain 'Daily_Return').
            feature_set (str): Feature set configuration ('v1_baseline' / 'family_a' or 'family_b' / 'v1_1').
        """
        if not portfolios:
            raise ValueError("Provided portfolios dictionary is empty.")
            
        self.portfolios = portfolios
        self.feature_set = feature_set

    @staticmethod
    def _compute_vol(returns: pd.Series) -> float:
        if len(returns) < 5: return np.nan
        return float(returns.std() * np.sqrt(252))

    @staticmethod
    def _compute_maxdd(returns: pd.Series) -> float:
        if len(returns) < 5: return np.nan
        cum_ret = (1.0 + returns).cumprod()
        running_max = cum_ret.cummax()
        drawdown = (cum_ret - running_max) / running_max
        return float(abs(drawdown.min())) if len(drawdown) > 0 else 0.0

    @staticmethod
    def _compute_ewma_vol(returns: pd.Series, decay: float = 0.94) -> float:
        r = returns.dropna().values
        if len(r) < 5:
            return np.nan
        n = len(r)
        weights = (1.0 - decay) * (decay ** np.arange(n - 1, -1, -1))
        weights /= weights.sum()
        ewma_var = np.sum(weights * (r ** 2))
        return float(np.sqrt(252.0 * ewma_var))

    @staticmethod
    def compute_forward_target(future_returns: pd.Series) -> Tuple[float, float]:
        """
        Computes forward-looking realized risk metrics for regression targets.
        """
        fwd_vol = DatasetBuilder._compute_vol(future_returns)
        fwd_maxdd = DatasetBuilder._compute_maxdd(future_returns)
        return fwd_vol, fwd_maxdd

    def build_panel_dataset(self, drop_incomplete: bool = True, feature_set: Optional[str] = None) -> pd.DataFrame:
        """
        Applies rolling windows to each reconstructed portfolio and computes temporal 
        lag features paired with forward-looking risk targets.
        
        Returns:
            pd.DataFrame: Panel Dataset structured as:
                          Portfolio_ID | Date_t | Target_Start | Target_End | EWMA_Vol_t |
                          Vol_t | Vol_t5 | ... | Forward_Vol | Forward_MaxDD
        """
        rows = []
        max_lag = max(self.LAGS)
        min_required = self.WINDOW_LENGTH + max_lag + self.FORECAST_HORIZON
        
        for portfolio_id, portfolio_df in self.portfolios.items():
            if 'Daily_Return' not in portfolio_df.columns:
                logger.warning(f"Portfolio {portfolio_id} is missing 'Daily_Return' column. Skipping.")
                continue
                
            daily_returns = portfolio_df['Daily_Return'].dropna()
            n_days = len(daily_returns)
            
            if n_days < min_required:
                logger.warning(
                    f"Not enough data for {portfolio_id}. Required: {min_required}, Available: {n_days}. Skipping."
                )
                continue
            
            # end_idx represents 't' (the boundary between lookback and forward forecast)
            start_t = self.WINDOW_LENGTH + max_lag
            max_t = n_days - self.FORECAST_HORIZON
            
            for end_idx in range(start_t, max_t + 1, self.STEP_SIZE):
                future_returns = daily_returns.iloc[end_idx : end_idx + self.FORECAST_HORIZON]
                
                # Verify exact horizon length
                if len(future_returns) < self.FORECAST_HORIZON:
                    continue
                
                # The exact date t that separates history from future
                window_date_t = daily_returns.index[end_idx - 1]
                target_start = future_returns.index[0]
                target_end = future_returns.index[-1]
                
                # Compute EWMA baseline forecast over the current 252-day lookback [t-252:t]
                current_lookback_returns = daily_returns.iloc[end_idx - self.WINDOW_LENGTH : end_idx]
                ewma_vol = self._compute_ewma_vol(current_lookback_returns)
                
                row = {
                    "Portfolio_ID": portfolio_id,
                    "Date_t": window_date_t,
                    "Target_Start": target_start,
                    "Target_End": target_end,
                    "EWMA_Vol_t": ewma_vol,
                }
                
                features_valid = True
                
                # Compute historical risk at t, t-5, t-21, t-63
                for lag in self.LAGS:
                    lag_end = end_idx - lag
                    lag_start = lag_end - self.WINDOW_LENGTH
                    
                    if lag_start < 0:
                        features_valid = False
                        break
                        
                    lag_returns = daily_returns.iloc[lag_start:lag_end]
                    
                    suffix = f"_t{lag}" if lag > 0 else "_t"
                    vol = self._compute_vol(lag_returns)
                    maxdd = self._compute_maxdd(lag_returns)
                    
                    if pd.isna(vol) or pd.isna(maxdd):
                        features_valid = False
                        break
                        
                    row[f"Vol{suffix}"] = vol
                    row[f"MaxDD{suffix}"] = maxdd
                
                if drop_incomplete and not features_valid:
                    continue
                    
                fwd_vol, fwd_maxdd = self.compute_forward_target(future_returns)
                
                if drop_incomplete and (pd.isna(fwd_vol) or pd.isna(fwd_maxdd)):
                    continue
                    
                row["Forward_Vol"] = fwd_vol
                row["Forward_MaxDD"] = fwd_maxdd
                
                rows.append(row)
                
        panel_df = pd.DataFrame(rows)
        active_fs = feature_set or self.feature_set
        if active_fs in ["family_b", "v1_1", "family_b_ewma", "family_b_plus_ewma"]:
            panel_df = self.add_family_b_features(panel_df)
        return panel_df

    @staticmethod
    def add_family_b_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Derives the exactly seven Family B features:
        - Volatility dynamics:
            Delta_Vol_5 = Vol_t - Vol_t5
            Delta_Vol_21 = Vol_t - Vol_t21
            Delta_Vol_63 = Vol_t - Vol_t63
        - Relative volatility / regime feature:
            Vol_Ratio_21_63 = Vol_t / Vol_t63
        - Drawdown dynamics:
            Delta_MaxDD_5 = MaxDD_t - MaxDD_t5
            Delta_MaxDD_21 = MaxDD_t - MaxDD_t21
            Delta_MaxDD_63 = MaxDD_t - MaxDD_t63
        """
        res = df.copy()
        res["Delta_Vol_5"] = res["Vol_t"] - res["Vol_t5"]
        res["Delta_Vol_21"] = res["Vol_t"] - res["Vol_t21"]
        res["Delta_Vol_63"] = res["Vol_t"] - res["Vol_t63"]
        res["Vol_Ratio_21_63"] = res["Vol_t"] / res["Vol_t63"]
        res["Delta_MaxDD_5"] = res["MaxDD_t"] - res["MaxDD_t5"]
        res["Delta_MaxDD_21"] = res["MaxDD_t"] - res["MaxDD_t21"]
        res["Delta_MaxDD_63"] = res["MaxDD_t"] - res["MaxDD_t63"]
        return res

    @classmethod
    def compute_inference_features(
        cls, 
        daily_returns: pd.Series, 
        feature_set: str = "family_b_ewma"
    ) -> Dict[str, Any]:
        """
        Computes the exact feature vector from portfolio daily returns for live inference.
        Requires at least WINDOW_LENGTH + max(LAGS) = 252 + 63 = 315 trading days.
        
        Returns:
            Dict containing:
                - 'features_df': pd.DataFrame with 1 row and exactly the model features
                - 'ewma_vol': float, benchmark EWMA volatility over current lookback
                - 'feature_values': Dict[str, float]
        """
        clean_returns = daily_returns.dropna()
        n_days = len(clean_returns)
        if n_days < cls.WINDOW_LENGTH:
            raise ValueError(
                f"Insufficient historical data: {n_days} trading days available, "
                f"at least {cls.WINDOW_LENGTH} required for rolling risk metrics."
            )
            
        min_required = cls.WINDOW_LENGTH + max(cls.LAGS)
        if n_days < min_required:
            pad_len = min_required - n_days
            first_val = float(clean_returns.iloc[0])
            pad_series = pd.Series([first_val] * pad_len, index=pd.date_range(end=clean_returns.index[0] - pd.Timedelta(days=1), periods=pad_len, freq="B"))
            clean_returns = pd.concat([pad_series, clean_returns])
            n_days = len(clean_returns)
            
        end_idx = n_days
        current_lookback_returns = clean_returns.iloc[end_idx - cls.WINDOW_LENGTH : end_idx]
        ewma_vol = cls._compute_ewma_vol(current_lookback_returns)
        
        row = {}
        for lag in cls.LAGS:
            lag_end = end_idx - lag
            lag_start = lag_end - cls.WINDOW_LENGTH
            lag_returns = clean_returns.iloc[lag_start:lag_end]
            
            suffix = f"_t{lag}" if lag > 0 else "_t"
            row[f"Vol{suffix}"] = cls._compute_vol(lag_returns)
            row[f"MaxDD{suffix}"] = cls._compute_maxdd(lag_returns)
        row["EWMA_Vol_t"] = ewma_vol
        df = pd.DataFrame([row])
        if feature_set in ["family_b", "v1_1", "family_b_ewma", "family_b_plus_ewma"]:
            df = cls.add_family_b_features(df)
            
        expected_features = cls.FEATURE_SETS.get(feature_set, cls.FEATURE_SETS["family_b_ewma"])
        df_features = df[expected_features]
        
        return {
            "features_df": df_features,
            "ewma_vol": float(ewma_vol) if not pd.isna(ewma_vol) else 0.0,
            "feature_values": {k: float(v) for k, v in df.iloc[0].to_dict().items()}
        }



