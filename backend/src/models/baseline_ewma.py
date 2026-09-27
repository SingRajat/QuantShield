import numpy as np
import pandas as pd
from typing import Dict, Any, Union
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

class EWMABaseline:
    """
    J.P. Morgan RiskMetrics Exponentially Weighted Moving Average (EWMA) Baseline.
    Serves as the benchmark statistical model for forward volatility and risk regime prediction.
    Default decay factor lambda = 0.94 for daily returns.
    """
    
    def __init__(self, decay: float = 0.94, vol_low: float = 0.15, vol_high: float = 0.22):
        self.decay = decay
        self.vol_low = vol_low
        self.vol_high = vol_high

    def forecast_volatility(self, returns: Union[pd.Series, np.ndarray]) -> float:
        """
        Forecasts next-period annualized volatility using exponentially weighted variance.
        
        Args:
            returns: Daily return series.
            
        Returns:
            float: Annualized EWMA volatility forecast.
        """
        if isinstance(returns, pd.Series):
            r = returns.dropna().values
        else:
            r = np.asarray(returns).ravel()
            r = r[~np.isnan(r)]
            
        if len(r) < 5:
            return np.nan
            
        n = len(r)
        # Exponential weights normalized to sum to 1
        weights = (1.0 - self.decay) * (self.decay ** np.arange(n - 1, -1, -1))
        weights /= weights.sum()
        
        ewma_var = np.sum(weights * (r ** 2))
        return float(np.sqrt(252.0 * ewma_var))

    def predict_regime_from_vol(self, vol: float) -> str:
        """
        Maps an annualized volatility forecast to a discrete risk class: Low, Medium, High.
        """
        if pd.isna(vol):
            return "Medium"
        if vol >= self.vol_high:
            return "High"
        elif vol < self.vol_low:
            return "Low"
        return "Medium"

    def predict_regime(self, returns: Union[pd.Series, np.ndarray]) -> str:
        """
        Computes the EWMA forecast from returns and returns the predicted regime.
        """
        vol = self.forecast_volatility(returns)
        return self.predict_regime_from_vol(vol)

    def evaluate(self, y_true: pd.Series, ewma_vols: pd.Series) -> Dict[str, Any]:
        """
        Evaluates discrete regime predictions against true labels (legacy support).
        """
        preds = [self.predict_regime_from_vol(v) for v in ewma_vols]
        acc = float(accuracy_score(y_true, preds))
        _, _, f1, _ = precision_recall_fscore_support(y_true, preds, average='macro', zero_division=0)
        return {
            "accuracy": acc,
            "macro_f1": float(f1),
            "predictions": preds
        }

    def evaluate_regression(self, y_true_vol: pd.Series, ewma_vols: pd.Series) -> Dict[str, Any]:
        """
        Evaluates EWMA baseline volatility forecasts against true future continuous volatility.
        
        Args:
            y_true_vol: True out-of-sample forward volatility.
            ewma_vols: EWMA volatility forecasts for the corresponding windows.
            
        Returns:
            Dict containing mae, rmse, and correlation.
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        from scipy.stats import pearsonr
        
        # Drop NaNs
        mask = ~np.isnan(y_true_vol) & ~np.isnan(ewma_vols)
        y_true = np.asarray(y_true_vol)[mask]
        y_pred = np.asarray(ewma_vols)[mask]
        
        if len(y_true) < 2:
            return {"mae": np.nan, "rmse": np.nan, "correlation": np.nan}
            
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        corr, _ = pearsonr(y_true, y_pred)
        
        return {
            "mae": mae,
            "rmse": rmse,
            "correlation": float(corr)
        }
