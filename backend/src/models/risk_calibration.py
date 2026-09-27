"""
Empirical Out-Of-Fold (OOF) Risk Calibration Module (Option A).

Provides statistically validated conditional risk probabilities P(Risk Class | Pred_Vol_ML)
calibrated from 4,917 out-of-fold historical predictions from the frozen Family B + EWMA
multi-output regression engine across 5 purged walk-forward cross-validation folds.

Target Risk Regimes (defined on actual realized 21-day forward volatility):
    Low:    Forward_Vol < 0.16 (annualized)
    Medium: 0.16 <= Forward_Vol < 0.25 (annualized)
    High:   Forward_Vol >= 0.25 (annualized)

Statistical Validation Summary (N = 4,917):
    - Brier Score: 0.5398 (vs 0.7210 heuristic, delta = -0.1812, 25.1% improvement)
    - Log Loss:    0.8900 (vs 1.2062 heuristic, delta = -0.3162, 26.2% improvement)
    - ECE:         0.0306 (vs 0.2546 heuristic, delta = -0.2240, 88.0% improvement)
    - Monotonicity: Strictly monotonic for Low and High regimes; stable across all 5 folds.
"""

from typing import Dict, Any
import numpy as np

# Intercepts and Slopes fitted on the complete pooled OOF dataset (N = 4,917)
# Model: Multinomial Logistic (Softmax / Platt Scaling)
CALIBRATION_COEFFICIENTS = {
    "Low": {
        "intercept": 4.9621,
        "slope": -23.7816
    },
    "Medium": {
        "intercept": -0.1479,
        "slope": 3.3690
    },
    "High": {
        "intercept": -4.8142,
        "slope": 20.4126
    }
}

CLASSES = ["Low", "Medium", "High"]


def calibrate_risk_probabilities(pred_vol: float) -> Dict[str, float]:
    """
    Computes calibrated empirical conditional probabilities for Low, Medium, and High
    risk regimes given the continuous predicted volatility forecast.
    
    Args:
        pred_vol (float): Annualized 21-day forward volatility forecast from CatBoost.
        
    Returns:
        Dict[str, float]: Dictionary with keys 'Low', 'Medium', 'High', summing to 1.0.
    """
    v = float(pred_vol)
    
    # Calculate logits for each regime
    logits = np.array([
        CALIBRATION_COEFFICIENTS["Low"]["intercept"] + CALIBRATION_COEFFICIENTS["Low"]["slope"] * v,
        CALIBRATION_COEFFICIENTS["Medium"]["intercept"] + CALIBRATION_COEFFICIENTS["Medium"]["slope"] * v,
        CALIBRATION_COEFFICIENTS["High"]["intercept"] + CALIBRATION_COEFFICIENTS["High"]["slope"] * v
    ], dtype=np.float64)
    
    # Numerically stable softmax: subtract max logit
    shifted_logits = logits - np.max(logits)
    exp_logits = np.exp(shifted_logits)
    probs = exp_logits / np.sum(exp_logits)
    
    p_low = float(probs[0])
    p_med = float(probs[1])
    p_high = float(probs[2])
    
    # Ensure exact sum to 1.0 after rounding to 4 decimals
    r_low = round(p_low, 4)
    r_high = round(p_high, 4)
    r_med = round(1.0 - r_low - r_high, 4)
    
    return {
        "Low": r_low,
        "Medium": r_med,
        "High": r_high
    }


def get_calibration_metadata() -> Dict[str, Any]:
    """Returns empirical calibration metadata and audit metrics for institutional transparency."""
    return {
        "method": "Empirical Out-Of-Fold Softmax Calibration (Platt Scaling)",
        "sample_size": 4917,
        "cross_validation": "5-Fold Purged & Embargoed Walk-Forward Time Series",
        "brier_score": 0.5398,
        "heuristic_brier_score": 0.7210,
        "brier_improvement_absolute": -0.1812,
        "log_loss": 0.8900,
        "heuristic_log_loss": 1.2062,
        "log_loss_improvement_absolute": -0.3162,
        "expected_calibration_error": 0.0306,
        "regime_thresholds": {
            "Low": "Forward_Vol < 0.16",
            "Medium": "0.16 <= Forward_Vol < 0.25",
            "High": "Forward_Vol >= 0.25"
        },
        "coefficients": CALIBRATION_COEFFICIENTS,
        "validation_status": "AUDIT_PASSED"
    }
