import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from backend.src.features.risk_metrics import RiskFeatureEngineer
except ImportError:
    from src.features.risk_metrics import RiskFeatureEngineer

@pytest.fixture
def mock_returns():
    """Mock daily portfolio returns: +1%, -2%, +3%, -1%, +0.5%"""
    return pd.Series([0.01, -0.02, 0.03, -0.01, 0.005])

@pytest.fixture
def mock_component_data():
    """Mock component returns and weights for Diversification Ratio tests"""
    returns = pd.DataFrame({
        'ETF1': [0.01, -0.01],
        'ETF2': [-0.01, 0.01]
    })
    weights = {'ETF1': 0.5, 'ETF2': 0.5}
    return returns, weights

def test_risk_metrics_initialization(mock_returns, mock_component_data):
    comp_returns, weights = mock_component_data
    engineer = RiskFeatureEngineer(mock_returns, comp_returns, weights)
    assert len(engineer.portfolio_returns) == 5
    assert len(engineer.weights) == 2

def test_risk_metrics_empty_data():
    with pytest.raises(ValueError):
        RiskFeatureEngineer(pd.Series(dtype=float))

def test_compute_annualized_volatility(mock_returns):
    engineer = RiskFeatureEngineer(mock_returns)
    daily_vol = mock_returns.std()
    expected_vol = daily_vol * np.sqrt(252)
    assert np.isclose(engineer.compute_annualized_volatility(), expected_vol)

def test_compute_historical_var_95(mock_returns):
    engineer = RiskFeatureEngineer(mock_returns)
    # The 5th percentile of [0.01, -0.02, 0.03, -0.01, 0.005] is approx -0.018
    # VaR should be returned as a positive number
    expected_var = float(abs(np.percentile(mock_returns, 5)))
    assert np.isclose(engineer.compute_historical_var_95(), expected_var)

def test_compute_max_drawdown():
    # Construct a specific series for max drawdown
    # Prices: 100 -> 110 -> 88 -> 96.8
    # Returns: +10%, -20%, +10%
    # Drawdowns from peak (110): 0, -20%
    # Max Drawdown: 20%
    returns = pd.Series([0.10, -0.20, 0.10])
    engineer = RiskFeatureEngineer(returns)
    
    # MDD should be a positive number representing loss
    mdd = engineer.compute_max_drawdown()
    assert np.isclose(mdd, 0.20)

def test_compute_diversification_ratio():
    # Example where individual components are volatile but inversely correlated
    # Resulting in 0 portfolio volatility
    comp_returns = pd.DataFrame({
        'ETF1': [0.10, -0.10, 0.10],
        'ETF2': [-0.10, 0.10, -0.10]
    })
    weights = {'ETF1': 0.5, 'ETF2': 0.5}
    
    # Portfolio returns will be exactly 0 every day
    # Daily volatilities of components:
    etf1_vol, etf2_vol = comp_returns.std()
    
    portfolio_returns = comp_returns.dot([0.5, 0.5])
    
    engineer = RiskFeatureEngineer(portfolio_returns, comp_returns, weights)
    
    # Since portfolio vol is 0, the method should fallback to returning 1.0
    dr = engineer.compute_diversification_ratio()
    assert dr == 1.0

def test_compute_diversification_ratio_normal():
    # Normal usage
    comp_returns = pd.DataFrame({
        'ETF1': [0.05, 0.02, -0.01],
        'ETF2': [0.02, 0.01, -0.02]
    })
    weights = {'ETF1': 0.7, 'ETF2': 0.3}
    portfolio_returns = comp_returns.dot([0.7, 0.3])
    
    engineer = RiskFeatureEngineer(portfolio_returns, comp_returns, weights)
    dr = engineer.compute_diversification_ratio()
    
    etf1_vol = comp_returns['ETF1'].std()
    etf2_vol = comp_returns['ETF2'].std()
    weighted_vol = (0.7 * etf1_vol) + (0.3 * etf2_vol)
    port_vol = portfolio_returns.std()
    
    expected_dr = weighted_vol / port_vol
    assert np.isclose(dr, expected_dr)

def test_compute_all_features(mock_returns, mock_component_data):
    comp_returns, weights = mock_component_data
    engineer = RiskFeatureEngineer(mock_returns, comp_returns, weights)
    
    features = engineer.compute_all_features()
    
    # Should contain all 11 approved risk features
    expected_keys = {
        "Annualized_Volatility", 
        "Historical_VaR_95", 
        "Maximum_Drawdown", 
        "Diversification_Ratio",
        "Skewness",
        "Kurtosis",
        "RollingVol20",
        "RollingVol60",
        "Sharpe",
        "Sortino",
        "Beta"
    }
    
    assert set(features.keys()) == expected_keys
    for k, v in features.items():
        assert isinstance(v, (float, np.floating))

def test_compute_beta_with_alignment():
    # 25 dates
    dates = pd.date_range("2023-01-01", periods=25, freq="B")
    port_ret = pd.Series(np.linspace(0.01, 0.05, 25), index=dates)
    # Market return perfectly proportional: port = 1.5 * mkt
    mkt_ret = port_ret / 1.5
    
    engineer = RiskFeatureEngineer(portfolio_returns=port_ret, market_returns=mkt_ret)
    beta = engineer.compute_beta(min_periods=20)
    assert np.isclose(beta, 1.5)

def test_compute_beta_missing_market():
    dates = pd.date_range("2023-01-01", periods=25, freq="B")
    port_ret = pd.Series(np.linspace(0.01, 0.05, 25), index=dates)
    # No market returns provided
    engineer = RiskFeatureEngineer(portfolio_returns=port_ret, market_returns=None)
    beta = engineer.compute_beta()
    assert np.isnan(beta)

def test_compute_beta_insufficient_overlap():
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    port_ret = pd.Series(np.linspace(0.01, 0.05, 10), index=dates)
    mkt_ret = port_ret / 1.5
    # Only 10 points when min_periods=20
    engineer = RiskFeatureEngineer(portfolio_returns=port_ret, market_returns=mkt_ret)
    beta = engineer.compute_beta(min_periods=20)
    assert np.isnan(beta)

def test_compute_diversification_ratio_missing_ticker():
    comp_returns = pd.DataFrame({
        'ETF1': [0.05, 0.02, -0.01]
    })
    # Weights has ETF2 which is missing from comp_returns
    weights = {'ETF1': 0.7, 'ETF2': 0.3}
    port_returns = pd.Series([0.05, 0.02, -0.01])
    
    engineer = RiskFeatureEngineer(port_returns, comp_returns, weights)
    dr = engineer.compute_diversification_ratio()
    assert np.isnan(dr)

