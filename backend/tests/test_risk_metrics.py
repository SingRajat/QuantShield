import pytest
import pandas as pd
import numpy as np
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
    
    # Should only contain the 4 permitted features
    expected_keys = {
        "Annualized_Volatility", 
        "Historical_VaR_95", 
        "Maximum_Drawdown", 
        "Diversification_Ratio"
    }
    
    assert set(features.keys()) == expected_keys
    for k, v in features.items():
        assert isinstance(v, float)
