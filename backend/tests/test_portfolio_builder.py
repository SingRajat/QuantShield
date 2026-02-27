import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.features.portfolio_builder import PortfolioBuilder

@pytest.fixture
def mock_price_data():
    """Returns a mock DataFrame of prices for 3 ETFs over 4 days."""
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(4)]
    data = {
        'ETF1': [100.0, 102.0, 101.0, 105.0],
        'ETF2': [50.0, 51.0, 50.5, 52.0],
        'ETF3': [200.0, 198.0, 195.0, 201.0]
    }
    return pd.DataFrame(data, index=dates)

def test_portfolio_builder_initialization(mock_price_data):
    builder = PortfolioBuilder(mock_price_data)
    
    assert len(builder.available_tickers) == 3
    # Returns DataFrame should have 3 rows (4 prices -> 3 returns)
    assert len(builder.daily_returns) == 3
    
    # Check specific return calculation (day 2 ETF1: (102-100)/100 = 0.02)
    assert np.isclose(builder.daily_returns['ETF1'].iloc[0], 0.02)

def test_portfolio_builder_empty_data():
    with pytest.raises(ValueError, match="empty"):
        PortfolioBuilder(pd.DataFrame())

def test_build_portfolio_valid_weights(mock_price_data):
    builder = PortfolioBuilder(mock_price_data)
    weights = {'ETF1': 0.5, 'ETF2': 0.5}
    
    result = builder.build_portfolio(weights)
    
    assert 'Daily_Return' in result.columns
    assert 'Cumulative_Return' in result.columns
    assert len(result) == 3
    
    # Manually calculate first day portfolio return:
    # ETF1 return = 0.02, ETF2 return = 0.02 ( (51-50)/50 )
    # Portfolio return = 0.5 * 0.02 + 0.5 * 0.02 = 0.02
    assert np.isclose(result['Daily_Return'].iloc[0], 0.02)

def test_build_portfolio_invalid_weights_sum(mock_price_data):
    builder = PortfolioBuilder(mock_price_data)
    
    with pytest.raises(ValueError, match="sum to 1.0"):
        builder.build_portfolio({'ETF1': 0.5, 'ETF2': 0.4})  # Sums to 0.9

def test_build_portfolio_invalid_ticker(mock_price_data):
    builder = PortfolioBuilder(mock_price_data)
    
    with pytest.raises(ValueError, match="not found"):
        builder.build_portfolio({'ETF1': 0.5, 'INVALID_ETF': 0.5})


