import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
try:
    from backend.src.data.etf_ingestion import ETFDataFetcher
except ImportError:
    from src.data.etf_ingestion import ETFDataFetcher
from unittest.mock import patch, MagicMock

@pytest.fixture
def valid_holdings_input():
    return {
        "etf_name": "TEST_ETF",
        "reporting_date": "2023-10-31",
        "holdings": [
            {"ticker": "TCS.NS", "weight": 0.6},
            {"ticker": "INFY.NS", "weight": 0.4}
        ]
    }

@pytest.fixture
def mock_yf_download_success():
    """Mocks yfinance download returning valid DataFrames"""
    dates = pd.date_range(start="2020-01-01", periods=5)
    
    # Create mock close prices
    data = {
        ('Adj Close', 'TCS.NS'): [100, 101, 102, 103, 104],
        ('Adj Close', 'INFY.NS'): [50, 51, 52, 53, 54]
    }
    
    # Create MultiIndex directly mirroring yfinance format
    columns = pd.MultiIndex.from_tuples([
        ('Adj Close', 'TCS.NS'), 
        ('Adj Close', 'INFY.NS')
    ])
    
    df = pd.DataFrame(
        [[100, 50], [101, 51], [102, 52], [103, 53], [104, 54]], 
        index=dates, 
        columns=columns
    )
    
    return df

def test_missing_required_keys():
    fetcher = ETFDataFetcher()
    with pytest.raises(ValueError, match="contain 'etf_name' and 'holdings'"):
        fetcher.fetch_data({"etf_name": "Test"}) # Missing holdings

def test_missing_holding_keys():
    fetcher = ETFDataFetcher()
    invalid_input = {
        "etf_name": "Test",
        "holdings": [{"ticker": "TCS.NS"}] # Missing weight
    }
    with pytest.raises(ValueError, match="'ticker' and 'weight'"):
        fetcher.fetch_data(invalid_input)

def test_empty_pandas_dataframe():
    fetcher = ETFDataFetcher()
    with patch('yfinance.download', return_value=pd.DataFrame()):
        with pytest.raises(ValueError, match="completely empty"):
            fetcher.fetch_data({
                "etf_name": "T", 
                "holdings": [{"ticker": "INVALID", "weight": 1.0}]
            })

@patch('yfinance.download')
def test_successful_data_fetch(mock_download, valid_holdings_input, mock_yf_download_success):
    mock_download.return_value = mock_yf_download_success
    
    fetcher = ETFDataFetcher()
    result = fetcher.fetch_data(valid_holdings_input)
    
    assert result["etf_name"] == "TEST_ETF"
    assert "TCS.NS" in result["weights"]
    assert "INFY.NS" in result["weights"]
    assert result["weights"]["TCS.NS"] == 0.6
    
    price_df = result["price_data"]
    assert isinstance(price_df, pd.DataFrame)
    assert list(price_df.columns) == ["TCS.NS", "INFY.NS"]
    assert len(price_df) == 5
    assert price_df.iloc[0]["TCS.NS"] == 100

@patch('yfinance.download')
def test_missing_ticker_in_response(mock_download, valid_holdings_input):
    # Mock yfinance only returning data for TCS, skipping INFY
    dates = pd.date_range(start="2020-01-01", periods=5)
    df = pd.DataFrame({'Adj Close': [100, 101, 102, 103, 104]}, index=dates)
    df.columns.name = 'Ticker'
    mock_download.return_value = df
    
    fetcher = ETFDataFetcher()
    # Should gracefully drop INFY and re-normalize TCS weight to 1.0
    result = fetcher.fetch_data(valid_holdings_input)
    assert "TCS.NS" in result["weights"]
    assert "INFY.NS" not in result["weights"]
    assert np.isclose(result["weights"]["TCS.NS"], 1.0)
    
    # Should raise ValueError if all tickers fail
    mock_download.return_value = pd.DataFrame()
    with pytest.raises(ValueError, match="Downloaded data is completely empty"):
        fetcher.fetch_data(valid_holdings_input)

@patch('yfinance.download')
def test_no_bfill_and_limited_ffill(mock_download, valid_holdings_input):
    """Verify that leading NaNs (pre-IPO) are preserved and ffill does not exceed 5 days."""
    dates = pd.date_range(start="2020-01-01", periods=10)
    # INFY listed on day 3 (leading NaNs on days 0, 1)
    # TCS has a gap of 6 days (days 2 to 7)
    df = pd.DataFrame({
        ('Adj Close', 'TCS.NS'): [100.0, 101.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 108.0, 109.0],
        ('Adj Close', 'INFY.NS'): [np.nan, np.nan, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0]
    }, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    mock_download.return_value = df

    fetcher = ETFDataFetcher()
    result = fetcher.fetch_data(valid_holdings_input)
    price_df = result["price_data"]

    # Leading NaNs for INFY must NOT be backfilled
    assert pd.isna(price_df.loc[dates[0], "INFY.NS"])
    assert pd.isna(price_df.loc[dates[1], "INFY.NS"])
    assert price_df.loc[dates[2], "INFY.NS"] == 50.0

    # TCS: gap of 6 consecutive NaNs starting at index 2
    # ffill(limit=5) fills up to 5 days, so index 7 remains NaN
    assert price_df.loc[dates[2], "TCS.NS"] == 101.0
    assert price_df.loc[dates[6], "TCS.NS"] == 101.0
    assert pd.isna(price_df.loc[dates[7], "TCS.NS"])

