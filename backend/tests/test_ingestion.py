import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.etf_ingestion import ETFDataFetcher
from datetime import datetime
from dateutil.relativedelta import relativedelta

class TestETFDataFetcher(unittest.TestCase):
    def setUp(self):
        self.fetcher = ETFDataFetcher(years=5)
        
    def test_fetch_data(self):
        df = self.fetcher.fetch_data()
        
        # Check shape/columns
        self.assertIsNotNone(df)
        self.assertEqual(len(df.columns), len(self.fetcher.DEFAULT_TICKERS))
        for ticker in self.fetcher.DEFAULT_TICKERS:
            self.assertIn(ticker, df.columns)
        
        # Check no missing values
        self.assertFalse(df.isnull().values.any())
        
        # Check duration (approximate, since trading days are less than calendar days)
        # 5 years is roughly 252 * 5 = 1260 trading days
        self.assertGreater(len(df), 1100)

if __name__ == '__main__':
    unittest.main()
