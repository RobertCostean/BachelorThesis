import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from app.algo.calculate_market_risk import fetch_market_data, calculate_volatility, calculate_vix_risk, calculate_market_risk

class TestCalculateMarketRisk(unittest.TestCase):

    @patch('app.algo.calculate_market_risk.yf.Ticker')
    def test_fetch_market_data(self, mock_ticker):
        # Mock the history method for each ticker
        mock_ticker.return_value.history.return_value = pd.DataFrame({
            'Close': np.random.rand(252) * 100  # Simulate 252 days of random closing prices
        })

        # Fetch market data
        sp500_hist, nasdaq_hist, dowjones_hist, vix_hist, gold_hist, oil_hist = fetch_market_data()

        # Assert that data is fetched for each ticker
        self.assertEqual(len(sp500_hist), 252)
        self.assertEqual(len(nasdaq_hist), 252)
        self.assertEqual(len(dowjones_hist), 252)
        self.assertEqual(len(vix_hist), 252)
        self.assertEqual(len(gold_hist), 252)
        self.assertEqual(len(oil_hist), 252)

    def test_calculate_volatility(self):
        # Create a DataFrame with simulated closing prices
        index_hist = pd.DataFrame({
            'Close': np.random.rand(252) * 100  # Simulate 252 days of random closing prices
        })
        volatility = calculate_volatility(index_hist)

        # Assert that volatility is a positive number
        self.assertTrue(volatility > 0)



    @patch('app.algo.calculate_market_risk.calculate_volatility')
    @patch('app.algo.calculate_market_risk.calculate_vix_risk')
    def test_calculate_market_risk(self, mock_calculate_vix_risk, mock_calculate_volatility):
        # Mock the volatility calculations
        mock_calculate_volatility.return_value = 0.2
        mock_calculate_vix_risk.return_value = 5

        # Create DataFrames with simulated data for each index
        index_hist = pd.DataFrame({
            'Close': np.random.rand(252) * 100  # Simulate 252 days of random closing prices
        })

        # Calculate market risk
        market_risk_score = calculate_market_risk(index_hist, index_hist, index_hist, index_hist, index_hist, index_hist)

        # Assert that market risk score is between 1 and 10
        self.assertTrue(1 <= market_risk_score <= 10)

if __name__ == '__main__':
    unittest.main()
