import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from app.algo.calculate_MC_VaR_risk_score import get_periodic_returns, var_to_risk_score, calculate_mc_var_risk_score
from app.data_fetcher.technical import fetch_technical_data, calculate_monte_carlo_var


class TestCalculateMCVaRRiskScore(unittest.TestCase):

    @patch('app.data_fetcher.technical.fetch_technical_data')
    def test_get_periodic_returns(self, mock_fetch_technical_data):
        # Mocking the technical data
        mock_technical_data = pd.DataFrame({
            'Close': [100, 102, 104, 106, 108, 110]
        }, index=pd.date_range(start='2023-01-01', periods=6, freq='D'))
        mock_fetch_technical_data.return_value = mock_technical_data

        # Fetch technical data
        technical_data = fetch_technical_data('AAPL')
        periodic_returns = get_periodic_returns(technical_data, period=1)

        # Check the length of periodic returns
        self.assertEqual(len(periodic_returns), 251)

        # Check if the returns are within expected range
        self.assertTrue(all(-1 < return_ < 1 for return_ in periodic_returns))

    def test_var_to_risk_score(self):
        test_cases = [
            (-0.01, 1),
            (-0.03, 2),
            (-0.05, 3),
            (-0.07, 4),
            (-0.09, 5),
            (-0.11, 6),
            (-0.13, 7),
            (-0.15, 8),
            (-0.17, 9),
            (-0.19, 10)
        ]
        for var, expected_score in test_cases:
            self.assertEqual(var_to_risk_score(var), expected_score)

    @patch('app.data_fetcher.technical.fetch_technical_data')
    @patch('app.data_fetcher.technical.calculate_monte_carlo_var')
    def test_calculate_mc_var_risk_score(self, mock_calculate_monte_carlo_var, mock_fetch_technical_data):
        # Mocking the technical data
        mock_technical_data = pd.DataFrame({
            'Close': [100, 102, 104, 106, 108, 110]
        }, index=pd.date_range(start='2023-01-01', periods=6, freq='D'))
        mock_fetch_technical_data.return_value = mock_technical_data

        # Mocking the Monte Carlo VaR calculation
        mock_calculate_monte_carlo_var.return_value = -0.03

        # Calculate risk score
        risk_score = calculate_mc_var_risk_score('AAPL', confidence_level=0.95, period=1)
        self.assertEqual(risk_score, 2)

if __name__ == '__main__':
    unittest.main()
