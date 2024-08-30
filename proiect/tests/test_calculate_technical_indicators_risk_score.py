import unittest
from unittest.mock import patch
import pandas as pd
from app.algo.calculate_technical_indicators_risk_score import calculate_technical_indicators_risk_score

class TestCalculateRiskScore(unittest.TestCase):
    def setUp(self):
        """
        Set up a mock DataFrame containing technical indicators for testing.
        """
        self.data = {
            'Close': [150, 152, 151, 153, 154],
            'SMA_50': [148, 149, 150, 151, 152],
            'RSI': [75, 35, 50, 65, 25],
            'MACD': [2, 1, 0, -1, -2],
            'MACD_Signal': [1, 1, 0, -1, -1],
            'Bollinger_Upper': [155, 156, 157, 158, 159],
            'Bollinger_Lower': [145, 146, 147, 148, 149],
            'ATR': [2, 2, 2, 2, 2],
            'OBV': [1000, 2000, 3000, 4000, 5000],
            'Volume': [1000, 2000, 3000, 4000, 5000]
        }
        self.mock_data = pd.DataFrame(self.data)

    @patch('app.algo.calculate_technical_indicators_risk_score.fetch_technical_data')
    def test_calculate_risk_score(self, mock_fetch_technical_data):
        """
        Test the calculate_risk_score function with mock data.
        """
        mock_fetch_technical_data.return_value = self.mock_data
        risk_score = calculate_technical_indicators_risk_score('AAPL')
        self.assertGreaterEqual(risk_score, 0)
        self.assertLessEqual(risk_score, 10)

if __name__ == "__main__":
    unittest.main()
