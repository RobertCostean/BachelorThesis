import unittest
from unittest.mock import patch, MagicMock
from app.algo.calculate_financial_risk_score import calculate_financial_risk_score

class TestCalculateFinancialRiskScore(unittest.TestCase):

    @patch('app.algo.calculate_financial_risk_score.fetch_financial_data')
    def test_calculate_financial_risk_score(self, mock_fetch_financial_data):
        # Mock financial data
        mock_financial_data = {
            'market_cap': 1.5e12,  # 1.5 trillion
            'ebitda': 4e10,  # 40 billion
            'dividend_yield': 0.025,  # 2.5%
            'profit_margins': 0.1,  # 10%
            'return_on_equity': 0.18,  # 18%
            'earnings_growth': 0.08,  # 8%
            'debt_to_equity': 0.8  # Ratio of 0.8
        }
        mock_fetch_financial_data.return_value = mock_financial_data

        financial_data = mock_fetch_financial_data('AAPL')
        self.assertIsNotNone(financial_data)

        # Calculate risk score
        risk_score = calculate_financial_risk_score('AAPL')
        self.assertIsNotNone(risk_score)
        self.assertTrue(1 <= risk_score <= 10, "Risk score should be between 1 and 10")
        print(f"Risk Score for AAPL: {risk_score}")

    @patch('app.algo.calculate_financial_risk_score.fetch_financial_data', return_value={})
    def test_calculate_financial_risk_score_no_data(self, mock_fetch_financial_data):
        # Test with no financial data
        risk_score = calculate_financial_risk_score('AAPL')
        self.assertEqual(risk_score, 10, "Risk score should be 10 for no data")

    @patch('app.algo.calculate_financial_risk_score.fetch_financial_data')
    def test_calculate_financial_risk_score_partial_data(self, mock_fetch_financial_data):
        # Test with partial financial data
        partial_financial_data = {
            'market_cap': 1.5e12,  # 1.5 trillion
            'debt_to_equity': 0.8  # Ratio of 0.8
        }
        mock_fetch_financial_data.return_value = partial_financial_data
        risk_score = calculate_financial_risk_score('AAPL')
        self.assertIsNotNone(risk_score)
        self.assertTrue(1 <= risk_score <= 10, "Risk score should be between 1 and 10")

if __name__ == '__main__':
    unittest.main()
