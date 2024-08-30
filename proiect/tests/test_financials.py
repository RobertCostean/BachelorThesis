import unittest
from app.data_fetcher.financials import fetch_financial_data


class TestEconomics(unittest.TestCase):

    def test_fetch_financial_data(self):
        symbol = "AAPL"
        data = fetch_financial_data(symbol)

        # Check that data is a dictionary
        self.assertIsInstance(data, dict)

        # Check that all expected keys are in the dictionary
        expected_keys = [
            'market_cap', 'ebitda', 'dividend_yield', 'profit_margins', 'return_on_equity', 'earnings_growth', 'debt_to_equity'
        ]
        for key in expected_keys:
            self.assertIn(key, data)

        # Check that the values are not None
        for key in expected_keys:
            self.assertIsNotNone(data[key])


if __name__ == "__main__":
    unittest.main()
