import unittest
import pandas as pd
from app.data_fetcher.technical import fetch_technical_data


class TestTechnical(unittest.TestCase):

    def test_fetch_technical_data(self):
        symbol = "AAPL"
        data = fetch_technical_data(symbol)

        # Check that data is a DataFrame
        self.assertIsInstance(data, pd.DataFrame)

        # Check that all expected columns are in the DataFrame
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'MACD_Signal']
        for column in expected_columns:
            self.assertIn(column, data.columns)

        # Allow for initial NaN values due to rolling calculations
        # Check for NaN values only after the maximum lookback period (200 for SMA_200)
        max_lookback = 200
        data_after_lookback = data.iloc[max_lookback:]

        self.assertFalse(data_after_lookback.isnull().values.any())


if __name__ == "__main__":
    unittest.main()
