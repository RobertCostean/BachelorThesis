import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import torch

# Import the functions from the main script
from app.algo.calculate_technical_risk_score import fetch_stock_data, build_dataset, scale_data, prepare_testing_data, \
    LSTMModel, make_predictions, calculate_technical_risk_score


class TestRiskScoreCalculation(unittest.TestCase):

    @patch('app.algo.calculate_technical_risk_score.yf.download')
    def test_fetch_stock_data(self, mock_download):
        # Mocking the yfinance download output
        mock_download.return_value = pd.DataFrame({'Close': [1, 2, 3, 4, 5]})
        stock_data = fetch_stock_data('AAPL', '2010-01-01')
        self.assertIsNotNone(stock_data)

    def test_build_dataset(self):
        mock_stock_data = pd.DataFrame({'Close': [1, 2, 3, 4, 5]})
        data, dataset = build_dataset(mock_stock_data)
        self.assertIsNotNone(data)
        self.assertEqual(len(dataset), 5)

    def test_scale_data(self):
        dataset = np.array([[1], [2], [3], [4], [5]])
        scaler, scaled_data = scale_data(dataset)
        self.assertIsNotNone(scaler)
        self.assertEqual(scaled_data.shape, dataset.shape)

    def test_prepare_testing_data(self):
        scaled_data = np.array([[0], [0.25], [0.5], [0.75], [1]])
        training_data_len = 3
        x_test = prepare_testing_data(scaled_data, training_data_len)
        if x_test.shape[0] > 0:
            self.assertEqual(x_test.shape[1], 60)

    @patch('app.algo.calculate_technical_risk_score.LSTMModel.forward')
    def test_make_predictions(self, mock_forward):
        mock_forward.return_value = torch.tensor([[1.0], [1.1], [1.2]])
        device = torch.device("cpu")
        model = LSTMModel(input_size=1, hidden_layer_size=128, output_size=1).to(device)
        x_test = np.random.rand(10, 60, 1)
        scaler = MagicMock()
        scaler.inverse_transform.return_value = np.array([[100], [101], [102]])
        predictions = make_predictions(model, x_test, scaler, device)
        self.assertEqual(predictions.shape, (3, 1))

    @patch('app.algo.calculate_technical_risk_score.make_predictions')
    @patch('app.algo.calculate_technical_risk_score.prepare_testing_data')
    @patch('app.algo.calculate_technical_risk_score.scale_data')
    @patch('app.algo.calculate_technical_risk_score.build_dataset')
    @patch('app.algo.calculate_technical_risk_score.fetch_stock_data')
    def test_calculate_technical_risk_score(self, mock_fetch_stock_data, mock_build_dataset, mock_scale_data, mock_prepare_testing_data, mock_make_predictions):
        # Mocking the dependencies
        mock_fetch_stock_data.return_value = pd.DataFrame({'Close': [1, 2, 3, 4, 5]})
        mock_build_dataset.return_value = (None, np.array([[1], [2], [3], [4], [5]]))
        mock_scale_data.return_value = (MagicMock(), np.array([[0], [0.25], [0.5], [0.75], [1]]))
        mock_prepare_testing_data.return_value = np.random.rand(10, 60, 1)
        mock_make_predictions.return_value = np.array([[100], [101], [102], [103], [104]])

        risk_score = calculate_technical_risk_score('AAPL')
        self.assertTrue(1 <= risk_score <= 10, "Risk score should be between 1 and 10")


if __name__ == '__main__':
    unittest.main()
