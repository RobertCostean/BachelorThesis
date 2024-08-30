import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from datetime import datetime
import torch
import torch.nn as nn


# Define the LSTM model in PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1])
        return predictions


# Function to fetch stock data
def fetch_stock_data(symbol, start_date):
    stock_data = yf.download(symbol, start=start_date, end=datetime.now())
    return stock_data


# Function to build the dataset
def build_dataset(stock_data):
    data = stock_data.filter(items=['Close'])
    dataset = data.values
    return data, dataset


# Function to scale the dataset
def scale_data(dataset):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    return scaler, scaled_data


# Function to prepare the testing dataset
def prepare_testing_data(scaled_data, training_data_len):
    test_data = scaled_data[training_data_len - 60:, :]
    if len(test_data) < 60:
        return np.empty((0, 60, 1))
    x_test = []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])
    x_test = np.array(x_test)
    if x_test.shape[0] == 0:
        return np.empty((0, 60, 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return x_test


# Function to make predictions
def make_predictions(model, x_test, scaler, device):
    if len(x_test) == 0:
        return np.empty((0, 1))
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        predictions = model(x_test).cpu().numpy()
    predictions = scaler.inverse_transform(predictions)
    return predictions


# Updated function to calculate risk score
def calculate_technical_risk_score(symbol):
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(input_size=1, hidden_layer_size=128, output_size=1).to(device)
    model.load_state_dict(torch.load('app/algo/lstm_model.pth', map_location=device))

    # Fetch and prepare data
    start_date = '2010-01-01'
    df = fetch_stock_data(symbol, start_date)
    if df.empty:
        print(f"No data for {symbol}")
        return 1  # No data implies minimal risk

    data, dataset = build_dataset(df)
    training_data_len = int(np.ceil(len(dataset) * .95))

    scaler, scaled_data = scale_data(dataset)
    x_test = prepare_testing_data(scaled_data, training_data_len)
    predictions = make_predictions(model, x_test, scaler, device)

    if len(predictions) == 0:
        return 1  # No data implies minimal risk

    predictions = np.array(predictions)
    if predictions.ndim == 0 or predictions.size == 0:
        return 1  # No data implies minimal risk

    try:
        returns = np.diff(predictions.squeeze()) / predictions[:-1].squeeze()
    except ValueError as e:
        print(f"Error calculating returns: {e}")
        return 1  # Handle case where predictions cannot be used

    negative_returns = returns[returns < 0]
    positive_returns = returns[returns > 0]

    negative_volatility = np.std(negative_returns) if len(negative_returns) > 0 else 0
    positive_volatility = np.std(positive_returns) if len(positive_returns) > 0 else 0

    # Weight negative volatility more heavily
    weighted_volatility = (2 * negative_volatility + positive_volatility) / 3

    # Debug: print intermediate values
    print(f"Negative Volatility: {negative_volatility}")
    print(f"Positive Volatility: {positive_volatility}")
    print(f"Weighted Volatility: {weighted_volatility}")

    # Normalize risk score to a range of 1 to 10
    min_std = 0.005
    max_std = 0.03
    normalized_risk_score = 1 + 9 * (weighted_volatility - min_std) / (max_std - min_std)
    normalized_risk_score = np.clip(normalized_risk_score, 1, 10)
    return normalized_risk_score


# Main function to get forecast and risk score for a user-specified symbol
def main():
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'GME', 'JNJ', 'V', 'WMT']

    for symbol in symbols:
        risk_score = calculate_technical_risk_score(symbol)
        print(f'Risk score for {symbol}: {risk_score:.2f}')


if __name__ == "__main__":
    main()
