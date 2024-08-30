import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim

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

# Function to prepare the training dataset
def prepare_training_data(scaled_data):
    train_data = scaled_data[:]
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    return x_train, y_train

# Function to train the model
def train_model(model, x_train, y_train, epochs=50, batch_size=64):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()

    for epoch in range(epochs):
        for i in range(0, len(x_train), batch_size):
            x_batch = x_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            x_batch = torch.tensor(x_batch, dtype=torch.float32).to(device)
            y_batch = torch.tensor(y_batch, dtype=torch.float32).to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

# Main Script for Training and Saving Model
stock_list = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'BRK.B', 'JNJ', 'V', 'WMT',
    'JPM', 'PG', 'NVDA', 'DIS', 'MA', 'HD', 'PYPL', 'UNH', 'VZ', 'ADBE',
    'NFLX', 'INTC', 'CMCSA', 'PFE', 'KO', 'PEP', 'CSCO', 'T', 'MRK', 'XOM',
    'ABT', 'CVX', 'MCD', 'NKE', 'BA', 'ORCL', 'IBM', 'MMM', 'GE', 'ACN',
    'CAT', 'HON', 'RTX', 'MDT', 'LOW', 'SPG', 'TGT', 'LMT', 'NEE', 'QCOM',
    'SBUX', 'WBA', 'COST', 'GS', 'MS', 'BAC', 'C', 'BK', 'AMGN', 'GILD',
    'BMY', 'LLY', 'MRNA', 'REGN', 'TMO', 'DHR', 'BDX', 'ISRG', 'ILMN', 'VRTX'
]
start_date = '2010-01-01'

all_x_train = []
all_y_train = []
scaler = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(input_size=1, hidden_layer_size=128, output_size=1).to(device)

# Aggregate data from all symbols
for stock in stock_list:
    try:
        df = fetch_stock_data(stock, start_date)
        if df.empty:
            print(f"No data for {stock}, skipping...")
            continue
        data, dataset = build_dataset(df)
        if scaler is None:
            scaler, scaled_data = scale_data(dataset)
        else:
            scaled_data = scaler.transform(dataset)
        x_train, y_train = prepare_training_data(scaled_data)
        all_x_train.append(x_train)
        all_y_train.append(y_train)
    except Exception as e:
        print(f"An error occurred for {stock}: {e}")
        continue

# Concatenate all training data
all_x_train = np.concatenate(all_x_train, axis=0)
all_y_train = np.concatenate(all_y_train, axis=0)

# Train the model
train_model(model, all_x_train, all_y_train)

# Save the model
torch.save(model.state_dict(), 'lstm_model.pth')

print("Training complete and model saved as 'lstm_model.pth'")