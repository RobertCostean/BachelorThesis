import yfinance as yf
import pandas as pd
import numpy as np

def fetch_technical_data(symbol):
    stock = yf.Ticker(symbol)
    hist = stock.history(period="1y")
    hist['SMA_50'] = hist['Close'].rolling(window=50).mean().fillna(0)
    hist['SMA_200'] = hist['Close'].rolling(window=200).mean().fillna(0)
    hist['RSI'] = calculate_rsi(hist['Close']).fillna(0)
    hist['MACD'], hist['MACD_Signal'] = calculate_macd(hist['Close'])
    hist['MACD'] = hist['MACD'].fillna(0)
    hist['MACD_Signal'] = hist['MACD_Signal'].fillna(0)
    hist['Bollinger_Upper'], hist['Bollinger_Lower'] = calculate_bollinger_bands(hist['Close'])
    hist['Bollinger_Upper'] = hist['Bollinger_Upper'].fillna(0)
    hist['Bollinger_Lower'] = hist['Bollinger_Lower'].fillna(0)
    hist['ATR'] = calculate_atr(hist).fillna(0)
    hist['OBV'] = calculate_obv(hist).fillna(0)
    return hist

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, n_fast=12, n_slow=26, n_signal=9):
    ema_fast = series.ewm(span=n_fast, min_periods=1).mean()
    ema_slow = series.ewm(span=n_slow, min_periods=1).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=n_signal, min_periods=1).mean()
    return macd, macd_signal

def calculate_bollinger_bands(series, window=20, num_std=2):
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def calculate_atr(hist, period=14):
    high_low = hist['High'] - hist['Low']
    high_close = (hist['High'] - hist['Close'].shift()).abs()
    low_close = (hist['Low'] - hist['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=period, min_periods=1).mean()
    return atr

def calculate_obv(hist):
    obv = (np.sign(hist['Close'].diff()) * hist['Volume']).fillna(0).cumsum()
    return obv


def calculate_monte_carlo_var(returns, confidence_level=0.99, num_simulations=10000):
    """
    Calculate the Value at Risk (VaR) using Monte Carlo simulation.

    Parameters:
    - returns: Array-like, historical returns of the portfolio
    - confidence_level: Confidence level for VaR calculation
    - num_simulations: Number of simulations to run

    Returns:
    - VaR: Value at Risk at the specified confidence level
    """
    # Simulate future returns
    mean_return = np.mean(returns)
    std_return = np.std(returns)

    simulated_returns = np.random.normal(mean_return, std_return, num_simulations)

    # Calculate VaR
    var = np.percentile(simulated_returns, (1 - confidence_level) * 100)

    return var

def get_daily_returns(technical_data, period=1):
    """
    Calculate daily returns based on the closing prices.

    Parameters:
    - technical_data: DataFrame containing historical stock data
    - period: Number of days for return calculation (1 for daily, 5 for weekly, etc.)

    Returns:
    - daily_returns: Series of daily returns
    """
    daily_returns = technical_data['Close'].pct_change(periods=period).dropna()
    return daily_returns


if __name__ == "__main__":
    symbol = 'TSLA'
    confidence_level = 0.99  # You can change this to 0.95, 0.99, etc.
    period = 5  # 1 for daily, 5 for weekly, 22 for monthly, etc.

    technical_data = fetch_technical_data(symbol)
    print(technical_data.to_string())

    # Calculate daily returns
    daily_returns = get_daily_returns(technical_data, period)

    # Calculate Monte Carlo VaR
    var = calculate_monte_carlo_var(daily_returns, confidence_level=confidence_level)
    print(f"Monte Carlo VaR for {symbol} at {confidence_level * 100}% confidence level: {var}")
