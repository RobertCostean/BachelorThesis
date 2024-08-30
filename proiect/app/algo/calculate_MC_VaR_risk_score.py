import yfinance as yf
import numpy as np
from app.data_fetcher.technical import fetch_technical_data, calculate_monte_carlo_var


def get_periodic_returns(technical_data, period=1):
    """
    Calculate returns based on the closing prices for a specified period.

    Parameters:
    - technical_data: DataFrame containing historical stock data
    - period: Number of days for return calculation (1 for daily, 5 for weekly, 22 for monthly, etc.)

    Returns:
    - periodic_returns: Series of returns
    """
    periodic_returns = technical_data['Close'].pct_change(periods=period).dropna()
    return periodic_returns

def var_to_risk_score(var):
    """
    Convert the VaR value to a risk score from 1 to 10.

    Parameters:
    - var: Value at Risk (VaR)

    Returns:
    - risk_score: Risk score from 1 to 10
    """
    if var > -0.02:
        return 1
    elif var > -0.04:
        return 2
    elif var > -0.06:
        return 3
    elif var > -0.08:
        return 4
    elif var > -0.10:
        return 5
    elif var > -0.12:
        return 6
    elif var > -0.14:
        return 7
    elif var > -0.16:
        return 8
    elif var > -0.18:
        return 9
    else:
        return 10


def calculate_mc_var_risk_score(symbol, confidence_level=0.95, period=1):
    """
    Calculate the Monte Carlo VaR and convert it to a risk score.

    Parameters:
    - symbol: Stock ticker symbol
    - confidence_level: Confidence level for VaR calculation
    - period: Number of days for return calculation (1 for daily, 5 for weekly, 22 for monthly, etc.)

    Returns:
    - risk_score: Risk score from 1 to 10
    """
    # Fetch technical data
    technical_data = fetch_technical_data(symbol)

    # Calculate periodic returns
    periodic_returns = get_periodic_returns(technical_data, period)

    # Calculate Monte Carlo VaR
    var = calculate_monte_carlo_var(periodic_returns, confidence_level=confidence_level)

    # Convert VaR to risk score
    risk_score = var_to_risk_score(var)

    return risk_score


if __name__ == "__main__":
    symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB', 'TSLA', 'BRK.B', 'JNJ', 'V', 'WMT',
        'JPM', 'PG', 'NVDA', 'DIS', 'MA', 'HD', 'PYPL', 'UNH', 'VZ', 'ADBE',
        'NFLX', 'INTC', 'CMCSA', 'PFE', 'KO', 'PEP', 'CSCO', 'T', 'MRK', 'XOM',
        'ABT', 'CVX', 'MCD', 'NKE', 'BA', 'ORCL', 'IBM', 'MMM', 'GE', 'ACN',
        'CAT', 'HON', 'RTX', 'MDT', 'LOW', 'SPG', 'TGT', 'LMT', 'NEE', 'QCOM'
    ]

    confidence_level = 0.99
    period = 5  # Monthly returns

    results = []
    for symbol in symbols:
        try:
            risk_score = calculate_mc_var_risk_score(symbol, confidence_level, period)
            results.append((symbol, risk_score))
            print(f"Monte Carlo VaR Risk Score for {symbol}: {risk_score}")
        except Exception as e:
            print(f"Failed to calculate risk score for {symbol}: {e}")