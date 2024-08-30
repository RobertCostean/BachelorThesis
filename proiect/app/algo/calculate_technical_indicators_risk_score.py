import numpy as np
import pandas as pd

from app.data_fetcher.technical import fetch_technical_data

def calculate_technical_indicators_risk_score(symbol):
    """
    Calculate a risk score based on various technical indicators.

    Parameters:
    - symbol: The stock symbol for which to fetch and analyze technical data.

    Returns:
    - risk_score: A composite score representing the risk level based on the technical indicators.
    """
    technical_data = fetch_technical_data(symbol)

    # Ensure the DataFrame contains the necessary columns
    required_columns = ['Close', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'Bollinger_Upper', 'Bollinger_Lower', 'ATR', 'OBV', 'Volume']
    for col in required_columns:
        if col not in technical_data.columns:
            raise ValueError(f"Column '{col}' is missing from the technical data.")

    # Initialize risk components
    sma_risk = 0
    rsi_risk = 0
    macd_risk = 0
    bollinger_risk = 0
    atr_risk = 0
    obv_risk = 0

    # Calculate risk for each technical indicator
    if not technical_data.empty:
        # SMA Risk
        sma_risk = abs((technical_data['Close'] / technical_data['SMA_50']).iloc[-1] - 1) * 20
        sma_risk = min(sma_risk, 10)  # Cap at 10

        # RSI Risk
        rsi_value = technical_data['RSI'].iloc[-1]
        if rsi_value > 70 or rsi_value < 30:
            rsi_risk = 2
        else:
            rsi_risk = abs(rsi_value - 50) / 25

        # MACD Risk
        macd_value = technical_data['MACD'].iloc[-1]
        macd_signal_value = technical_data['MACD_Signal'].iloc[-1]
        macd_risk = abs(macd_value - macd_signal_value) / (abs(macd_signal_value) + 0.01) * 20
        macd_risk = min(macd_risk, 10)  # Cap at 10

        # Bollinger Bands Risk
        bollinger_upper = technical_data['Bollinger_Upper'].iloc[-1]
        bollinger_lower = technical_data['Bollinger_Lower'].iloc[-1]
        bollinger_risk = abs((technical_data['Close'].iloc[-1] - bollinger_lower) / (bollinger_upper - bollinger_lower) - 0.5) * 4
        bollinger_risk = min(bollinger_risk, 10)  # Cap at 10

        # ATR Risk
        atr_value = technical_data['ATR'].iloc[-1]
        atr_risk = (atr_value / technical_data['Close'].iloc[-1]) * 20
        atr_risk = min(atr_risk, 10)  # Cap at 10

        # OBV Risk
        obv_value = technical_data['OBV'].iloc[-1]
        volume_value = technical_data['Volume'].iloc[-1]
        if volume_value != 0:
            obv_risk = abs(obv_value / volume_value) * 20
        obv_risk = min(obv_risk, 10)  # Cap at 10

    # Combine all risk components
    risk_score = (sma_risk + rsi_risk + macd_risk + bollinger_risk + atr_risk + obv_risk) / 6

    # Cap the risk score to be within the range of 1 to 10
    risk_score = max(1, min(risk_score, 10))

    return risk_score

if __name__ == "__main__":
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'GME', 'JNJ', 'V', 'WMT']

    for symbol in symbols:
        risk_score = calculate_technical_indicators_risk_score(symbol)
        print(f"Risk Score for {symbol}: {risk_score:.2f}")
