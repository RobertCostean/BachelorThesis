import yfinance as yf
import numpy as np

def fetch_market_data():
    # Fetch data for major indices and VIX
    sp500 = yf.Ticker('^GSPC')
    nasdaq = yf.Ticker('^IXIC')
    dowjones = yf.Ticker('^DJI')
    vix = yf.Ticker('^VIX')
    gold = yf.Ticker('GC=F')
    oil = yf.Ticker('CL=F')

    sp500_hist = sp500.history(period='1y')
    nasdaq_hist = nasdaq.history(period='1y')
    dowjones_hist = dowjones.history(period='1y')
    vix_hist = vix.history(period='1y')
    gold_hist = gold.history(period='1y')
    oil_hist = oil.history(period='1y')

    return sp500_hist, nasdaq_hist, dowjones_hist, vix_hist, gold_hist, oil_hist

def calculate_volatility(index_hist):
    log_returns = np.log(index_hist['Close'] / index_hist['Close'].shift(1))
    volatility = log_returns.std() * np.sqrt(252)  # Annualize the volatility
    return volatility

def calculate_vix_risk(vix_hist):
    avg_vix = vix_hist['Close'].mean()
    current_vix = vix_hist['Close'][-1]

    vix_risk = np.clip((current_vix - avg_vix) / avg_vix * 10, 1, 10)
    return vix_risk

def calculate_market_risk(sp500_hist, nasdaq_hist, dowjones_hist, vix_hist, gold_hist, oil_hist):
    sp500_volatility = calculate_volatility(sp500_hist)
    nasdaq_volatility = calculate_volatility(nasdaq_hist)
    dowjones_volatility = calculate_volatility(dowjones_hist)
    gold_volatility = calculate_volatility(gold_hist)
    oil_volatility = calculate_volatility(oil_hist)
    vix_risk = calculate_vix_risk(vix_hist)

    avg_volatility = (sp500_volatility + nasdaq_volatility + dowjones_volatility + gold_volatility + oil_volatility) / 5

    market_risk = np.clip(avg_volatility * 40, 1, 10)  # Scale to 1-10 range

    combined_risk = (market_risk + vix_risk) / 2
    return combined_risk

if __name__ == "__main__":
    sp500_hist, nasdaq_hist, dowjones_hist, vix_hist, gold_hist, oil_hist = fetch_market_data()
    market_risk_score = calculate_market_risk(sp500_hist, nasdaq_hist, dowjones_hist, vix_hist, gold_hist, oil_hist)
    print(f"Market Risk Score: {market_risk_score}")
