import yfinance as yf


def fetch_financial_data(symbol):
    stock = yf.Ticker(symbol)
    info = stock.info

    financial_data = {
        'market_cap': info.get('marketCap'),
        'ebitda': info.get('ebitda'),
        'dividend_yield': info.get('dividendYield'),
        'profit_margins': info.get('profitMargins'),
        'return_on_equity': info.get('returnOnEquity'),
        'earnings_growth': info.get('earningsGrowth'),
        'debt_to_equity': info.get('debtToEquity'),
        'pe_ratio': info.get('forwardPE'),
        'pb_ratio': info.get('priceToBook')
    }

    return financial_data


if __name__ == "__main__":
    symbol = "AAPL"
    financial_data = fetch_financial_data(symbol)
    print(financial_data)
