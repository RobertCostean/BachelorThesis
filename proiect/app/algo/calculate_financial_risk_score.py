from app.data_fetcher.financials import fetch_financial_data

def calculate_financial_risk_score(symbol):
    financial_data = fetch_financial_data(symbol)
    if not financial_data:
        return 10  # High risk if no financial data is available

    # Updated realistic baseline values for normalization
    baselines = {
        'market_cap': 2e12,  # 2 trillion
        'ebitda': 5e10,  # 50 billion
        'dividend_yield': 0.03,  # 3%
        'profit_margins': 0.15,  # 15%
        'return_on_equity': 0.2,  # 20%
        'earnings_growth': 0.1,  # 10%
        'debt_to_equity': 1.0  # Ratio of 1
    }

    # Updated weightings
    weights = {
        'market_cap': -0.2,  # Large market cap reduces risk
        'ebitda': -0.1,  # Higher EBITDA reduces risk
        'dividend_yield': -0.1,  # Higher dividend yield reduces risk
        'profit_margins': -0.2,  # Higher profit margins reduce risk
        'return_on_equity': -0.2,  # Higher return on equity reduces risk
        'earnings_growth': -0.1,  # Higher earnings growth reduces risk
        'debt_to_equity': 0.3  # Higher debt to equity increases risk
    }

    normalized_scores = {}
    for metric, weight in weights.items():
        value = financial_data.get(metric)
        if value is not None and value != 0:
            # Normalize the value using the baseline
            normalized_value = value / baselines[metric]
            # Cap normalized values to avoid extreme outliers
            normalized_value = min(max(normalized_value, -1), 1)
            # Apply the weight to the normalized value
            normalized_scores[metric] = weight * normalized_value
        else:
            normalized_scores[metric] = 0

    total_score = sum(normalized_scores.values())

    # Ensure the total score is within the range of -1 to 1
    total_score = max(min(total_score, 1), -1)

    # Convert the total score to a 1-10 scale
    financial_risk_score = (total_score + 1) * 5  # Map [-1, 1] to [0, 10]

    financial_risk_score = max(1, min(financial_risk_score, 10))  # Ensure score is within 1-10 range

    return financial_risk_score


if __name__ == "__main__":
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'GME', 'AMC', 'TSLA', 'NKLA', 'PLTR', 'RIOT', 'SPCE', 'LKNCY']
    for symbol in symbols:
        risk_score = calculate_financial_risk_score(symbol)
        print(f"Financial Risk Score for {symbol}: {risk_score}")
