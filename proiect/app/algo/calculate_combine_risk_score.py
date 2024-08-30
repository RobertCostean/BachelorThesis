def normalize_score(score, max_score=10):
    """Normalize the score to be between 0 and 10."""
    return max(min(score, max_score), 0)

def calculate_combined_risk_score(technical_risk_score, financial_risk_score, sentiment_risk_score, var_risk_score, technical_indicators_risk_score):
    """
    Combine the technical, financial, and sentiment risk scores into a single risk score.

    Args:
        technical_risk_score (float): The risk score derived from technical analysis.
        financial_risk_score (float): The risk score derived from financial analysis.
        sentiment_risk_score (float): The risk score derived from sentiment analysis.

    Returns:
        float: The combined risk score.
    """
    # Normalize the individual scores
    technical_risk_score = normalize_score(technical_risk_score)
    financial_risk_score = normalize_score(financial_risk_score)
    sentiment_risk_score = normalize_score(sentiment_risk_score)
    var_risk_score = normalize_score(var_risk_score)
    technical_indicators_risk_score = normalize_score(technical_indicators_risk_score)


    # Combine the scores with equal weights
    combined_risk_score = (technical_risk_score + financial_risk_score + sentiment_risk_score + var_risk_score + technical_indicators_risk_score) / 5
    return combined_risk_score

def main():
    # Example risk scores
    technical_risk_score = 7.5
    financial_risk_score = 8.0
    sentiment_risk_score = 6.5
    var_risk_score = 5.0
    technical_indicators_risk_score = 7.0

    # Calculate the combined risk score
    combined_risk_score = calculate_combined_risk_score(technical_risk_score, financial_risk_score, sentiment_risk_score, var_risk_score, technical_indicators_risk_score)

    print(f"Technical Risk Score: {technical_risk_score}")
    print(f"Financial Risk Score: {financial_risk_score}")
    print(f"Sentiment Risk Score: {sentiment_risk_score}")
    print(f"VAR Risk Score: {var_risk_score}")
    print(f"Technical Indicators Risk Score: {technical_indicators_risk_score}")
    print(f"Combined Risk Score: {combined_risk_score:.2f}")

if __name__ == "__main__":
    main()