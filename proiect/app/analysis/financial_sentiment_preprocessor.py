import pandas as pd
import numpy as np
from app.data_fetcher.financials import fetch_financial_data
from app.data_fetcher.news import fetch_news_rss, analyze_sentiments
from app.data_fetcher.technical import fetch_technical_data

def preprocess_financial_sentiment_data(symbol):
    financial_data = fetch_financial_data(symbol)
    articles = fetch_news_rss(symbol)
    analyzed_articles = analyze_sentiments(articles)
    sentiment_scores = [article['sentiment'] for article in analyzed_articles]
    avg_sentiment_score = aggregate_sentiment_scores(sentiment_scores)

    financial_data['average_sentiment_score'] = avg_sentiment_score
    return financial_data


def preprocess_and_combine_data(symbol):
    financial_sentiment_data = preprocess_financial_sentiment_data(symbol)
    financial_sentiment_df = pd.DataFrame([financial_sentiment_data])
    technical_data = fetch_technical_data(symbol)

    # Ensure no NaNs in technical data
    technical_data = technical_data.dropna()

    combined_df = pd.concat([financial_sentiment_df] * len(technical_data), ignore_index=True)
    combined_df = pd.concat([combined_df, technical_data.reset_index(drop=True)], axis=1)

    # Ensure no NaNs in combined data
    combined_df = combined_df.dropna()

    return combined_df

def define_risk_events(data):
    data['price_change'] = data['Close'].pct_change()
    data['risk_event'] = np.where(data['price_change'] < -0.03, 1, 0)  # Adjust threshold to -0.03

    if data['risk_event'].sum() == 0:
        print("Warning: No risk events found in the data.")
        data['risk_event'] = np.where(data['price_change'] < -0.01, 1, 0)  # Further adjust threshold to -0.01

    return data

def aggregate_sentiment_scores(sentiments):
    if not sentiments:
        return 0
    return sum(sentiments) / len(sentiments)
