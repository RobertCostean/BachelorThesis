from collections import defaultdict
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.utils import resample
from app.data_fetcher.news import fetch_news_rss, analyze_sentiments, aggregate_sentiment_scores
from app.data_fetcher.technical import fetch_technical_data


def preprocess_technical_data(symbol):
    technical_data = fetch_technical_data(symbol)
    technical_data = technical_data.dropna()
    return technical_data


def preprocess_sentiment_data(symbol):
    articles = fetch_news_rss(symbol)
    sentiments = analyze_sentiments(articles)
    daily_sentiments = defaultdict(list)

    for article in sentiments:
        date = datetime.strptime(article['published'], '%a, %d %b %Y %H:%M:%S %z').date()
        daily_sentiments[date].append(article['sentiment'])

    # Aggregate daily sentiment scores
    daily_avg_sentiments = {date: sum(scores)/len(scores) for date, scores in daily_sentiments.items()}
    return daily_avg_sentiments

if __name__ == "__main__":
    symbol = "AAPL"
    sentiment_data = preprocess_sentiment_data(symbol)
    print(sentiment_data)


def define_risk_events(data):
    data['price_change'] = data['Close'].pct_change()
    risk_thresholds = [-0.02, -0.015, -0.01, -0.005]

    for threshold in risk_thresholds:
        data['risk_event'] = np.where(data['price_change'] < threshold, 1, 0)
        if data['risk_event'].sum() > 0:
            break
    if data['risk_event'].sum() == 0:
        print(f"Warning: No risk events found in the data with thresholds {risk_thresholds}")
        data['risk_event'] = np.where(data['price_change'] < -0.005, 1, 0)

    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    return data


def balance_dataset(X, y):
    dataset = pd.concat([X, y], axis=1)
    majority_class = dataset[dataset['risk_event'] == 0]
    minority_class = dataset[dataset['risk_event'] == 1]

    if len(minority_class) == 0:
        raise ValueError("No risk events found in the data. Adjust the risk event threshold.")

    minority_class_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=42)
    upsampled_dataset = pd.concat([majority_class, minority_class_upsampled])
    upsampled_dataset = upsampled_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    return upsampled_dataset.drop(columns=['risk_event']), upsampled_dataset['risk_event']
