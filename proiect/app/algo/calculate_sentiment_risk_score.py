import os
import urllib.parse

import feedparser
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from datetime import datetime, timedelta
from collections import defaultdict, Counter

# Define the path to the model file
model_path = os.path.join(os.path.dirname(__file__), '../models/bert_sentiment_model.pth')

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

def fetch_news_rss(ticker):
    base_url = "https://feeds.finance.yahoo.com/rss/2.0/headline"
    query_params = {
        's': ticker,
        'region': 'US',
        'lang': 'en-US'
    }
    query_string = urllib.parse.urlencode(query_params)
    url = f"{base_url}?{query_string}"
    feeds = feedparser.parse(url)
    articles = []

    # Consider articles from the past 30 days
    past_30_days = datetime.now() - timedelta(days=30)

    for entry in feeds.entries:
        published_date = entry.get('published_parsed')
        if published_date:
            published_date = datetime(*published_date[:6])
            if published_date > past_30_days:
                articles.append({
                    'title': entry.get('title', ''),
                    'summary': entry.get('summary', ''),
                    'published': entry.get('published', ''),
                    'published_date': published_date
                })

    return articles

def analyze_sentiments(articles):
    for article in articles:
        inputs = tokenizer(article['summary'], return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        sentiment_class = torch.argmax(outputs.logits, dim=1).item()  # Get the sentiment class (0, 1, or 2)
        article['sentiment'] = sentiment_class  # Assign the sentiment class to the article
    return articles

def calculate_weighted_sentiment_score(sentiments):
    weight_decay = 0.9  # Weight decay factor for older news
    sentiment_score = 0.0
    total_weight = 0.0

    # Calculate weighted sentiment scores based on the age of the news
    for article in sentiments:
        days_old = (datetime.now() - article['published_date']).days
        weeks_old = days_old // 7
        weight = weight_decay ** weeks_old

        if article['sentiment'] == 2:  # Positive sentiment
            sentiment_score += weight
        elif article['sentiment'] == 0:  # Negative sentiment
            sentiment_score -= weight

        total_weight += weight

    # Normalize the sentiment score to be between 0 and 1
    if total_weight != 0:
        sentiment_score = (sentiment_score / total_weight + 1) / 2

    return sentiment_score

def calculate_sentiment_risk_score(symbol):
    articles = fetch_news_rss(symbol)
    sentiments = analyze_sentiments(articles)
    sentiment_score = calculate_weighted_sentiment_score(sentiments)
    risk_score = 1 - sentiment_score  # Inverse of sentiment score for risk
    return risk_score * 10

if __name__ == "__main__":
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'GME', 'AMC', 'TSLA', 'NKLA', 'PLTR', 'RIOT', 'SPCE', 'LKNCY']
    for symbol in symbols:
        risk_score = calculate_sentiment_risk_score(symbol)
        print(f"Sentiment Risk Score for {symbol}: {risk_score:.2f}")
