import os
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
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    feeds = feedparser.parse(url)
    articles = []

    # Consider articles from the past 30 days
    past_30_days = datetime.now() - timedelta(days=30)

    for entry in feeds.entries:
        published_date = datetime(*entry.published_parsed[:6])
        if published_date > past_30_days:
            articles.append({'title': entry.title, 'summary': entry.summary, 'published': entry.published})

    return articles

def analyze_sentiments(articles):
    for article in articles:
        inputs = tokenizer(article['summary'], return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        sentiment_class = torch.argmax(outputs.logits, dim=1).item()  # Get the sentiment class (0, 1, or 2)
        article['sentiment'] = sentiment_class  # Assign the sentiment class to the article
    return articles

def aggregate_sentiment_scores(sentiments):
    daily_sentiments = defaultdict(list)
    for article in sentiments:
        date = datetime.strptime(article['published'], '%a, %d %b %Y %H:%M:%S %z').date()
        daily_sentiments[date].append(article['sentiment'])

    # Get the most frequent sentiment for each day
    most_frequent_sentiments = {date: Counter(scores).most_common(1)[0][0] for date, scores in daily_sentiments.items()}
    return most_frequent_sentiments

if __name__ == "__main__":
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'GME', 'AMC', 'TSLA', 'NKLA', 'PLTR', 'RIOT', 'SPCE', 'LKNCY']
    for symbol in symbols:
        articles = fetch_news_rss(symbol)
        sentiments = analyze_sentiments(articles)
        daily_sentiments = aggregate_sentiment_scores(sentiments)
        print(f"For symbol {symbol}: {daily_sentiments}")
        for article in articles:
            print(f"Date: {article['published']}, Sentiment: {article['sentiment']}, Summary: {article['summary']}")
