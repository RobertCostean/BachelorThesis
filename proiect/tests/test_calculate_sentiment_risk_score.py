import unittest
from unittest.mock import patch, MagicMock
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from datetime import datetime, timedelta
from collections import defaultdict, Counter

# Import the functions from the main script
from app.algo.calculate_sentiment_risk_score import fetch_news_rss, analyze_sentiments, calculate_weighted_sentiment_score

class TestSentimentAnalysis(unittest.TestCase):

    @patch('app.algo.calculate_sentiment_risk_score.feedparser.parse')
    def test_fetch_news_rss(self, mock_parse):
        # Mocking the feedparser output
        mock_feed = MagicMock()
        mock_feed.entries = [{
            'title': 'Test Article',
            'summary': 'This is a test summary.',
            'published': (datetime.now() - timedelta(days=1)).strftime('%a, %d %b %Y %H:%M:%S %z'),
            'published_parsed': (datetime.now() - timedelta(days=1)).timetuple()
        }]
        mock_parse.return_value = mock_feed

        articles = fetch_news_rss('AAPL')
        self.assertEqual(len(articles), 1)
        self.assertIn('title', articles[0])
        self.assertIn('summary', articles[0])
        self.assertIn('published', articles[0])

    @patch('app.algo.calculate_sentiment_risk_score.tokenizer')
    @patch('app.algo.calculate_sentiment_risk_score.model')
    def test_analyze_sentiments(self, mock_model, mock_tokenizer):
        # Mocking the tokenizer and model output
        mock_tokenizer.return_value = {'input_ids': torch.tensor([[1, 2, 3]]), 'attention_mask': torch.tensor([[1, 1, 1]])}
        mock_model.return_value = MagicMock(logits=torch.tensor([[0.1, 0.2, 0.7]]))

        articles = [{'summary': 'This is a test summary.', 'published': 'Mon, 01 Jan 2023 00:00:00 +0000'}]
        sentiments = analyze_sentiments(articles)
        self.assertEqual(len(sentiments), 1)
        self.assertIn('sentiment', sentiments[0])
        self.assertEqual(sentiments[0]['sentiment'], 2)

    def test_calculate_weighted_sentiment_score(self):
        articles = [
            {'published_date': datetime.now() - timedelta(days=1), 'sentiment': 2},
            {'published_date': datetime.now() - timedelta(days=1), 'sentiment': 0},
            {'published_date': datetime.now() - timedelta(days=2), 'sentiment': 1}
        ]
        sentiment_score = calculate_weighted_sentiment_score(articles)
        self.assertTrue(0 <= sentiment_score <= 1)

if __name__ == '__main__':
    unittest.main()
