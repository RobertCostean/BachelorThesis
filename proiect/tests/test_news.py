import unittest
import datetime
from collections import defaultdict

# Sample data from news.py
news_data = [
    {"date": "Tue, 18 Jun 2024 18:55:46 +0000", "sentiment": 2,
     "summary": "Nvidia has overtaken Microsoft and Apple to become the world’s most valuable company for the first time."},
    {"date": "Tue, 18 Jun 2024 18:25:14 +0000", "sentiment": 0,
     "summary": "One of America's largest technology-focused ETFs will likely be forced to buy billions of dollars worth of Nvidia stock when it rebalances Friday, a byproduct of both the chip giant’s meteoric rise and arcane fund diversification rules."},
    # Add all other news items here...
]


# Function to parse date and aggregate sentiment
def process_news_data(news_data):
    aggregated_data = defaultdict(int)
    for item in news_data:
        try:
            # Parse date
            date = datetime.datetime.strptime(item["date"], "%a, %d %b %Y %H:%M:%S %z").date()

            # Validate sentiment
            sentiment = int(item["sentiment"])
            if sentiment not in [0, 1, 2]:  # Assuming sentiment can only be 0, 1, or 2
                raise ValueError(f"Invalid sentiment value: {sentiment}")

            # Aggregate sentiment by date
            aggregated_data[date] += sentiment
        except ValueError as e:
            print(f"Error processing news item: {e}")
            continue
    return dict(aggregated_data)


# Unit test for the process_news_data function
class TestNewsAggregation(unittest.TestCase):

    def test_process_news_data(self):
        # Call the function with the sample data
        aggregated_news_data = process_news_data(news_data)

        # Define the expected output
        expected_output = {datetime.date(2024, 6, 18): 2}

        # Assert the function output matches the expected output
        self.assertEqual(aggregated_news_data, expected_output,
                         f"Expected {expected_output}, but got {aggregated_news_data}")


if __name__ == '__main__':
    unittest.main()