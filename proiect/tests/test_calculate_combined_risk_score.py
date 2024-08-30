import unittest

from app.algo.calculate_combine_risk_score import normalize_score, calculate_combined_risk_score


class TestCalculateCombinedRiskScore(unittest.TestCase):

    def test_normalize_score(self):
        # Test normalizing within range
        self.assertEqual(normalize_score(5), 5)
        self.assertEqual(normalize_score(10), 10)

        # Test normalizing out of range
        self.assertEqual(normalize_score(-5), 0)
        self.assertEqual(normalize_score(15), 10)

        # Test normalizing with custom max_score
        self.assertEqual(normalize_score(15, max_score=15), 15)
        self.assertEqual(normalize_score(20, max_score=15), 15)

    def test_calculate_combined_risk_score(self):
        # Test with different risk scores
        technical_risk_score = 7.5
        financial_risk_score = 8.0
        sentiment_risk_score = 6.5
        var_risk_score = 5.0
        technical_indicators_risk_score = 7.0

        combined_risk_score = calculate_combined_risk_score(
            technical_risk_score,
            financial_risk_score,
            sentiment_risk_score,
            var_risk_score,
            technical_indicators_risk_score
        )

        # Test if combined score is calculated correctly
        expected_score = (7.5 + 8.0 + 6.5 + 5.0 + 7.0) / 5
        self.assertAlmostEqual(combined_risk_score, expected_score)

        # Test with out of range values
        technical_risk_score = 12.0
        financial_risk_score = -3.0
        sentiment_risk_score = 6.5
        var_risk_score = 5.0
        technical_indicators_risk_score = 7.0

        combined_risk_score = calculate_combined_risk_score(
            technical_risk_score,
            financial_risk_score,
            sentiment_risk_score,
            var_risk_score,
            technical_indicators_risk_score
        )

        # Normalized values: 10, 0, 6.5, 5, 7
        expected_score = (10 + 0 + 6.5 + 5 + 7) / 5
        self.assertAlmostEqual(combined_risk_score, expected_score)


if __name__ == '__main__':
    unittest.main()
