from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# nltk.download('vader_lexicon')  # Uncomment if running for the first time

class RealTimeSentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.history = []

    def analyze(self, text):
        try:
            scores = self.sia.polarity_scores(text)
            sentiment = self._categorize(scores['compound'])
            result = {
                'text': text,
                'compound': scores['compound'],
                'neg': scores['neg'],
                'neu': scores['neu'],
                'pos': scores['pos'],
                'sentiment': sentiment
            }
            self.history.append(result)
            return result
        except Exception as e:
            print(f"[Sentiment Error] {e}")
            return {
                'text': text,
                'compound': None,
                'neg': None,
                'neu': None,
                'pos': None,
                'sentiment': 'Error'
            }

    def _categorize(self, compound):
        if compound is None:
            return 'Unknown'
        if compound >= 0.05:
            return 'Positive'
        elif compound <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    def get_history(self):
        return self.history

    def reset_history(self):
        self.history = []
