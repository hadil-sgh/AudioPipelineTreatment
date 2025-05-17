from typing import Dict, Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

# nltk.download('vader_lexicon')  # Uncomment if running for the first time

class RealTimeSentimentAnalyzer:
    def __init__(self, min_words: int = 6):
        self.min_words = min_words
        self.speaker_buffers = {}  # Dict to store text buffers per speaker
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
    def _is_sentence_complete(self, text: str) -> bool:
        """Check if text ends with sentence-ending punctuation."""
        return bool(re.search(r'[.!?]$', text.strip()))
    
    def _should_analyze(self, text: str) -> bool:
        """Determine if text meets criteria for analysis."""
        words = text.split()
        return len(words) >= self.min_words or self._is_sentence_complete(text)
    
    def analyze(self, text: str) -> Dict:
        """Analyze sentiment of text using VADER."""
        if not text.strip():
            return {
                "sentiment": "NEUTRAL",
                "compound": 0.0
            }
            
        scores = self.sentiment_analyzer.polarity_scores(text)
        compound = scores['compound']
        
        # Determine sentiment label
        if compound >= 0.05:
            sentiment = "POSITIVE"
        elif compound <= -0.05:
            sentiment = "NEGATIVE"
        else:
            sentiment = "NEUTRAL"
            
        return {
            "sentiment": sentiment,
            "compound": compound
        }

    def get_history(self):
        return self.history

    def reset_history(self):
        self.history = []
