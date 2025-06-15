from typing import Dict, Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import logging

logger = logging.getLogger(__name__)

class TranscriptSentiment:
    def __init__(self, min_words: int = 6):
        self.min_words = min_words
        self.speaker_buffers = {}  
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
    def _is_sentence_end(self, text: str) -> bool:
       
        return bool(re.search(r'[.!?]\s*$', text))
    
    def analyze(self, speaker_id: int, text: str) -> dict:
        
        if not text.strip():
            return None
            
        # Initialize buffer for new speakers
        if speaker_id not in self.speaker_buffers:
            self.speaker_buffers[speaker_id] = ""
            
        # Append new text to buffer
        self.speaker_buffers[speaker_id] += " " + text
        buf = self.speaker_buffers[speaker_id].strip()
        
        # Check if we have enough words or a complete sentence
        word_count = len(buf.split())
        if word_count < self.min_words and not self._is_sentence_end(buf):
            return None
            
        try:
            # Analyze sentiment
            scores = self.sentiment_analyzer.polarity_scores(buf)
            compound = scores['compound']
            
            # Determine sentiment category
            if compound >= 0.05:
                sentiment = "POSITIVE"
            elif compound <= -0.05:
                sentiment = "NEGATIVE"
            else:
                sentiment = "NEUTRAL"
                
            # Clear buffer after analysis
            self.speaker_buffers[speaker_id] = ""
            
            return {
                "sentiment": sentiment,
                "compound": compound,
                "scores": scores,
                "text": buf
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return None
            
    def reset(self):
        """Reset all speaker buffers"""
        self.speaker_buffers = {}

    def get_history(self):
        return self.history

    def reset_history(self):
        self.history = []
