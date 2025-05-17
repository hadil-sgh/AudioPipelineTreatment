import time
from sentiment.SentimentFromTrans import RealTimeSentimentAnalyzer
# Simulated list of transcribed sentences (like real-time output from Whisper)
simulated_transcription = [
    "I love how this works!",
    "The system is a bit slow sometimes.",
    "This is terrible. I hate it.",
    "Hmm, I'm not sure how I feel about this.",
    "Great job! Everything is running perfectly.",
    "Why does it keep crashing?",
    "Okay, now it's working again.",
    "This is the best feature so far.",
    "I'm so frustrated with this bug.",
    "Let's finish this today."
]

def test_realtime_sentiment():
    analyzer = RealTimeSentimentAnalyzer()

    print("🔍 Starting real-time sentiment test...\n")
    for text in simulated_transcription:
        result = analyzer.analyze(text)
        print(f"🗣️ Text: {result['text']}")
        print(f"📊 Sentiment: {result['sentiment']} (compound={result['compound']:.3f})\n")
        time.sleep(1)  # Simulates delay between real-time utterances

    print("✅ Done. Full history:\n")
    for entry in analyzer.get_history():
        print(entry)

if __name__ == "__main__":
    test_realtime_sentiment()
