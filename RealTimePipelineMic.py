import asyncio
import logging
import wave
import numpy as np
import signal
import torch
import torchaudio
import time
import re
from collections import deque

from capture.audio_capture import AudioCapture
from transcription.speech_to_text import SpeechToText
from diarization.speaker_diarization import SpeakerDiarization
from sentiment.SentimentFromTrans import RealTimeSentimentAnalyzer
from sentiment.VoiceEmotionRecognizer import VoiceEmotionRecognizer

# ————————————————————————————————————————————————————————————————————————
# 1) Route all logs to file (pipeline.log); only transcripts print to console
# ————————————————————————————————————————————————————————————————————————
LOG_FILE = "pipeline.log"
fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s [%(name)s] %(message)s"))
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger().addHandler(fh)
# remove default console log handlers
for h in list(logging.getLogger().handlers):
    if isinstance(h, logging.StreamHandler):
        logging.getLogger().removeHandler(h)

# ————————————————————————————————————————————————————————————————————————
# 2) Enhanced emotion analysis class with better anger detection
# ————————————————————————————————————————————————————————————————————————
class EmotionAnalyzer:
    def __init__(self):
        self.conversation_history = deque(maxlen=10)
        self.speaker_emotion_history = {}
        
        # Enhanced keyword dictionaries
        self.anger_keywords = {
            'explicit': ['angry', 'mad', 'furious', 'pissed', 'rage', 'livid', 'outraged'],
            'profanity': ['hell', 'damn', 'shit', 'fuck', 'crap', 'wtf', 'goddamn', 'f***', 'f**k'],
            'intensity': ['totally', 'completely', 'absolutely', 'extremely', 'very', 'really'],
            'negative_judgment': ['unacceptable', 'ridiculous', 'pathetic', 'terrible', 'awful', 'worst', 'horrible', 'stupid'],
            'frustration': ['frustrated', 'annoyed', 'irritated', 'fed up', 'sick of', 'tired of'],
            'finality': ['done', 'enough', 'quit', 'over', 'finished', 'through'],
            'complaint': ['problem', 'issue', 'wrong', 'broken', 'failed', 'useless']
        }
        
        self.positive_keywords = {
            'enthusiasm': ['excited', 'amazing', 'awesome', 'fantastic', 'great', 'wonderful', 'excellent'],
            'satisfaction': ['good', 'fine', 'okay', 'thanks', 'appreciate', 'helpful'],
            'politeness': ['please', 'thank you', 'sorry', 'excuse me', 'pardon']
        }
        
    def analyze_emotion(self, text, text_sentiment, voice_emotion, voice_features, voice_score, speaker_id):
        # Get voice intensity
        voice_intensity = self._calculate_voice_intensity(voice_features)
        
        # Enhanced text analysis
        text_analysis = self._analyze_text_content(text)
        
        # Context analysis for escalation
        context_score = self._analyze_context(speaker_id)
        
        # Initialize emotion analysis
        emotion_analysis = {
            'primary_emotion': 'NEUTRAL',
            'confidence': 0.0,
            'reasoning': [],
            'voice_intensity': voice_intensity,
            'text_indicators': text_analysis,
            'escalation_level': context_score
        }
        
        # Enhanced emotion detection logic
        anger_score = self._calculate_anger_score(text_analysis, voice_intensity, text_sentiment, text)
        
        # Anger detection boosters
        if any(prof in text.lower() for prof in self.anger_keywords['profanity']):
            anger_score = min(anger_score + 0.3, 1.0)
            emotion_analysis['reasoning'].append("Profanity detected: +0.3 anger score")
        
        if 'override_reason' in text_sentiment and 'negative' in text_sentiment['override_reason'].lower():
            anger_score = min(anger_score + 0.25, 1.0)
            emotion_analysis['reasoning'].append("Negative override: +0.25 anger score")
        
        if anger_score >= 0.7:
            emotion_analysis['primary_emotion'] = 'ANGRY'
            emotion_analysis['confidence'] = min(anger_score, 0.95)
            emotion_analysis['reasoning'].append(f"High anger score: {anger_score:.2f}")
            
        elif anger_score >= 0.5:
            emotion_analysis['primary_emotion'] = 'FRUSTRATED'
            emotion_analysis['confidence'] = anger_score
            emotion_analysis['reasoning'].append(f"Moderate anger score: {anger_score:.2f}")
            
        elif text_sentiment.get('sentiment') == "NEGATIVE" and voice_intensity > 0.4:
            emotion_analysis['primary_emotion'] = 'IRRITATED'
            emotion_analysis['confidence'] = 0.6
            emotion_analysis['reasoning'].append("Negative text + elevated voice")
            
        elif text_sentiment.get('sentiment') == "NEGATIVE":
            emotion_analysis['primary_emotion'] = 'DISAPPOINTED'
            emotion_analysis['confidence'] = 0.5
            emotion_analysis['reasoning'].append("Negative text + calm voice")
            
        elif text_sentiment.get('sentiment') == "POSITIVE" and voice_intensity > 0.5:
            emotion_analysis['primary_emotion'] = 'ENTHUSIASTIC'
            emotion_analysis['confidence'] = 0.7
            emotion_analysis['reasoning'].append("Positive text + excited voice")
            
        elif text_sentiment.get('sentiment') == "POSITIVE":
            emotion_analysis['primary_emotion'] = 'SATISFIED'
            emotion_analysis['confidence'] = 0.6
            emotion_analysis['reasoning'].append("Positive text + calm voice")
            
        else:  # NEUTRAL text
            if voice_intensity > 0.7:
                emotion_analysis['primary_emotion'] = 'SURPRISED'
                emotion_analysis['confidence'] = 0.5
                emotion_analysis['reasoning'].append("Neutral text + high voice intensity")
            else:
                emotion_analysis['primary_emotion'] = 'NEUTRAL'
                emotion_analysis['confidence'] = 0.4
                emotion_analysis['reasoning'].append("Neutral text + calm voice")
        
        # Apply escalation multiplier
        if context_score > 0.5:
            emotion_analysis['confidence'] = min(emotion_analysis['confidence'] * 1.2, 0.95)
            emotion_analysis['reasoning'].append(f"Escalation detected: {context_score:.2f}")
        
        # Add detailed text indicators
        if text_analysis['categories']:
            categories_str = ', '.join(text_analysis['categories'])
            emotion_analysis['reasoning'].append(f"Text categories: {categories_str}")
        
        # Update conversation history
        self._update_history(speaker_id, text, emotion_analysis)
        
        return emotion_analysis
    
    def _calculate_anger_score(self, text_analysis, voice_intensity, text_sentiment, text):
        """Calculate comprehensive anger score"""
        score = 0.0
        
        # Text sentiment base score
        if text_sentiment.get('sentiment') == "NEGATIVE":
            score += 0.3
        
        # Voice intensity contribution
        score += voice_intensity * 0.3
        
        # Text analysis contributions
        keyword_weights = {
            'explicit': 0.4,
            'profanity': 0.3,
            'negative_judgment': 0.25,
            'frustration': 0.2,
            'finality': 0.15,
            'complaint': 0.1
        }
        
        for category, weight in keyword_weights.items():
            if category in text_analysis['categories']:
                score += weight * min(text_analysis['keyword_density'], 1.0)
        
        # Intensity multiplier
        if 'intensity' in text_analysis['categories']:
            score *= 1.2
        
        # Sentence structure indicators
        if text_analysis['has_questions']:
            score += 0.1
        if text_analysis['has_exclamations']:
            score += 0.1
        if text_analysis['all_caps_ratio'] > 0.3:
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_voice_intensity(self, voice_features):
        """Enhanced voice intensity calculation"""
        pitch = voice_features.get('pitch', 0.0)
        energy = voice_features.get('energy', 0.0)
        rate = voice_features.get('speaking_rate', 0.0)
        
        if pitch == 0.0 and energy == 0.0 and rate == 0.0:
            return 0.0
        
        # Improved normalization
        pitch_norm = 0.0
        if pitch > 0:
            if pitch < 300:
                pitch_norm = 0.0
            elif pitch < 500:
                pitch_norm = (pitch - 300) / 200
            else:
                pitch_norm = 1.0
        
        energy_norm = 0.0
        if energy > 0:
            if energy < 0.15:
                energy_norm = 0.0
            elif energy < 0.25:
                energy_norm = (energy - 0.15) / 0.10
            else:
                energy_norm = 1.0
        
        rate_norm = 0.0
        if rate > 0:
            if rate < 600:
                rate_norm = 0.0
            elif rate < 800:
                rate_norm = (rate - 600) / 200
            else:
                rate_norm = 1.0
        
        intensity = (pitch_norm * 0.25 + energy_norm * 0.5 + rate_norm * 0.25)
        return min(intensity, 1.0)
    
    def _analyze_text_content(self, text):
        """Enhanced text content analysis"""
        text_lower = text.lower()
        text_clean = re.sub(r'[^\w\s]', ' ', text_lower)
        words = text_clean.split()
        
        analysis = {
            'categories': [],
            'keyword_density': 0.0,
            'has_questions': '?' in text,
            'has_exclamations': '!' in text,
            'all_caps_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'word_count': len(words)
        }
        
        total_keywords = 0
        
        # Check anger keywords
        for category, keywords in self.anger_keywords.items():
            found_keywords = [kw for kw in keywords if kw in text_lower]
            if found_keywords:
                analysis['categories'].append(category)
                total_keywords += len(found_keywords)
        
        # Check positive keywords
        for category, keywords in self.positive_keywords.items():
            found_keywords = [kw for kw in keywords if kw in text_lower]
            if found_keywords:
                analysis['categories'].append(category)
        
        # Calculate keyword density
        if len(words) > 0:
            analysis['keyword_density'] = total_keywords / len(words)
        
        return analysis
    
    def _analyze_context(self, speaker_id):
        """Enhanced context analysis with speaker tracking"""
        if len(self.conversation_history) < 2:
            return 0.0
        
        # Get speaker's recent emotional history
        speaker_history = [entry for entry in self.conversation_history 
                          if entry['speaker_id'] == speaker_id]
        
        if len(speaker_history) < 2:
            return 0.0
        
        # Check for escalation patterns
        negative_emotions = ['ANGRY', 'FRUSTRATED', 'IRRITATED', 'DISAPPOINTED']
        recent_emotions = [entry['emotion'] for entry in speaker_history[-3:]]
        
        escalation_score = 0.0
        
        # Count negative emotions in recent history
        negative_count = sum(1 for emotion in recent_emotions if emotion in negative_emotions)
        escalation_score += negative_count / 3.0
        
        # Check for progression toward anger
        if len(recent_emotions) >= 2:
            emotion_intensity = {'NEUTRAL': 0, 'DISAPPOINTED': 1, 'IRRITATED': 2, 'FRUSTRATED': 3, 'ANGRY': 4}
            if recent_emotions[-1] in emotion_intensity and recent_emotions[-2] in emotion_intensity:
                if emotion_intensity[recent_emotions[-1]] > emotion_intensity[recent_emotions[-2]]:
                    escalation_score += 0.3
        
        return min(escalation_score, 1.0)
    
    def _update_history(self, speaker_id, text, emotion_analysis):
        """Update conversation history with speaker tracking"""
        entry = {
            'speaker_id': speaker_id,
            'text': text,
            'emotion': emotion_analysis['primary_emotion'],
            'confidence': emotion_analysis['confidence'],
            'timestamp': time.time()
        }
        
        self.conversation_history.append(entry)
        
        # Update speaker-specific history
        if speaker_id not in self.speaker_emotion_history:
            self.speaker_emotion_history[speaker_id] = deque(maxlen=5)
        
        self.speaker_emotion_history[speaker_id].append(entry)

# ————————————————————————————————————————————————————————————————————————
# 3) Enhanced sentiment analyzer wrapper
# ————————————————————————————————————————————————————————————————————————
class EnhancedSentimentAnalyzer:
    def __init__(self, base_analyzer):
        self.base_analyzer = base_analyzer
        
        # Manual overrides for common misclassifications
        self.negative_phrases = [
            'what the hell', 'totally unacceptable', 'worst', 'pathetic', 'ridiculous',
            'can\'t believe', 'fed up', 'sick of', 'tired of', 'done with',
            'problem with you', 'screw this', 'this sucks', 'hate this'
        ]
        
    def analyze(self, text):
        """Enhanced sentiment analysis with manual overrides"""
        # Get base sentiment
        base_result = self.base_analyzer.analyze(text)
        
        # Check for manual overrides
        text_lower = text.lower()
        
        # Override to NEGATIVE if clearly negative phrases are present
        for phrase in self.negative_phrases:
            if phrase in text_lower:
                return {
                    'sentiment': 'NEGATIVE',
                    'confidence': 0.8,
                    'override_reason': f'Detected negative phrase: "{phrase}"'
                }
        
        # Check for profanity or strong negative words
        profanity_words = ['hell', 'damn', 'shit', 'fuck', 'crap', 'f***', 'f**k']
        if any(word in text_lower for word in profanity_words):
            return {
                'sentiment': 'NEGATIVE',
                'confidence': 0.7,
                'override_reason': 'Profanity detected'
            }
        
        return base_result

# ————————————————————————————————————————————————————————————————————————
# 4) Main pipeline with synchronized voice analysis
# ————————————————————————————————————————————————————————————————————————
async def main():
    # Initialize components
    audio_cap = AudioCapture()
    await audio_cap.start()
    device_sr = audio_cap.sample_rate

    # Resample to 16 kHz
    target_sr = 16_000
    resampler = torchaudio.transforms.Resample(device_sr, target_sr)
    stt = SpeechToText(sample_rate=target_sr)

    # Diarization
    diar = SpeakerDiarization(
        min_speech_duration=0.5,
        threshold=0.03,
        n_speakers=2,
        process_buffer_duration=5.0,
        device="cuda"
    )

    # Initialize enhanced analyzers
    base_sentiment = RealTimeSentimentAnalyzer(min_words=6)
    sentiment_analyzer = EnhancedSentimentAnalyzer(base_sentiment)
    voice_analyzer = VoiceEmotionRecognizer()
    emotion_analyzer = EmotionAnalyzer()

    # Optional: save to WAV
    wav = wave.open("output.wav", "wb")
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(target_sr)

    # Setup Ctrl+C handler
    stop = False
    def on_sigint(sig, frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, on_sigint)

    print("[Enhanced Pipeline] Started — press Ctrl+C to stop")

    # Clock to track elapsed time
    elapsed = 0.0
    chunk_duration = 1.0
    
    # Enhanced audio buffer management
    speaker_audio_buffers = {}
    last_voice_results = {}

    try:
        while not stop:
            chunk = audio_cap.get_audio_chunk(min_samples=target_sr)
            if chunk is None:
                await asyncio.sleep(0.05)
                continue

            # Resample & normalize
            if chunk.dtype != np.float32:
                chunk = chunk.astype(np.float32)
            tensor = torch.from_numpy(chunk).unsqueeze(0)
            audio16 = resampler(tensor).squeeze(0).numpy()
            audio16 /= np.max(np.abs(audio16)) or 1.0

            # Write to WAV
            wav.writeframes((audio16 * 32767).astype(np.int16).tobytes())

            # Feed diarization & STT
            await diar.process_audio(audio16)
            await stt.process_audio(audio16)

            # Fetch new transcriptions & analyze
            new_segs = stt.get_all_transcriptions()
            if new_segs:
                for text in new_segs:
                    spk = diar.get_current_speaker()
                    if spk is None:
                        continue
                        
                    # Initialize speaker buffer if needed
                    if spk not in speaker_audio_buffers:
                        speaker_audio_buffers[spk] = {
                            'buffer': [],
                            'last_analysis_time': 0
                        }
                    
                    # Always accumulate audio
                    speaker_audio_buffers[spk]['buffer'].extend(audio16)
                    
                    # Keep last 2 seconds of audio
                    buffer_length = len(speaker_audio_buffers[spk]['buffer'])
                    if buffer_length > 2 * target_sr:
                        keep_samples = int(0.5 * target_sr)  # Keep 0.5s for continuity
                        speaker_audio_buffers[spk]['buffer'] = speaker_audio_buffers[spk]['buffer'][-keep_samples:]
                    
                    # Prepare analysis buffer (last 2 seconds)
                    analysis_buffer = np.array(speaker_audio_buffers[spk]['buffer'][-2 * target_sr:])
                    
                    # Analyze voice if we have sufficient audio and enough time passed
                    current_time = time.time()
                    if (len(analysis_buffer) >= target_sr and 
                        current_time - speaker_audio_buffers[spk]['last_analysis_time'] > 1.5):
                        
                        voice_result = voice_analyzer.analyze(analysis_buffer)
                        speaker_audio_buffers[spk]['last_analysis_time'] = current_time
                        last_voice_results[spk] = voice_result
                    else:
                        # Reuse last result if available
                        voice_result = last_voice_results.get(spk, {
                            "voice": {
                                "emotion": "NATURAL",
                                "score": 0.5,
                                "features": {"pitch": 0.0, "energy": 0.0, "speaking_rate": 0.0}
                            }
                        })
                    
                    # Enhanced sentiment analysis
                    sentiment_result = sentiment_analyzer.analyze(text)

                    # Enhanced emotion analysis
                    emotion_analysis = emotion_analyzer.analyze_emotion(
                        text=text,
                        text_sentiment=sentiment_result,
                        voice_emotion=voice_result['voice']['emotion'],
                        voice_features=voice_result['voice']['features'],
                        voice_score=voice_result['voice']['score'],
                        speaker_id=spk
                    )

                    # Enhanced output display
                    ts_str = time.strftime('%H:%M:%S', time.gmtime(elapsed))
                    print(f"\n{ts_str} | SPEAKER_{spk:02d} | {text}")
                    
                    # Emotion display with color coding
                    primary_emotion = emotion_analysis['primary_emotion']
                    confidence = emotion_analysis['confidence']
                    
                    # Color coding for different emotions
                    if primary_emotion == 'ANGRY':
                        emoji = '🔥'
                    elif primary_emotion == 'FRUSTRATED':
                        emoji = '😠'
                    elif primary_emotion == 'IRRITATED':
                        emoji = '😤'
                    elif primary_emotion == 'DISAPPOINTED':
                        emoji = '😞'
                    elif primary_emotion == 'ENTHUSIASTIC':
                        emoji = '🎉'
                    elif primary_emotion == 'SATISFIED':
                        emoji = '😊'
                    else:
                        emoji = '😐'
                    
                    print(f"{emoji} [{ts_str}] {primary_emotion} (confidence: {confidence:.2f})")
                    
                    # Show detailed analysis
                    print(f"   Components - Text: {sentiment_result.get('sentiment', 'NEUTRAL')}, "
                          f"Voice: {voice_result['voice']['emotion']}, "
                          f"Intensity: {emotion_analysis['voice_intensity']:.2f}")
                    
                    # Show escalation level
                    if emotion_analysis['escalation_level'] > 0:
                        print(f"   Escalation Level: {emotion_analysis['escalation_level']:.2f}")
                    
                    # Show voice features
                    features = voice_result['voice']['features']
                    print(f"   Voice - Pitch: {features['pitch']:.2f}, "
                          f"Energy: {features['energy']:.2f}, "
                          f"Rate: {features['speaking_rate']:.2f}")
                    
                    # Show text analysis details
                    if emotion_analysis['text_indicators']['categories']:
                        categories = ', '.join(emotion_analysis['text_indicators']['categories'])
                        print(f"   Text Analysis - Categories: {categories}")
                    
                    # Show reasoning
                    if emotion_analysis['reasoning']:
                        print(f"   Reasoning: {'; '.join(emotion_analysis['reasoning'])}")
                    
                    # Show sentiment override if any
                    if 'override_reason' in sentiment_result:
                        print(f"   Sentiment Override: {sentiment_result['override_reason']}")

                elapsed += chunk_duration
                stt.reset()

    finally:
        await audio_cap.stop()
        wav.close()
        if diar.device == "cuda":
            torch.cuda.empty_cache()
        print("[Enhanced Pipeline] Stopped — logs in pipeline.log")

if __name__ == "__main__":
    asyncio.run(main())