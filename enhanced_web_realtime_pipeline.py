"""
Enhanced Web Real-time Audio Pipeline
Integrates RealTimePipelineMic.py components with WebSocket for frontend communication
"""

import asyncio
import logging
import wave
import numpy as np
import json
import time
import base64
import io
from datetime import datetime
from typing import Dict, Optional, List
import torch
import torchaudio
from collections import deque

# Import components from RealTimePipelineMic
from capture.audio_capture import AudioCapture
from transcription.speech_to_text import SpeechToText
from diarization.speaker_diarization import SpeakerDiarization
from sentiment.SentimentFromTrans import RealTimeSentimentAnalyzer
from sentiment.VoiceEmotionRecognizer import VoiceEmotionRecognizer

# Configure logging for web pipeline
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy and torch types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (torch.Tensor,)):
            return obj.detach().cpu().numpy().tolist()
        elif hasattr(obj, 'item'):  # Handle numpy scalars
            return obj.item()
        elif hasattr(obj, 'tolist'):  # Handle other array-like objects
            return obj.tolist()
        return super().default(obj)

def safe_json_dumps(obj):
    """Safely serialize objects to JSON, handling numpy/torch types"""
    return json.dumps(obj, cls=JSONEncoder)

class EmotionAnalyzer:
    """Enhanced emotion analysis class from RealTimePipelineMic.py"""
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
        """Comprehensive emotion analysis from RealTimePipelineMic.py"""
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
        else:
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
        
        # Update conversation history
        self._update_history(speaker_id, text, emotion_analysis)
        
        return emotion_analysis

    def _calculate_anger_score(self, text_analysis, voice_intensity, text_sentiment, text):
        """Calculate comprehensive anger score"""
        score = 0.0
        
        # Text sentiment base score
        if text_sentiment.get('sentiment') == "NEGATIVE":
            score += 0.4
        
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
                score += weight
        
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
        import re
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


class EnhancedSentimentAnalyzer:
    """Enhanced sentiment analyzer wrapper from RealTimePipelineMic.py"""
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


class EnhancedWebRealTimePipeline:
    """Main pipeline class that bridges RealTimePipelineMic.py with WebSocket"""
    
    def __init__(self):
        self.is_running = False
        self.session_id = None
        self.websocket = None
        
        # Audio processing components
        self.stt = None
        self.diarization = None
        self.sentiment_analyzer = None
        self.voice_analyzer = None
        self.emotion_analyzer = None
        self.resampler = None
        
        # Audio buffers and processing
        self.audio_buffer = deque(maxlen=1000)
        self.speaker_audio_buffers = {}
        self.last_voice_results = {}
        
        # Target sample rate
        self.target_sr = 16000
        
        # Audio recording for quality monitoring (DISABLED)
        self.recording_file = None
        self.recording_enabled = False
        
        # Audio quality monitoring
        self.audio_stats = {
            "chunks_processed": 0,
            "chunks_with_speech": 0,
            "total_audio_duration": 0.0,
            "speech_duration": 0.0,
            "avg_volume": 0.0,
            "peak_volume": 0.0
        }

    async def initialize_components(self):
        """Initialize all audio processing components"""
        try:
            logger.info("Initializing enhanced audio processing components...")
            
            # Initialize resampler (frontend now sends 16kHz to match audio_capture.py)
            device_sr = 16000  # Frontend now sends 16kHz directly
            if device_sr != self.target_sr:
                self.resampler = torchaudio.transforms.Resample(device_sr, self.target_sr)
            else:
                self.resampler = None  # No resampling needed
            
            # Initialize Speech-to-Text with more sensitive settings
            self.stt = SpeechToText(sample_rate=self.target_sr)
            
            # Initialize Speaker Diarization with optimized settings
            self.diarization = SpeakerDiarization(
                min_speech_duration=0.3,  # Reduced for faster detection
                threshold=0.02,  # Lower threshold for better sensitivity
                n_speakers=2,
                process_buffer_duration=2.0,  # Shorter buffer for faster response
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # Initialize Sentiment Analysis
            base_sentiment = RealTimeSentimentAnalyzer(min_words=2)
            self.sentiment_analyzer = EnhancedSentimentAnalyzer(base_sentiment)
            
            # Initialize Voice Emotion Recognition
            self.voice_analyzer = VoiceEmotionRecognizer()
            
            # Initialize Enhanced Emotion Analyzer
            self.emotion_analyzer = EmotionAnalyzer()
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            return False

    def set_websocket(self, websocket):
        """Set the WebSocket connection for sending results"""
        self.websocket = websocket

    async def start_session(self, session_id: str):
        """Start a new audio processing session"""
        if self.is_running:
            logger.warning("Session already running")
            return False
            
        self.session_id = session_id
        
        # Initialize components if not already done
        if not self.stt:
            if not await self.initialize_components():
                return False
        
        # Start audio recording for quality monitoring
        if self.recording_enabled:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            recording_filename = f"recording_{session_id}_{timestamp}.wav"
            self.recording_file = wave.open(recording_filename, "wb")
            self.recording_file.setnchannels(1)
            self.recording_file.setsampwidth(2)
            self.recording_file.setframerate(self.target_sr)
            logger.info(f"Started recording to: {recording_filename}")
        
        self.is_running = True
        logger.info(f"Started session: {session_id}")
        
        # Send session started confirmation
        if self.websocket:
            await self.websocket.send_text(safe_json_dumps({
                "type": "session_started",
                "sessionId": session_id,
                "timestamp": datetime.now().isoformat()
            }))
        
        return True

    async def stop_session(self):
        """Stop the current audio processing session"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Close recording file
        if self.recording_file:
            self.recording_file.close()
            self.recording_file = None
            logger.info("Recording saved")
        
        # Clear buffers
        self.audio_buffer.clear()
        self.speaker_audio_buffers.clear()
        self.last_voice_results.clear()
        
        # Reset STT
        if self.stt:
            self.stt.reset()
        
        # Clean up GPU memory if using CUDA
        if self.diarization and hasattr(self.diarization, 'device') and self.diarization.device == "cuda":
            torch.cuda.empty_cache()
        
        logger.info(f"Stopped session: {self.session_id}")
        
        # Send session stopped confirmation
        if self.websocket:
            await self.websocket.send_text(safe_json_dumps({
                "type": "session_stopped",
                "sessionId": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "stats": self.audio_stats
            }))

    def convert_webm_to_numpy(self, audio_bytes: bytes) -> np.ndarray:
        """Convert WebM audio bytes to numpy array"""
        try:
            # Try to decode as raw PCM first
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Convert to float32 and normalize
            audio_data = audio_data.astype(np.float32) / 32768.0
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Error converting audio data: {e}")
            return np.array([])

    def preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Enhanced audio preprocessing for better speech recognition"""
        try:
            # Ensure float32
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Remove DC offset
            audio_data = audio_data - np.mean(audio_data)
            
            # Normalize to prevent clipping but preserve dynamics
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                # Use a softer normalization that preserves quiet speech
                audio_data = audio_data / max_val * 0.8
            
            # Apply gentle noise gate to remove very quiet background noise
            noise_floor = np.percentile(np.abs(audio_data), 10)  # 10th percentile as noise floor
            gate_threshold = noise_floor * 3  # Gentle gate threshold
            
            # Apply gate with smooth transitions
            mask = np.abs(audio_data) > gate_threshold
            audio_data = audio_data * mask
            
            # Final level check and soft limiting
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = np.tanh(audio_data)  # Soft limiting
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Error in audio preprocessing: {e}")
            return audio_data

    async def process_audio_chunk(self, audio_bytes: bytes) -> Optional[Dict]:
        """Process audio chunk using the enhanced pipeline"""
        if not self.is_running or not all([self.stt, self.diarization, self.sentiment_analyzer, self.voice_analyzer]):
            return None
        
        try:
            # Convert audio data
            audio_np = self.convert_webm_to_numpy(audio_bytes)
            
            if len(audio_np) == 0:
                return None
            
            # Skip very small chunks
            if len(audio_np) < 1000:
                return None
            
            # Resample to target sample rate
            if self.resampler:
                tensor = torch.from_numpy(audio_np).unsqueeze(0)
                resampled = self.resampler(tensor).squeeze(0)
                audio_np = resampled.numpy()
            
            # Preprocess audio
            audio_np = self.preprocess_audio(audio_np)
            
            # Update audio statistics  
            self.audio_stats["chunks_processed"] += 1
            self.audio_stats["total_audio_duration"] += float(len(audio_np) / self.target_sr)
            volume = float(np.sqrt(np.mean(audio_np**2)))
            self.audio_stats["avg_volume"] = float((self.audio_stats["avg_volume"] + volume) / 2)
            self.audio_stats["peak_volume"] = float(max(self.audio_stats["peak_volume"], np.max(np.abs(audio_np))))
            
            # Record to file for quality monitoring
            if self.recording_file:
                audio_int16 = (audio_np * 32767).astype(np.int16)
                self.recording_file.writeframes(audio_int16.tobytes())
            
            # Add to buffer for processing
            self.audio_buffer.append(audio_np)
            
            # Process accumulated audio (combine recent chunks for better context)
            if len(self.audio_buffer) >= 3:  # Process every 3 chunks
                combined_audio = np.concatenate(list(self.audio_buffer)[-5:])  # Use last 5 chunks
                
                # Feed to diarization and STT
                await self.diarization.process_audio(combined_audio)
                await self.stt.process_audio(combined_audio)
                
                # Get NEW transcriptions only (not accumulated)
                new_transcription = self.stt.get_latest_transcription()
                
                if new_transcription and len(new_transcription.strip()) >= 3:
                    text = new_transcription.strip()  # Use the new transcription text
                    
                    # Get current speaker
                    speaker = self.diarization.get_current_speaker()
                    if speaker is None:
                        speaker = 0
                    
                    speaker_id = f"SPEAKER_{speaker:02d}"
                    
                    # Initialize speaker buffer if needed
                    if speaker not in self.speaker_audio_buffers:
                        self.speaker_audio_buffers[speaker] = {
                            'buffer': [],
                            'last_analysis_time': 0
                        }
                    
                    # Accumulate audio for voice emotion analysis
                    self.speaker_audio_buffers[speaker]['buffer'].extend(combined_audio)
                    
                    # Keep last 2 seconds of audio
                    buffer_length = len(self.speaker_audio_buffers[speaker]['buffer'])
                    if buffer_length > 2 * self.target_sr:
                        keep_samples = int(0.5 * self.target_sr)  # Keep 0.5s for continuity
                        self.speaker_audio_buffers[speaker]['buffer'] = self.speaker_audio_buffers[speaker]['buffer'][-keep_samples:]
                    
                    # Prepare analysis buffer (last 2 seconds)
                    analysis_buffer = np.array(self.speaker_audio_buffers[speaker]['buffer'][-2 * self.target_sr:])
                    
                    # Analyze voice if we have sufficient audio and enough time passed
                    current_time = time.time()
                    if (len(analysis_buffer) >= self.target_sr and 
                        current_time - self.speaker_audio_buffers[speaker]['last_analysis_time'] > 1.5):
                        
                        voice_result = self.voice_analyzer.analyze(analysis_buffer)
                        self.speaker_audio_buffers[speaker]['last_analysis_time'] = current_time
                        self.last_voice_results[speaker] = voice_result
                    else:
                        # Reuse last result if available
                        voice_result = self.last_voice_results.get(speaker, {
                            "voice": {
                                "emotion": "NEUTRAL",
                                "score": 0.5,
                                "features": {"pitch": 0.0, "energy": 0.0, "speaking_rate": 0.0}
                            }
                        })
                    
                    # Enhanced sentiment analysis
                    sentiment_result = self.sentiment_analyzer.analyze(text)
                    
                    # Enhanced emotion analysis
                    emotion_analysis = self.emotion_analyzer.analyze_emotion(
                        text=text,
                        text_sentiment=sentiment_result,
                        voice_emotion=voice_result['voice']['emotion'],
                        voice_features=voice_result['voice']['features'],
                        voice_score=voice_result['voice']['score'],
                        speaker_id=speaker
                    )
                    
                    # Create result for frontend
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    
                    result = {
                        "timestamp": timestamp,
                        "speaker": speaker_id,
                        "text": text,
                        "sentiment": {
                            "text": sentiment_result.get("sentiment", "NEUTRAL"),
                            "voice": emotion_analysis['primary_emotion'],
                                "score": emotion_analysis['confidence']
                            },
                        "voiceFeatures": voice_result['voice']['features'],
                        "sessionId": self.session_id,
                        "analysisType": "real-time",
                        "emotionAnalysis": emotion_analysis,
                        "audioStats": self.audio_stats
                    }
                    
                    # Mark that this chunk had speech
                    self.audio_stats["chunks_with_speech"] += 1
                    self.audio_stats["speech_duration"] += float(len(combined_audio) / self.target_sr)
                    
                    # Reset STT for next chunk to avoid accumulation
                    self.stt.reset()
                    
                    # Also clear any accumulated audio buffer to prevent overlap
                    self.audio_buffer.clear()
                    
                    return result
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return {
                "type": "error",
                "message": f"Audio processing error: {str(e)}",
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "sessionId": self.session_id
            }

    async def send_result(self, result: Dict):
        """Send analysis result to frontend via WebSocket"""
        if self.websocket and result:
            try:
                await self.websocket.send_text(safe_json_dumps(result))
            except Exception as e:
                logger.error(f"Error sending result to WebSocket: {e}")

    def get_session_info(self) -> Dict:
        """Get current session information"""
        return {
            "session_id": self.session_id,
            "is_running": self.is_running,
            "components_initialized": all([self.stt, self.diarization, self.sentiment_analyzer, self.voice_analyzer]),
            "audio_stats": self.audio_stats,
            "timestamp": datetime.now().isoformat()
        }


# Global pipeline instance
global_pipeline = EnhancedWebRealTimePipeline()


async def get_pipeline() -> EnhancedWebRealTimePipeline:
    """Get the global pipeline instance"""
    return global_pipeline