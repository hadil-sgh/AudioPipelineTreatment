# pipeline.py
import asyncio
import numpy as np
from datetime import datetime
from capture.audio_capture import AudioCapture
from diarization.speaker_diarization import SpeakerDiarization
from transcription.speech_to_text import SpeechToText
from sentiment.VoiceEmotionRecognizer import SentimentAnalyzer
import logging
import json
from typing import Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioPipeline:
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        n_speakers: int = 2,
        device_index: Optional[int] = None
    ):
        # Console colors
        self.COLOR_HEADER = '\033[95m'
        self.COLOR_OKBLUE = '\033[94m'
        self.COLOR_OKGREEN = '\033[92m'
        self.COLOR_WARNING = '\033[93m'
        self.COLOR_FAIL = '\033[91m'
        self.COLOR_ENDC = '\033[0m'

        # Initialize components with consistent parameters
        self.audio_cap = AudioCapture(
            sample_rate=sample_rate,
            chunk_size=chunk_size,
            device_index=device_index
        )
        self.diarization = SpeakerDiarization(
            sample_rate=sample_rate,
            n_speakers=n_speakers
        )
        self.stt = SpeechToText(
            sample_rate=sample_rate,
            buffer_size=chunk_size
        )
        self.sentiment = SentimentAnalyzer()
        
        self.is_running = False
        self.last_output: Optional[Dict] = None

    def print_header(self, message):
        print(f"\n{self.COLOR_HEADER}=== {message} ==={self.COLOR_ENDC}")

    def print_success(self, message):
        print(f"{self.COLOR_OKGREEN}✓ {message}{self.COLOR_ENDC}")

    def print_warning(self, message):
        print(f"{self.COLOR_WARNING}⚠️ {message}{self.COLOR_ENDC}")

    def print_error(self, message):
        print(f"{self.COLOR_FAIL}✗ {message}{self.COLOR_ENDC}")

    async def start(self):
        """Start the audio processing pipeline"""
        if self.is_running:
            self.print_warning("Pipeline already running")
            return

        self.print_header("Starting Audio Pipeline")
        
        try:
            # Start audio capture
            await self.audio_cap.start()
            self.print_success("Audio capture initialized")
            
            # Initialize components
            self.diarization = SpeakerDiarization()
            self.stt = SpeechToText()
            self.sentiment = SentimentAnalyzer()
            
            self.print_success("Processing components loaded")
            self.print_header("Begin Real-Time Processing")

            self.is_running = True
            while self.is_running:
                await self.process_audio()
                await asyncio.sleep(0.01)  # Reduced sleep time for better responsiveness

        except Exception as e:
            self.print_error(f"Pipeline error: {str(e)}")
            logger.exception("Pipeline error")
        finally:
            await self.stop()

    async def stop(self):
        """Stop the pipeline"""
        if not self.is_running:
            return

        self.print_header("Stopping Pipeline")
        try:
            await self.audio_cap.stop()
            self.is_running = False
            self.print_success("Pipeline stopped cleanly")
        except Exception as e:
            self.print_error(f"Error during shutdown: {str(e)}")
            logger.exception("Shutdown error")

    async def process_audio(self):
        """Process audio data through the pipeline"""
        try:
            # Get audio chunk (min 1 second of audio)
            chunk = self.audio_cap.get_audio_chunk()
            
            if chunk is None:
                return
                
            self.print_debug(f"Processing chunk: {len(chunk)} samples")

            # Speaker diarization
            await self.diarization.process_audio(chunk)
            speaker = self.diarization.get_current_speaker()
            self.print_debug(f"Speaker detected: {speaker}")

            # Speech to text
            await self.stt.process_audio(chunk)
            text = self.stt.get_latest_transcription()
            
            if not text:
                return
                
            self.print_debug(f"Transcription: {text}")

            # Sentiment analysis
            sentiment = self.sentiment.analyze(text, chunk)
            self.print_debug(f"Sentiment: {sentiment}")

            # Format output
            timestamp = datetime.now().strftime("%H:%M:%S")
            output = {
                "timestamp": timestamp,
                "speaker": f"SPEAKER_{speaker:02d}" if speaker is not None else "UNKNOWN",
                "text": text,
                "sentiment": sentiment
            }

            # Only print if output changed
            if output != self.last_output:
                self.print_output(output)
                self.last_output = output

        except Exception as e:
            self.print_error(f"Processing error: {str(e)}")
            logger.exception("Processing error")

    def print_debug(self, message):
        """Debug-level logging"""
        logger.debug(f"{self.COLOR_OKBLUE}[DEBUG] {message}{self.COLOR_ENDC}")

    def print_output(self, output):
        """Formatted output display"""
        print(f"\n{self.COLOR_OKGREEN}▶ {output['timestamp']} {output['speaker']}")
        print(f"📝 {output['text']}")
        print(f"😃 {output['sentiment']['text']['emotion']} "
              f"(Confidence: {output['sentiment']['text']['score']:.2f})")
        print(f"🎤 {output['sentiment']['voice']['emotion']} "
              f"(Pitch: {output['sentiment']['voice']['features']['pitch']:.1f}Hz){self.COLOR_ENDC}")

if __name__ == "__main__":
    # List available audio devices
    devices = AudioCapture().list_devices()
    print("\nAvailable audio devices:")
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:  # Only show input devices
            print(f"{i}: {device['name']}")
    
    # Create and run pipeline
    pipeline = AudioPipeline()
    
    try:
        asyncio.run(pipeline.start())
    except KeyboardInterrupt:
        print("\n")
        pipeline.print_header("Keyboard Interrupt Received")
        asyncio.run(pipeline.stop())