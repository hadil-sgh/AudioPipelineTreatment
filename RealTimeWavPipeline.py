import asyncio
import time
import numpy as np
import librosa

from diarization.speaker_diarization import SpeakerDiarization
from transcription.speech_to_text import SpeechToText

class RealTimeWavPipeline:
    def __init__(
        self,
        wav_path: str,
        sample_rate: int = 16000,
        chunk_duration: float = 0.15,  # Further reduced for more precise processing
        log_path: str = "transcription.log"
    ):
        self.wav_path = wav_path
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.log_path = log_path

        # Queues for pipelines
        self.diar_queue = asyncio.Queue(maxsize=100)  # Added maxsize
        self.stt_queue = asyncio.Queue(maxsize=100)   # Added maxsize
        self.diar_results = asyncio.Queue(maxsize=100)  # Added maxsize
        self.stt_results = asyncio.Queue(maxsize=100)   # Added maxsize

        # Components
        self.diar = SpeakerDiarization(device="cuda", process_buffer_duration=0.15)
        self.stt = SpeechToText(device="cuda", sample_rate=sample_rate)

        # Open log file
        self.log_file = open(self.log_path, 'w', encoding='utf-8')
        self.log_file.write("Timestamp | Speaker | Transcription\n")
        self.log_file.flush()

        # State to avoid repeats
        self.last_speaker = None
        self.last_text = ""
        self.text_buffer = []
        self.max_buffer_size = 3
        
        # Synchronization
        self.producer_done = asyncio.Event()
        self.diar_done = asyncio.Event()
        self.stt_done = asyncio.Event()

    def load_wav_chunks(self):
        try:
            audio, sr = librosa.load(self.wav_path, sr=self.sample_rate, mono=True)
            total_samples = len(audio)
            chunk_size = int(self.sample_rate * self.chunk_duration)
            timestamp = 0.0
            
            print(f"Loading audio file: {self.wav_path}")
            print(f"Total duration: {total_samples/sr:.2f}s")
            print(f"Chunk size: {chunk_size/sr:.2f}s")
            
            for start in range(0, total_samples, chunk_size):
                end = min(start + chunk_size, total_samples)
                chunk = audio[start:end]
                if len(chunk) > 0:  # Only yield non-empty chunks
                    yield chunk.astype(np.float32), timestamp
                timestamp += self.chunk_duration
        except Exception as e:
            print(f"Error loading audio file: {e}")
            raise

    async def producer(self):
        try:
            chunk_count = 0
            for audio, ts in self.load_wav_chunks():
                chunk_count += 1
                packet = {"audio": audio, "timestamp": ts}
                
                # Wait if queues are full
                while self.diar_queue.full() or self.stt_queue.full():
                    await asyncio.sleep(0.1)
                
                await self.diar_queue.put(packet)
                await self.stt_queue.put(packet)
                await asyncio.sleep(self.chunk_duration)
                
                if chunk_count % 10 == 0:
                    print(f"Processed {chunk_count} chunks...")
            
            print("Finished producing all chunks")
            self.producer_done.set()
            await self.diar_queue.put(None)
            await self.stt_queue.put(None)
        except Exception as e:
            print(f"Error in producer: {e}")
            self.producer_done.set()

    async def diar_consumer(self):
        try:
            while not self.producer_done.is_set() or not self.diar_queue.empty():
                try:
                    packet = await asyncio.wait_for(self.diar_queue.get(), timeout=2.0)
                    if packet is None:
                        print("Diarization consumer received end signal")
                        await self.diar.finalize_processing()
                        break
                    
                    await self.diar.process_audio(packet['audio'])
                    speaker = self.diar.get_current_speaker()
                    ts = packet['timestamp']
                    
                    if speaker is not None:
                        while self.diar_results.full():
                            await asyncio.sleep(0.1)
                        await self.diar_results.put({'start': ts, 'speaker': speaker})
                        
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    print(f"Error processing diarization chunk: {e}")
            
            self.diar_done.set()
        except Exception as e:
            print(f"Error in diarization consumer: {e}")
            self.diar_done.set()

    async def stt_consumer(self):
        try:
            while not self.producer_done.is_set() or not self.stt_queue.empty():
                try:
                    packet = await asyncio.wait_for(self.stt_queue.get(), timeout=2.0)
                    if packet is None:
                        print("STT consumer received end signal")
                        break
                    
                    await self.stt.process_audio(packet['audio'])
                    text = self.stt.get_latest_transcription()
                    ts = packet['timestamp']
                    
                    if text and text.strip():
                        while self.stt_results.full():
                            await asyncio.sleep(0.1)
                        await self.stt_results.put({'start': ts, 'text': text})
                        
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    print(f"Error processing STT chunk: {e}")
            
            self.stt_done.set()
        except Exception as e:
            print(f"Error in STT consumer: {e}")
            self.stt_done.set()

    async def aligner(self):
        buffer = []
        max_buffer_size = 5
        last_activity = time.time()
        processed_chunks = 0
        last_ts = 0
        
        try:
            while not (self.diar_done.is_set() and self.stt_done.is_set()) or not (self.diar_results.empty() and self.stt_results.empty()):
                if time.time() - last_activity > 15.0:
                    print("Warning: No activity in aligner for 15 seconds")
                    break
                
                try:
                    diar_evt = await asyncio.wait_for(self.diar_results.get(), timeout=1.0)
                    last_activity = time.time()
                    processed_chunks += 1
                    
                    ts = diar_evt['start']
                    if ts - last_ts > 2.0:
                        print(f"Time gap detected: {ts - last_ts:.2f}s")
                    last_ts = ts
                    
                    speaker = diar_evt['speaker']
                    
                    # Get best matching text event
                    text = None
                    best_match_diff = float('inf')
                    best_match = None
                    
                    # Process all available text events
                    while not self.stt_results.empty():
                        try:
                            stt_evt = await asyncio.wait_for(self.stt_results.get(), timeout=0.1)
                            time_diff = abs(stt_evt['start'] - ts)
                            
                            if time_diff < self.chunk_duration * 2:
                                if time_diff < best_match_diff:
                                    best_match_diff = time_diff
                                    best_match = stt_evt
                            else:
                                buffer.append(stt_evt)
                        except asyncio.TimeoutError:
                            break
                    
                    # Put back unmatched events
                    for evt in buffer:
                        await self.stt_results.put(evt)
                    buffer = buffer[-max_buffer_size:]
                    
                    if best_match is not None:
                        text = best_match['text']
                    
                    if text is None:
                        continue
                    
                    # Clean up text
                    text = text.strip()
                    if not text:
                        continue
                    
                    # Handle text buffering and speaker changes
                    if speaker != self.last_speaker:
                        # Write any buffered text for previous speaker
                        if self.text_buffer:
                            buffered_text = " ".join(self.text_buffer)
                            if buffered_text.strip():
                                ts_str = time.strftime('%H:%M:%S', time.gmtime(ts - self.chunk_duration))
                                line = f"{ts_str} | SPEAKER_{self.last_speaker:02d} | {buffered_text}\n"
                                print(line, end='')
                                self.log_file.write(line)
                                self.log_file.flush()
                            self.text_buffer = []
                        
                        # Start new buffer for new speaker
                        self.text_buffer = [text]
                        self.last_speaker = speaker
                    else:
                        # Add to current speaker's buffer
                        if text not in self.text_buffer:
                            self.text_buffer.append(text)
                            if len(self.text_buffer) > self.max_buffer_size:
                                self.text_buffer.pop(0)
                    
                    # Write if buffer is full
                    if len(self.text_buffer) >= self.max_buffer_size:
                        buffered_text = " ".join(self.text_buffer)
                        if buffered_text.strip():
                            ts_str = time.strftime('%H:%M:%S', time.gmtime(ts))
                            line = f"{ts_str} | SPEAKER_{speaker:02d} | {buffered_text}\n"
                            print(line, end='')
                            self.log_file.write(line)
                            self.log_file.flush()
                        self.text_buffer = []
                    
                    if processed_chunks % 5 == 0:
                        print(f"Processed {processed_chunks} chunks in aligner...")
                        
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    print(f"Error in aligner processing: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error in aligner: {e}")
        finally:
            # Write any remaining buffered text
            if self.text_buffer:
                buffered_text = " ".join(self.text_buffer)
                if buffered_text.strip():
                    ts_str = time.strftime('%H:%M:%S', time.gmtime(last_ts))
                    line = f"{ts_str} | SPEAKER_{self.last_speaker:02d} | {buffered_text}\n"
                    print(line, end='')
                    self.log_file.write(line)
                    self.log_file.flush()
            print(f"Aligner completed. Processed {processed_chunks} chunks.")

    async def run(self):
        try:
            print("Starting pipeline...")
            await asyncio.gather(
                self.producer(),
                self.diar_consumer(),
                self.stt_consumer(),
                self.aligner()
            )
        except Exception as e:
            print(f"Error in pipeline: {e}")
        finally:
            self.log_file.close()
            print("\nPipeline completed.")

if __name__ == "__main__":
    pipeline = RealTimeWavPipeline(
        "taken_in.wav",
        sample_rate=16000,
        chunk_duration=0.15,  # Reduced for more precise processing
        log_path="transcription.log"
    )
    asyncio.run(pipeline.run())