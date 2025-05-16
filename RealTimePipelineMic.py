import asyncio
import time
import wave
import numpy as np
import torch
import torchaudio

from capture.audio_capture import AudioCapture
from diarization.speaker_diarization import SpeakerDiarization
from transcription.speech_to_text import SpeechToText

async def audio_producer(capture: AudioCapture,
                         diar_queue: asyncio.Queue,
                         stt_queue: asyncio.Queue,
                         save_path="output.wav"):

    await capture.start()
    device_sr = capture.sample_rate
    target_sr = 16000
    resampler = torchaudio.transforms.Resample(orig_freq=device_sr, new_freq=target_sr)

    start_time = time.time()
    frames = []

    try:
        while True:
            chunk = capture.get_audio_chunk(min_samples=device_sr)
            if chunk is None:
                await asyncio.sleep(0.01)
                continue

            # Convert to float32 if needed
            if chunk.dtype != np.float32:
                chunk = chunk.astype(np.float32)

            # Convert to tensor and resample to 16kHz
            tensor = torch.from_numpy(chunk).unsqueeze(0)
            resampled = resampler(tensor).squeeze(0).numpy()

            # Normalize audio
            max_val = np.max(np.abs(resampled)) or 1.0
            cleaned = resampled / max_val

            # Save frame for output WAV
            frames.append((cleaned * 32767).astype(np.int16))

            timestamp = time.time() - start_time
            packet = {"audio": cleaned, "timestamp": timestamp}
            await diar_queue.put(packet)
            await stt_queue.put(packet)

    except asyncio.CancelledError:
        pass
    finally:
        await capture.stop()

        if frames:
            with wave.open(save_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 2 bytes = int16
                wf.setframerate(target_sr)
                wf.writeframes(b''.join([f.tobytes() for f in frames]))
            print(f"Audio saved to {save_path}")

async def diarization_consumer(diar: SpeakerDiarization,
                               diar_queue: asyncio.Queue,
                               diar_results: asyncio.Queue):
    while True:
        packet = await diar_queue.get()
        audio = packet["audio"]
        ts = packet["timestamp"]
        await diar.process_audio(audio)
        speaker = diar.get_current_speaker()
        if speaker is not None:
            await diar_results.put({
                "start": ts,
                "speaker": speaker
            })

async def stt_consumer(stt: SpeechToText,
                       stt_queue: asyncio.Queue,
                       stt_results: asyncio.Queue):
    while True:
        packet = await stt_queue.get()
        audio = packet["audio"]
        ts = packet["timestamp"]
        await stt.process_audio(audio)
        text = stt.get_latest_transcription()
        if text:
            await stt_results.put({
                "start": ts,
                "text": text
            })

async def aligner(diar_results: asyncio.Queue,
                  stt_results: asyncio.Queue):
    while True:
        diar_evt = await diar_results.get()
        best_text = None
        while not stt_results.empty():
            stt_evt = stt_results.get_nowait()
            if abs(stt_evt["start"] - diar_evt["start"]) < 1.0:
                best_text = stt_evt["text"]
        if best_text:
            print(f"[{diar_evt['start']:.2f}s] Speaker {diar_evt['speaker']}: {best_text}")

async def main():
    capture = AudioCapture()
    diar = SpeakerDiarization()
    stt = SpeechToText()

    diar_queue = asyncio.Queue()
    stt_queue = asyncio.Queue()
    diar_results = asyncio.Queue()
    stt_results = asyncio.Queue()

    producer_task = asyncio.create_task(audio_producer(capture, diar_queue, stt_queue))
    diar_task = asyncio.create_task(diarization_consumer(diar, diar_queue, diar_results))
    stt_task = asyncio.create_task(stt_consumer(stt, stt_queue, stt_results))
    aligner_task = asyncio.create_task(aligner(diar_results, stt_results))

    try:
        await asyncio.gather(producer_task, diar_task, stt_task, aligner_task)
    except asyncio.CancelledError:
        print("Tasks cancelled")
    except KeyboardInterrupt:
        print("Keyboard interrupt received, stopping...")
        producer_task.cancel()
        diar_task.cancel()
        stt_task.cancel()
        aligner_task.cancel()
        await asyncio.gather(producer_task, diar_task, stt_task, aligner_task, return_exceptions=True)

if __name__ == "__main__":
    asyncio.run(main())
