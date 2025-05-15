import asyncio
import time

from capture.audio_capture import AudioCapture
from diarization.speaker_diarization import SpeakerDiarization
from transcription.speech_to_text import SpeechToText

async def audio_producer(capture: AudioCapture,
                         diar_queue: asyncio.Queue,
                         stt_queue: asyncio.Queue):
    """
    Continuously polls AudioCapture for full chunks and distributes
    them to both diarization and transcription queues.
    """
    await capture.start()
    start_time = time.time()
    try:
        while True:
            chunk = capture.get_audio_chunk(min_samples=capture.sample_rate)
            if chunk is not None:
                timestamp = time.time() - start_time
                # Tag chunk with its start timestamp
                packet = {"audio": chunk, "timestamp": timestamp}
                await diar_queue.put(packet)
                await stt_queue.put(packet)
            await asyncio.sleep(0.01)
    finally:
        await capture.stop()

async def diarization_consumer(diar: SpeakerDiarization,
                               diar_queue: asyncio.Queue,
                               diar_results: asyncio.Queue):
    """
    Consumes audio packets, feeds them into SpeakerDiarization,
    and outputs timestamped speaker labels.
    """
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
    """
    Consumes audio packets, feeds them into SpeechToText,
    and outputs timestamped transcriptions.
    """
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
    """
    Merges diarization and transcription events by matching closest timestamps.
    Emits final speaker-tagged transcripts.
    """
    while True:
        diar_evt = await diar_results.get()
        best_text = None
        # Look for the STT event with the closest timestamp
        # (this is a simple example; you might want sliding window or sorted buffer)
        while not stt_results.empty():
            stt_evt = stt_results.get_nowait()
            if abs(stt_evt["start"] - diar_evt["start"]) < 1.0:
                best_text = stt_evt["text"]
        if best_text:
            print(f"[{diar_evt['start']:.2f}s] Speaker {diar_evt['speaker']}: {best_text}")

async def main():
    # Instantiate modules
    capture = AudioCapture()
    diar = SpeakerDiarization()
    stt = SpeechToText()

    # Queues for inter-task communication
    diar_queue = asyncio.Queue()
    stt_queue = asyncio.Queue()
    diar_results = asyncio.Queue()
    stt_results = asyncio.Queue()

    # Launch all tasks
    await asyncio.gather(
        audio_producer(capture, diar_queue, stt_queue),
        diarization_consumer(diar, diar_queue, diar_results),
        stt_consumer(stt, stt_queue, stt_results),
        aligner(diar_results, stt_results)
    )

if __name__ == "__main__":
    asyncio.run(main())
