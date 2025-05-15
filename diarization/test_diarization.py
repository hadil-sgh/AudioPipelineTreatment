import asyncio
import numpy as np
import librosa
import matplotlib.pyplot as plt
from speaker_diarization import SpeakerDiarization

async def test_vad(audio_path: str, diarization: SpeakerDiarization):
    """Test Voice Activity Detection"""
    print("\n=== Testing VAD ===")
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    # Get speech segments
    segments = diarization._detect_speech_segments(audio)
    
    # Plot audio and VAD results
    plt.figure(figsize=(15, 5))
    plt.plot(audio)
    for start, end in segments:
        start_sample = int(start * diarization.vad_hop_length)
        end_sample = int(end * diarization.vad_hop_length)
        plt.axvspan(start_sample, end_sample, alpha=0.3, color='red')
    plt.title('Audio with VAD Segments')
    plt.savefig('vad_test.png')
    plt.close()
    
    print(f"Found {len(segments)} speech segments")
    for i, (start, end) in enumerate(segments):
        duration = (end - start) * diarization.vad_hop_length / sr
        print(f"Segment {i+1}: {duration:.2f}s")

async def test_embeddings(audio_path: str, diarization: SpeakerDiarization):
    """Test Speaker Embedding Extraction"""
    print("\n=== Testing Embeddings ===")
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    # Process audio in chunks
    chunk_size = int(sr * 0.5)  # 500ms chunks
    embeddings = []
    
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        if len(chunk) < 1600:  # Skip chunks too short for model
            continue
            
        embedding = diarization._extract_features(chunk)
        if embedding is not None:
            embeddings.append(embedding)
            print(f"Chunk {i//chunk_size + 1}: Embedding shape {embedding.shape}")
    
    if embeddings:
        # Plot embedding similarity matrix
        embeddings_matrix = np.vstack(embeddings)
        similarity = np.dot(embeddings_matrix, embeddings_matrix.T)
        plt.figure(figsize=(10, 8))
        plt.imshow(similarity, cmap='viridis')
        plt.colorbar()
        plt.title('Embedding Similarity Matrix')
        plt.savefig('embedding_similarity.png')
        plt.close()
        
        print(f"\nProcessed {len(embeddings)} chunks successfully")
        print(f"Average embedding norm: {np.mean([np.linalg.norm(e) for e in embeddings]):.3f}")
    else:
        print("No valid embeddings extracted!")

async def test_clustering(audio_path: str, diarization: SpeakerDiarization):
    """Test Speaker Clustering"""
    print("\n=== Testing Clustering ===")
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    # Process audio and collect embeddings
    chunk_size = int(sr * 0.5)  # 500ms chunks
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        if len(chunk) < 1600:
            continue
        await diarization.process_audio(chunk)
    
    await diarization.finalize_processing()
    
    # Analyze clustering results
    if diarization.speaker_labels:
        print(f"\nTotal segments: {len(diarization.speaker_labels)}")
        print(f"Speaker distribution: {np.bincount(diarization.speaker_labels)}")
        
        # Plot speaker labels over time
        plt.figure(figsize=(15, 3))
        plt.plot(diarization.speaker_labels, 'b-', alpha=0.5)
        plt.title('Speaker Labels Over Time')
        plt.xlabel('Segment Index')
        plt.ylabel('Speaker Label')
        plt.savefig('speaker_labels.png')
        plt.close()
    else:
        print("No speaker labels generated!")

async def main():
    audio_path = "taken_in.wav"
    
    # Initialize diarization with debug parameters
    diarization = SpeakerDiarization(
        min_speech_duration=0.1,
        threshold=0.005,
        n_speakers=2,
        process_buffer_duration=0.15
    )
    
    # Run component tests
    await test_vad(audio_path, diarization)
    await test_embeddings(audio_path, diarization)
    await test_clustering(audio_path, diarization)
    
    # Cleanup
    diarization.reset()

if __name__ == "__main__":
    asyncio.run(main()) 