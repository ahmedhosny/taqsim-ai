#!/usr/bin/env python3
"""
Audio Chunker Utility

This script splits audio files into overlapping chunks of specified duration.
It can be used as a standalone utility or imported as a module.
"""

import os
import traceback

import librosa
import numpy as np
import soundfile as sf


def create_chunks_directory(output_path=None):
    """
    Create necessary directories for storing audio chunks.

    Args:
        output_path: Optional custom path for the output directory

    Returns:
        Path to the audio chunks directory
    """
    if output_path is None:
        # Get the path to the data directory
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
        )
        # Create audio chunks directory
        output_path = os.path.join(data_dir, "audio_chunks")

    os.makedirs(output_path, exist_ok=True)
    return output_path


def process_audio_into_chunks(audio_file, target_sr=16000, chunk_duration=30):
    """
    Process audio file:
    1. Load and convert to target sample rate
    2. Split into non-overlapping chunks of fixed duration
    3. Pad the last chunk with silence if needed

    Args:
        audio_file: Path to the audio file
        target_sr: Target sample rate in Hz
        chunk_duration: Duration of each chunk in seconds

    Returns:
        List of audio chunks as numpy arrays
    """
    print(f"Processing audio file into chunks: {audio_file}")
    try:
        print(f"Loading audio file: {audio_file}")
        # Set duration=None to load the entire file
        audio, sr = librosa.load(audio_file, sr=target_sr, mono=True)
        print(
            f"Loaded audio file with sample rate {sr} Hz, duration: {len(audio) / sr:.2f}s"
        )

        if len(audio) == 0:
            raise ValueError("Loaded audio has zero length")

        print(
            f"Successfully loaded audio: {len(audio) / target_sr:.2f} seconds at {target_sr}Hz"
        )

        # Calculate chunk size in samples (30 seconds * sample rate)
        chunk_size = chunk_duration * target_sr

        # Calculate total number of chunks needed (ceiling division)
        total_chunks = (len(audio) + chunk_size - 1) // chunk_size

        # Split audio into non-overlapping chunks
        chunks = []
        for i in range(total_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, len(audio))

            # Extract the chunk
            chunk = audio[start_idx:end_idx]

            # If chunk is shorter than chunk_size, pad with zeros (silence)
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), "constant")

            chunks.append(chunk)

        print(
            f"Split audio into {len(chunks)} non-overlapping chunks of {chunk_duration} seconds each"
        )

        return chunks
    except Exception as e:
        print(f"Error processing audio into chunks: {e}")
        traceback.print_exc()
        return []


def save_audio_chunks(
    chunks, output_dir="audio_chunks", uuid="chunk", sr=16000, chunk_duration=30
):
    """
    Save audio chunks to disk with format {uuid}_{chunk_number}_{starting_second}_{end_second}.

    Args:
        chunks: List of audio chunks as numpy arrays
        output_dir: Directory to save the chunks
        uuid: Unique identifier for the audio file
        sr: Sample rate of the audio chunks
        chunk_duration: Duration of each chunk in seconds

    Returns:
        List of paths to the saved chunk files
    """
    chunk_paths = []
    for i, chunk in enumerate(chunks):
        # Calculate start and end seconds (chunk_number starts at 1)
        chunk_number = i + 1
        start_second = i * chunk_duration
        end_second = start_second + chunk_duration

        # Create filename with format {uuid}_{chunk_number}_{starting_second}_{end_second}
        filename = f"{uuid}_{chunk_number}_{start_second}_{end_second}.wav"
        chunk_path = os.path.join(output_dir, filename)

        sf.write(chunk_path, chunk, sr, "PCM_16")
        chunk_paths.append(chunk_path)

    print(f"Saved {len(chunks)} audio chunks to {output_dir}")
    return chunk_paths


def chunk_audio_file(
    input_file, output_dir=None, chunk_duration=30, target_sr=16000, uuid=None
):
    """
    Process a single audio file to create non-overlapping chunks of fixed duration.
    The last chunk will be padded with silence if needed.

    Args:
        input_file: Path to the input audio file
        output_dir: Directory to save the chunks (if None, will use default)
        chunk_duration: Duration of each chunk in seconds
        target_sr: Target sample rate in Hz

    Returns:
        List of paths to the saved chunk files, or empty list on failure
    """
    try:
        # Create output directory if needed
        if output_dir is None:
            output_dir = create_chunks_directory()

        # Process audio into chunks
        chunks = process_audio_into_chunks(
            input_file, target_sr=target_sr, chunk_duration=chunk_duration
        )

        if not chunks:
            print(f"No chunks were created from {input_file}")
            return []

        # Save chunks to disk with the specified format
        chunk_paths = save_audio_chunks(
            chunks,
            output_dir=output_dir,
            uuid=uuid,
            sr=target_sr,
            chunk_duration=chunk_duration,
        )

        return chunk_paths
    except Exception as e:
        print(f"Error chunking audio file: {e}")
        traceback.print_exc()
        return []
