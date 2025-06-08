#!/usr/bin/env python3
"""
Silence Remover Utility

This script removes silence from the beginning and end of audio files.
It can process a single file or batch process files listed in a CSV.
"""

import os
import traceback

import librosa
import soundfile as sf


def create_processed_directory(output_path=None):
    """
    Create necessary directories for storing processed audio files.

    Args:
        output_path: Optional custom path for the output directory

    Returns:
        Path to the downloads no silence directory
    """
    if output_path is None:
        # Get the path to the data directory
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
        )
        # Create downloads directory
        output_path = os.path.join(data_dir, "processed")

    os.makedirs(output_path, exist_ok=True)
    return output_path


def remove_silence(audio, sr, threshold_db=40, min_silence_duration=0.5):
    """
    Remove silence from the beginning and end of an audio signal only.
    Preserves all audio content (including silences) in the middle of the track.
    Uses librosa's trim function which is specifically designed for trimming silence
    at the beginning and end of audio.

    Args:
        audio: Audio signal as numpy array
        sr: Sample rate of the audio
        threshold_db: Threshold in decibels below reference to consider as silence (positive value)
        min_silence_duration: Minimum duration of silence in seconds (not used in trim method)

    Returns:
        Trimmed audio signal with silence removed only from beginning and end
    """
    print("Removing silence from beginning and end of audio only...")

    # First attempt with the specified threshold
    trimmed_audio, trim_indices = librosa.effects.trim(
        audio, top_db=threshold_db, frame_length=2048, hop_length=512
    )

    # If trimming removed too much (resulting in very short audio), try with a more lenient threshold
    if len(trimmed_audio) / sr < 10.0:  # at least 10 seconds
        more_lenient_threshold = (
            threshold_db - 10
        )  # Lower threshold to detect more sounds
        print(
            f"First trim resulted in very short audio. Trying more lenient threshold: {more_lenient_threshold}dB"
        )
        trimmed_audio, trim_indices = librosa.effects.trim(
            audio, top_db=more_lenient_threshold, frame_length=2048, hop_length=512
        )

    # If still too short or no trimming occurred, return original
    if len(trimmed_audio) / sr < 5.0 or len(trimmed_audio) == len(audio):
        print("Minimal or no trimming possible, returning original audio")
        return audio

    # Get the start and end indices
    start_sample, end_sample = trim_indices

    # Convert to time for logging
    start_time = start_sample / sr
    end_time = (len(audio) - end_sample) / sr
    duration = len(trimmed_audio) / sr

    print(f"Trimmed {start_time:.2f}s from beginning and {end_time:.2f}s from end")
    print(f"New audio duration: {duration:.2f}s")

    # Return the trimmed audio - this only removes silence from beginning and end
    return trimmed_audio


def find_downloaded_file(uuid, downloads_dir):
    """
    Find a previously downloaded file using UUID.

    Args:
        uuid: Unique identifier for the file
        downloads_dir: Directory to look for the downloaded file

    Returns:
        Path to the downloaded file if found, None otherwise
    """
    # Check if the file is already downloaded
    for file in os.listdir(downloads_dir):
        if uuid in file:
            existing_file = os.path.join(downloads_dir, file)
            print(f"Found downloaded file: {existing_file}")
            return existing_file

    print(f"No downloaded file found for UUID: {uuid}")
    return None


def process_audio_file(input_file, output_file, target_sr=16000):
    """
    Process a single audio file to remove silence from beginning and end.

    Args:
        input_file: Path to the input audio file
        output_file: Path to save the processed audio file
        target_sr: Target sample rate in Hz (default 16kHz for ML applications)

    Returns:
        True if processing was successful, False otherwise
    """
    try:
        print(f"Processing audio file: {input_file}")

        # Check if output file already exists
        if os.path.exists(output_file):
            print(f"Output file already exists: {output_file}. Skipping.")
            return True

        # Load audio file with consistent sample rate for ML applications
        audio, sr = librosa.load(input_file, sr=target_sr, mono=True)
        print(
            f"Loaded audio file with sample rate {sr} Hz, duration: {len(audio) / sr:.2f}s"
        )

        # Remove silence from beginning and end
        processed_audio = remove_silence(audio, sr)

        # Save processed audio
        print(f"Saving processed audio to: {output_file}")
        # Use consistent format for ML applications
        if output_file.lower().endswith(".wav"):
            # For WAV files, use PCM_16 format for consistency in ML
            sf.write(output_file, processed_audio, sr, subtype="PCM_16")
        else:
            # For other formats, use default settings
            sf.write(output_file, processed_audio, sr)
        print(f"Successfully saved processed audio: {output_file}")

        return True
    except Exception as e:
        print(f"Error processing audio file: {e}")
        traceback.print_exc()
        return False
