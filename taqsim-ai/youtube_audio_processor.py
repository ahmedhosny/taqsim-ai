"""
YouTube Audio Processor and Classifier

This script:
1. Downloads audio from a YouTube video
2. Converts it to 16kHz sample rate
3. Splits it into 30-second chunks
4. Classifies each chunk using the MAEST audio classification model
"""

import argparse
import csv
import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from transformers import pipeline

# Import YouTube downloader functionality for finding files
from youtube_downloader import extract_video_id


# Create necessary directories
def create_directories():
    """Create necessary directories for storing audio files, classification results, and embeddings."""
    # Get the path to the data directory
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )

    # Create subdirectories under data/
    audio_chunks_dir = os.path.join(data_dir, "audio_chunks")
    downloads_dir = os.path.join(data_dir, "downloads")
    classifications_dir = os.path.join(data_dir, "genre_classifications")
    embeddings_dir = os.path.join(data_dir, "embeddings")

    os.makedirs(audio_chunks_dir, exist_ok=True)
    os.makedirs(downloads_dir, exist_ok=True)
    os.makedirs(classifications_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)

    return audio_chunks_dir, downloads_dir, classifications_dir, embeddings_dir


# Note: download_youtube_audio is now imported from youtube_downloader module


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


def process_audio(audio_file, target_sr=16000, chunk_duration=30, step_size=1):
    """
    Process audio file:
    1. Load and convert to target sample rate
    2. Remove silence from beginning and end
    3. Split into overlapping chunks of specified duration with step size

    Args:
        audio_file: Path to the audio file
        target_sr: Target sample rate in Hz
        chunk_duration: Duration of each chunk in seconds
        step_size: Step size between consecutive chunks in seconds

    Returns:
        List of audio chunks as numpy arrays
    """
    print(f"Processing audio file: {audio_file}")
    try:
        # Try to convert webm to wav first if it's a webm file
        converted_file = audio_file
        if audio_file.lower().endswith(".webm"):
            try:
                import subprocess

                wav_file = audio_file.rsplit(".", 1)[0] + ".wav"
                print(f"Attempting to convert webm to wav: {wav_file}")

                try:
                    # Try using ffmpeg if available
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-i",
                            audio_file,
                            "-acodec",
                            "pcm_s16le",
                            "-ar",
                            str(target_sr),
                            wav_file,
                        ],
                        check=True,
                        capture_output=True,
                    )
                    converted_file = wav_file
                    print("Successfully converted to wav using ffmpeg")
                except (subprocess.SubprocessError, FileNotFoundError):
                    print(
                        "FFmpeg conversion failed, will try to process the original file"
                    )
            except Exception as conv_e:
                print(f"Error during conversion: {conv_e}")

        print(f"Loading audio file: {converted_file}")
        # Set duration=None to load the entire file
        audio, sr = librosa.load(converted_file, sr=target_sr, mono=True)
        print(
            f"Loaded audio file with sample rate {sr} Hz, duration: {len(audio) / sr:.2f}s"
        )

        # Remove silence from beginning and end
        audio = remove_silence(audio, sr)
        if len(audio) == 0:
            raise ValueError("Loaded audio has zero length")

        print(
            f"Successfully loaded audio: {len(audio) / target_sr:.2f} seconds at {target_sr}Hz"
        )

        # Calculate chunk size and step size in samples
        chunk_size = chunk_duration * target_sr
        step_samples = step_size * target_sr

        # Calculate total number of chunks
        total_chunks = max(1, 1 + (len(audio) - chunk_size) // step_samples)

        # Split audio into overlapping chunks
        chunks = []
        for i in range(total_chunks):
            start_idx = i * step_samples
            end_idx = min(start_idx + chunk_size, len(audio))

            # Extract the chunk
            chunk = audio[start_idx:end_idx]

            # If chunk is shorter than chunk_size, pad with zeros
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), "constant")

            chunks.append(chunk)

        print(
            f"Split audio into {len(chunks)} overlapping chunks of {chunk_duration} seconds each (step size: {step_size}s)"
        )

        return chunks
    except Exception as e:
        print(f"Error processing audio: {e}")
        import traceback

        traceback.print_exc()
        return []


def save_audio_chunks(
    chunks, output_dir="audio_chunks", base_filename="chunk", sr=16000
):
    """
    Save audio chunks to disk.

    Args:
        chunks: List of audio chunks as numpy arrays
        output_dir: Directory to save the chunks
        base_filename: Base name for the chunk files
        sr: Sample rate of the audio chunks

    Returns:
        List of paths to the saved chunk files
    """
    import soundfile as sf

    chunk_paths = []
    for i, chunk in enumerate(chunks):
        # Use i+1 to start numbering from 1 instead of 0
        chunk_path = os.path.join(output_dir, f"{base_filename}_{i + 1:03d}.wav")
        sf.write(chunk_path, chunk, sr, "PCM_16")
        chunk_paths.append(chunk_path)

    print(f"Saved {len(chunks)} audio chunks to {output_dir}")
    return chunk_paths


def extract_embeddings_from_maest(
    chunks, model_name="mtg-upf/discogs-maest-30s-pw-129e", transformer_block=6
):
    """
    Extract embeddings from the MAEST model for each audio chunk.

    Args:
        chunks: List of audio chunks as numpy arrays
        model_name: Name of the MAEST model to use
        transformer_block: Which transformer block to extract embeddings from (default: 6)

    Returns:
        Dictionary with chunk indices as keys and embeddings as values
    """
    print(f"Loading MAEST model for embedding extraction: {model_name}")

    try:
        # Try to import the MAEST model directly
        try:
            # First try to import from maest package if installed
            from maest import get_maest

            model = get_maest(arch=model_name.split("/")[-1])
            print("Successfully loaded MAEST model from maest package")
            direct_maest = True
        except ImportError:
            # If that fails, use the transformers model
            from transformers import AutoFeatureExtractor, AutoModel

            model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                model_name, trust_remote_code=True
            )
            print("Using transformers model for embedding extraction")
            direct_maest = False

        # Set model to evaluation mode
        model.eval()

        embeddings_dict = {}

        import torch

        with torch.no_grad():
            for i, chunk in enumerate(chunks):
                print(f"Extracting embeddings for chunk {i + 1}/{len(chunks)}...")

                if direct_maest:
                    # Use the direct MAEST interface
                    _, embeddings = model(chunk, transformer_block=transformer_block)
                    # Convert to numpy array
                    embeddings = embeddings.cpu().numpy()
                else:
                    # Use the transformers interface
                    inputs = feature_extractor(
                        chunk, sampling_rate=16000, return_tensors="pt"
                    )
                    outputs = model(**inputs, output_hidden_states=True)

                    # Get embeddings from the specified transformer block
                    hidden_states = outputs.hidden_states[transformer_block]

                    # Extract CLS token, DIST token, and average of rest of tokens
                    cls_token = hidden_states[0, 0].cpu().numpy()  # CLS token
                    dist_token = hidden_states[0, 1].cpu().numpy()  # DIST token
                    rest_tokens = (
                        hidden_states[0, 2:].mean(dim=0).cpu().numpy()
                    )  # Average of rest

                    # Stack them together
                    embeddings = np.vstack([cls_token, dist_token, rest_tokens])

                # Store embeddings for this chunk
                embeddings_dict[i + 1] = embeddings

        return embeddings_dict

    except Exception as e:
        print(f"Error extracting embeddings: {e}")
        import traceback

        traceback.print_exc()
        return {}


def save_embeddings_to_file(embeddings_dict, output_file="embeddings.npz"):
    """
    Save embeddings to a numpy compressed file.

    Args:
        embeddings_dict: Dictionary with chunk indices as keys and embeddings as values
        output_file: Path to the output file
    """
    try:
        # Convert integer keys to strings for np.savez_compressed
        string_dict = {f"chunk_{key}": value for key, value in embeddings_dict.items()}
        np.savez_compressed(output_file, **string_dict)
        print(f"Saved embeddings to {output_file}")
        return True
    except Exception as e:
        print(f"Error saving embeddings: {e}")
        import traceback

        traceback.print_exc()
        return False


def classify_audio_chunks(chunks, extract_embeddings=False):
    """
    Classify audio chunks using the MAEST model.

    Args:
        chunks: List of audio chunks as numpy arrays
        extract_embeddings: Whether to also extract embeddings from the model

    Returns:
        Tuple of (classification results, embeddings_dict) if extract_embeddings is True,
        otherwise just classification results
    """
    print("Loading audio classification model...")
    pipe = pipeline(
        "audio-classification",
        model="mtg-upf/discogs-maest-30s-pw-129e",
        trust_remote_code=True,
    )

    results = []
    for i, chunk in enumerate(chunks):
        print(f"Classifying chunk {i + 1}/{len(chunks)}...")
        result = pipe(chunk)
        results.append(result)

        # Print top 3 predictions for this chunk
        print(f"\nChunk {i + 1} Classification Results:")
        for pred in result[:3]:
            print(f"Label: {pred['label']}, Score: {pred['score']:.4f}")

    # Extract embeddings if requested
    if extract_embeddings:
        print("\nExtracting embeddings from the model...")
        embeddings_dict = extract_embeddings_from_maest(chunks)
        return results, embeddings_dict

    return results


def save_results_to_csv(results, output_file="classification_results.csv"):
    """
    Save classification results to a CSV file.
    Only saves the top three predictions for each chunk.

    Args:
        results: List of classification results for each chunk
        output_file: Path to the output CSV file
    """
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Chunk", "Rank", "Label", "Score"])

        for chunk_idx, chunk_results in enumerate(results):
            # Only save the top three predictions for each chunk
            for rank, pred in enumerate(chunk_results[:3]):
                writer.writerow([chunk_idx + 1, rank + 1, pred["label"], pred["score"]])

    print(
        f"Saved classification results to {output_file} (top 3 predictions per chunk)"
    )


def find_downloaded_file(youtube_url, downloads_dir):
    """
    Find a previously downloaded file for a YouTube URL.

    Args:
        youtube_url: URL of the YouTube video
        downloads_dir: Directory to look for the downloaded file

    Returns:
        Path to the downloaded file if found, None otherwise
    """
    # Extract video ID for finding the file
    video_id = extract_video_id(youtube_url)

    # Check if the video is already downloaded
    for file in os.listdir(downloads_dir):
        if video_id in file:
            existing_file = os.path.join(downloads_dir, file)
            print(f"Found downloaded file: {existing_file}")
            return existing_file

    print(f"No downloaded file found for video ID: {video_id}")
    print("Please run youtube_downloader.py first to download the file.")
    return None


def process_youtube_link(youtube_url, extract_embeddings=True):
    """
    Process a YouTube link: find downloaded file, process, and classify.
    Assumes the file has already been downloaded using youtube_downloader.py.

    Args:
        youtube_url: URL of the YouTube video
        extract_embeddings: Whether to extract embeddings from the model

    Returns:
        Tuple of (classification results, embeddings_dict) if extract_embeddings is True,
        otherwise just classification results
    """
    # Create directories and get paths
    audio_chunks_dir, downloads_dir, classifications_dir, embeddings_dir = (
        create_directories()
    )

    # Find the downloaded audio file
    audio_file = find_downloaded_file(youtube_url, downloads_dir)
    if not audio_file:
        return None

    # Process audio and remove silence
    chunks = process_audio(audio_file, step_size=1)
    if not chunks:
        return None

    # Extract video ID for naming files
    video_id = extract_video_id(youtube_url)

    # Save chunks
    save_audio_chunks(
        chunks, output_dir=audio_chunks_dir, base_filename=f"youtube_{video_id}"
    )

    # Classify chunks and optionally extract embeddings
    if extract_embeddings:
        results, embeddings_dict = classify_audio_chunks(
            chunks, extract_embeddings=True
        )

        # Create embeddings directory if it doesn't exist
        embeddings_dir = os.path.join(os.path.dirname(audio_chunks_dir), "embeddings")
        os.makedirs(embeddings_dir, exist_ok=True)

        # Save embeddings to file
        embeddings_file = os.path.join(
            embeddings_dir, f"embeddings_youtube_{video_id}.npz"
        )
        save_embeddings_to_file(embeddings_dict, output_file=embeddings_file)
        print(f"Embeddings saved to: {embeddings_file}")
    else:
        results = classify_audio_chunks(chunks, extract_embeddings=False)

    # Save results to genre_classifications directory
    results_file = os.path.join(
        classifications_dir, f"classification_youtube_{video_id}.csv"
    )
    save_results_to_csv(results, output_file=results_file)

    if extract_embeddings:
        return results, embeddings_dict
    return results


def main():
    """Main function to process YouTube links from command line or CSV file."""
    parser = argparse.ArgumentParser(
        description="Process YouTube videos for audio classification"
    )
    parser.add_argument("--url", help="YouTube URL to process")
    parser.add_argument("--csv", help="CSV file containing YouTube URLs")
    parser.add_argument(
        "--all", action="store_true", help="Process all URLs in the CSV file"
    )
    parser.add_argument(
        "--no-embeddings", action="store_true", help="Skip embedding extraction"
    )

    args = parser.parse_args()
    extract_embeddings = not args.no_embeddings

    if args.url:
        print("\nProcessing a single YouTube URL...")
        process_youtube_link(args.url, extract_embeddings=extract_embeddings)
    elif args.csv:
        print(f"\nProcessing YouTube URLs from CSV file: {args.csv}")
        # Use pandas to read the CSV file
        df = pd.read_csv(args.csv)

        if df.empty:
            print("CSV file is empty.")
            return

        # Check if 'link' column exists
        if "link" not in df.columns:
            print("Error: CSV file must contain a 'link' column with YouTube URLs.")
            return

        # Process each row
        for index, row in df.iterrows():
            url = row["link"]  # URL is in the 'link' column
            song_info = f" - {row['song_name']}" if "song_name" in df.columns else ""
            print(f"\nProcessing YouTube URL: {url}{song_info}")
            process_youtube_link(url, extract_embeddings=extract_embeddings)
    else:
        # Use URLs from the CSV file
        csv_path = Path(__file__).parent.parent / "data" / "taqsim_ai.csv"
        if csv_path.exists():
            print(f"\nUsing default CSV file: {csv_path}")
            try:
                # Use pandas to read the CSV file
                df = pd.read_csv(csv_path)

                if df.empty:
                    print("CSV file is empty.")
                    return

                # Check if 'link' column exists
                if "link" not in df.columns:
                    print(
                        "Error: CSV file must contain a 'link' column with YouTube URLs."
                    )
                    return

                processed = False
                # Process each row (or just the first if args.all is False)
                for index, row in df.iterrows():
                    url = row["link"]
                    song_info = (
                        f" - {row['song_name']}" if "song_name" in df.columns else ""
                    )
                    print(f"\nProcessing YouTube URL: {url}{song_info}")
                    process_youtube_link(url, extract_embeddings=extract_embeddings)
                    processed = True
                    if not args.all:
                        break  # Process only the first URL by default

                if not processed:
                    print("No valid YouTube URLs found in the CSV file.")
            except Exception as e:
                print(f"Error reading CSV file: {e}")
                return
        else:
            print("No YouTube URL provided and no CSV file found.")
            print("Please provide a YouTube URL with --url or a CSV file with --csv")
            print(
                "\nFirst run youtube_downloader.py to download the audio files, then run this script to process them."
            )
            print(
                "Example: python youtube_downloader.py --csv data/taqsim_ai.csv --all"
            )
            print(
                "Then: python youtube_audio_processor.py --csv data/taqsim_ai.csv --all"
            )


if __name__ == "__main__":
    main()
