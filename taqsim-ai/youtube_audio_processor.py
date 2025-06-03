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

# Import chunking functionality from audio_chunker
from audio_chunker import chunk_audio_file
from transformers import pipeline


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

# Note: Silence removal functionality has been moved to silence_remover.py

# Note: Audio chunking functionality has been moved to audio_chunker.py


def classify_audio_files(audio_file_paths, extract_embeddings=False):
    """
    Classify audio files using the MAEST model.

    Args:
        audio_file_paths: List of paths to audio files
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
    chunks = []

    # First load all audio files
    for i, file_path in enumerate(audio_file_paths):
        print(f"Loading audio file {i + 1}/{len(audio_file_paths)}: {file_path}")
        try:
            # Load audio file with librosa
            audio, sr = librosa.load(file_path, sr=16000, mono=True)
            chunks.append(audio)

            # Classify the audio
            print(f"Classifying chunk {i + 1}/{len(audio_file_paths)}...")
            result = pipe(audio)
            results.append(result)

            # Print top 3 predictions for this chunk
            print(f"\nChunk {i + 1} Classification Results:")
            for pred in result[:3]:
                print(f"Label: {pred['label']}, Score: {pred['score']:.4f}")

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            import traceback

            traceback.print_exc()

    # Extract embeddings if requested
    if extract_embeddings and chunks:
        print("\nExtracting embeddings from the model...")
        embeddings_dict = extract_embeddings_from_maest(chunks)
        return results, embeddings_dict

    return results


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


def find_downloaded_file(youtube_url, downloads_dir, uuid):
    """
    Find a previously downloaded file for a YouTube URL using UUID.

    Args:
        youtube_url: URL of the YouTube video (for reference only)
        downloads_dir: Directory to look for the downloaded file
        uuid: Required unique identifier for the file

    Returns:
        Path to the downloaded file if found, None otherwise

    Raises:
        ValueError: If uuid is not provided
    """
    # Require UUID parameter
    if not uuid:
        raise ValueError(
            "UUID is required for finding files. Cannot proceed without a UUID."
        )

    # Check if the file is already downloaded
    for file in os.listdir(downloads_dir):
        if uuid in file:
            existing_file = os.path.join(downloads_dir, file)
            print(f"Found downloaded file: {existing_file}")
            return existing_file

    print(f"No downloaded file found for UUID: {uuid}")
    print("Please run youtube_downloader.py first to download the file.")
    return None


def process_youtube_link(youtube_url, extract_embeddings=True, uuid=None):
    """
    Process a YouTube link: find downloaded file, process, and classify.
    Assumes the file has already been downloaded using youtube_downloader.py.

    Args:
        youtube_url: URL of the YouTube video
        extract_embeddings: Whether to extract embeddings from the model
        uuid: Required unique identifier for the file

    Returns:
        Tuple of (classification results, embeddings_dict) if extract_embeddings is True,
        otherwise just classification results

    Raises:
        ValueError: If uuid is not provided
    """
    # Require UUID parameter
    if not uuid:
        raise ValueError(
            "UUID is required for processing. Cannot proceed without a UUID."
        )

    # Create directories and get paths
    audio_chunks_dir, downloads_dir, classifications_dir, embeddings_dir = (
        create_directories()
    )

    # Find the downloaded audio file using UUID
    audio_file = find_downloaded_file(youtube_url, downloads_dir, uuid)
    if not audio_file:
        return None

    # Process audio and remove silence
    # Use the new chunking functionality from audio_chunker.py
    audio_chunks_dir = create_directories()[0]  # Get the audio chunks directory
    chunk_paths = chunk_audio_file(audio_file, output_dir=audio_chunks_dir, uuid=uuid)

    # If chunking failed, return early
    if not chunk_paths:
        print(f"Failed to create chunks from {audio_file}")
        return None

    # Use the UUID for naming files
    file_id = uuid

    # Classify chunk files and optionally extract embeddings
    if extract_embeddings:
        results, embeddings_dict = classify_audio_files(
            chunk_paths, extract_embeddings=True
        )

        # Create embeddings directory if it doesn't exist
        embeddings_dir = os.path.join(os.path.dirname(audio_chunks_dir), "embeddings")
        os.makedirs(embeddings_dir, exist_ok=True)

        # Save embeddings to file
        embeddings_file = os.path.join(
            embeddings_dir, f"embeddings_youtube_{file_id}.npz"
        )
        save_embeddings_to_file(embeddings_dict, output_file=embeddings_file)
        print(f"Embeddings saved to: {embeddings_file}")
    else:
        results = classify_audio_files(chunk_paths, extract_embeddings=False)

    # Save results to genre_classifications directory
    results_file = os.path.join(
        classifications_dir, f"classification_youtube_{file_id}.csv"
    )
    save_results_to_csv(results, output_file=results_file)

    if extract_embeddings:
        return results, embeddings_dict
    return results


def process_csv_file(csv_file, extract_embeddings=True):
    """
    Process a CSV file containing YouTube URLs and process each URL.

    Args:
        csv_file: Path to the CSV file containing YouTube URLs
        extract_embeddings: Whether to extract embeddings from the model

    Returns:
        None
    """
    try:
        # Use pandas to read the CSV file
        df = pd.read_csv(csv_file)

        if df.empty:
            print("CSV file is empty.")
            return

        # Check if required columns exist
        if "link" not in df.columns:
            print("Error: CSV file must contain a 'link' column with YouTube URLs.")
            return

        if "uuid" not in df.columns:
            print(
                "Error: CSV file must contain a 'uuid' column with unique identifiers."
            )
            return

        # Process each row
        for index, row in df.iterrows():
            try:
                url = row["link"]
                uuid = row["uuid"]

                # Skip rows with missing UUID
                if pd.isna(uuid) or not uuid:
                    print(f"\nSkipping row {index}: Missing UUID for URL {url}")
                    continue

                song_info = (
                    f" - {row['song_name']}"
                    if "song_name" in df.columns and not pd.isna(row["song_name"])
                    else ""
                )
                print(f"\nProcessing YouTube URL: {url}{song_info} (UUID: {uuid})")

                process_youtube_link(
                    url, extract_embeddings=extract_embeddings, uuid=uuid
                )
            except ValueError as e:
                print(f"Error processing row {index}: {e}")
                continue
    except Exception as e:
        print(f"Error processing CSV file: {e}")


def main():
    """
    Main function to process YouTube URLs from command line or CSV file.
    """
    parser = argparse.ArgumentParser(
        description="Process YouTube URLs for audio classification."
    )
    parser.add_argument(
        "--url",
        help="YouTube URL to process (requires --uuid parameter)",
    )
    parser.add_argument(
        "--uuid",
        help="UUID to use for the file (required when using --url)",
    )
    parser.add_argument(
        "--csv",
        help="Path to CSV file containing YouTube URLs and UUIDs",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all URLs in the CSV file (default: only first URL)",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip extracting embeddings (faster processing)",
    )
    args = parser.parse_args()
    extract_embeddings = not args.no_embeddings

    if args.url:
        if not args.uuid:
            print("Error: --uuid parameter is required when using --url")
            return
        print(f"\nProcessing YouTube URL: {args.url} (UUID: {args.uuid})")
        process_youtube_link(
            args.url, extract_embeddings=extract_embeddings, uuid=args.uuid
        )
    elif args.csv:
        print(f"\nProcessing YouTube URLs from CSV file: {args.csv}")
        process_csv_file(args.csv, extract_embeddings=extract_embeddings)
    else:
        # Use default CSV file
        csv_path = Path(__file__).parent.parent / "data" / "taqsim_ai.csv"
        if os.path.exists(csv_path):
            print(f"\nUsing default CSV file: {csv_path}")
            process_csv_file(csv_path, extract_embeddings=extract_embeddings)
        else:
            print(f"Error: Default CSV file not found at {csv_path}")
            print(
                "Please specify a CSV file with --csv or provide a URL with --url and --uuid"
            )
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
