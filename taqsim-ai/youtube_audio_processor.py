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
import yt_dlp
from transformers import pipeline


# Create necessary directories
def create_directories():
    """Create necessary directories for storing audio files."""
    # Get the path to the data directory
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )

    # Create subdirectories under data/
    audio_chunks_dir = os.path.join(data_dir, "audio_chunks")
    downloads_dir = os.path.join(data_dir, "downloads")

    os.makedirs(audio_chunks_dir, exist_ok=True)
    os.makedirs(downloads_dir, exist_ok=True)

    return audio_chunks_dir, downloads_dir


def download_youtube_audio(youtube_url, output_path=None):
    """
    Download audio from a YouTube video using yt-dlp.

    Args:
        youtube_url: URL of the YouTube video
        output_path: Directory to save the downloaded audio. If None, uses data/downloads

    Returns:
        Path to the downloaded audio file
    """
    print(f"Downloading audio from: {youtube_url}")

    # Use the downloads directory if no output path is specified
    if output_path is None:
        _, output_path = create_directories()

    try:
        # Extract video ID for naming the output file
        try:
            video_id = youtube_url.split("v=")[1]
            if "&" in video_id:
                video_id = video_id.split("&")[0]
        except IndexError:
            # Fallback if URL format is unexpected
            import uuid

            video_id = str(uuid.uuid4())[:8]
            print(
                f"Could not extract video ID from URL, using generated ID: {video_id}"
            )

        # Check for ffmpeg availability
        ffmpeg_available = False
        ffmpeg_path = None

        try:
            import shutil

            # Try to find ffmpeg in PATH
            ffmpeg_path = shutil.which("ffmpeg")
            if ffmpeg_path:
                print(f"Found ffmpeg at: {ffmpeg_path}")
                ffmpeg_available = True
            else:
                # Try common locations
                common_paths = [
                    "/usr/bin/ffmpeg",
                    "/usr/local/bin/ffmpeg",
                    "/opt/homebrew/bin/ffmpeg",
                ]
                for path in common_paths:
                    if os.path.exists(path) and os.access(path, os.X_OK):
                        ffmpeg_path = path
                        ffmpeg_available = True
                        print(f"Found ffmpeg at: {ffmpeg_path}")
                        break
        except Exception as e:
            print(f"Error checking for ffmpeg: {e}")

        if not ffmpeg_available:
            print("WARNING: ffmpeg not found. Audio processing may be limited.")
            print("Consider installing ffmpeg: brew install ffmpeg")

        # Set up yt-dlp options
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(output_path, f"%(title)s_{video_id}.%(ext)s"),
            "quiet": False,
            "no_warnings": False,
            "ignoreerrors": False,
        }

        # Add post-processing if ffmpeg is available
        if ffmpeg_available:
            # Force extraction to wav format which librosa can handle better
            ydl_opts["postprocessors"] = [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",  # Use wav instead of mp3 for better compatibility
                    "preferredquality": "192",
                }
            ]
            # If we found a specific ffmpeg path, use it
            if ffmpeg_path:
                ydl_opts["ffmpeg_location"] = os.path.dirname(ffmpeg_path)

        print("Initializing YouTube downloader...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print("Extracting video information...")
            info = ydl.extract_info(youtube_url, download=False)

            if not info:
                raise Exception("Could not extract video information")

            print(f"Downloading: {info.get('title', 'Unknown title')}")
            ydl.download([youtube_url])

            # Get the output filename
            title = info.get("title", "Unknown")
            safe_title = "".join([c if c.isalnum() else "_" for c in title])

            # Check for the downloaded file
            found_file = None

            # First check for the expected filename with wav extension if ffmpeg was available
            if ffmpeg_available:
                expected_file = os.path.join(
                    output_path, f"{safe_title}_{video_id}.wav"
                )
                if os.path.exists(expected_file):
                    found_file = expected_file

            # If not found, search for any file with the video_id in the name
            if not found_file:
                for file in os.listdir(output_path):
                    if video_id in file:
                        found_file = os.path.join(output_path, file)
                        break

            if found_file:
                print(f"Downloaded to: {found_file}")
                return found_file

            raise Exception(f"Downloaded file not found in {output_path}")

    except Exception as e:
        print(f"Error downloading YouTube video: {e}")
        print("This could be due to YouTube API changes or restrictions.")
        print("Check if the video is region-restricted or age-restricted")
        return None


def process_audio(audio_file, target_sr=16000, chunk_duration=30):
    """
    Process audio file:
    1. Load and convert to target sample rate
    2. Split into chunks of specified duration

    Args:
        audio_file: Path to the audio file
        target_sr: Target sample rate in Hz
        chunk_duration: Duration of each chunk in seconds

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
        y, sr = librosa.load(converted_file, sr=target_sr, mono=True)

        if len(y) == 0:
            raise ValueError("Loaded audio has zero length")

        print(
            f"Successfully loaded audio: {len(y) / target_sr:.2f} seconds at {target_sr}Hz"
        )

        # Calculate chunk size in samples
        chunk_size = chunk_duration * target_sr

        # Split audio into chunks
        chunks = []
        for i in range(0, len(y), chunk_size):
            chunk = y[i : i + chunk_size]

            # If chunk is shorter than chunk_size, pad with zeros
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), "constant")

            chunks.append(chunk)

        print(f"Split audio into {len(chunks)} chunks of {chunk_duration} seconds each")
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


def classify_audio_chunks(chunks):
    """
    Classify audio chunks using the MAEST model.

    Args:
        chunks: List of audio chunks as numpy arrays

    Returns:
        List of classification results for each chunk
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

    return results


def save_results_to_csv(results, output_file="classification_results.csv"):
    """
    Save classification results to a CSV file.

    Args:
        results: List of classification results for each chunk
        output_file: Path to the output CSV file
    """
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Chunk", "Rank", "Label", "Score"])

        for chunk_idx, chunk_results in enumerate(results):
            for rank, pred in enumerate(chunk_results):
                writer.writerow([chunk_idx + 1, rank + 1, pred["label"], pred["score"]])

    print(f"Saved classification results to {output_file}")


def process_youtube_link(youtube_url):
    """
    Process a YouTube link: download, process, and classify.

    Args:
        youtube_url: URL of the YouTube video

    Returns:
        List of classification results
    """
    # Create directories and get paths
    audio_chunks_dir, downloads_dir = create_directories()

    # Download audio
    audio_file = download_youtube_audio(youtube_url, output_path=downloads_dir)
    if not audio_file:
        return None

    # Process audio
    chunks = process_audio(audio_file)
    if not chunks:
        return None

    # Extract video ID for naming files
    try:
        video_id = youtube_url.split("v=")[1]
        if "&" in video_id:
            video_id = video_id.split("&")[0]
    except IndexError:
        import uuid

        video_id = str(uuid.uuid4())[:8]

    # Save chunks
    save_audio_chunks(
        chunks, output_dir=audio_chunks_dir, base_filename=f"youtube_{video_id}"
    )

    # Classify chunks
    results = classify_audio_chunks(chunks)

    # Save results to data directory
    data_dir = os.path.dirname(audio_chunks_dir)
    results_file = os.path.join(data_dir, f"classification_youtube_{video_id}.csv")
    save_results_to_csv(results, output_file=results_file)

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

    args = parser.parse_args()

    if args.url:
        process_youtube_link(args.url)
    elif args.csv:
        with open(args.csv, "r") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header row
            for row in reader:
                if row:  # Skip empty rows
                    url = row[0]
                    print(f"\nProcessing YouTube URL: {url}")
                    process_youtube_link(url)
    else:
        # Use URLs from the CSV file
        csv_path = Path(__file__).parent.parent / "data" / "taqsim_youtube_links.csv"
        if csv_path.exists():
            with open(csv_path, "r") as csvfile:
                reader = csv.reader(csvfile)
                try:
                    next(reader)  # Skip header row
                except StopIteration:
                    print("CSV file is empty.")
                    return

                processed = False
                for row in reader:
                    if row:  # Skip empty rows
                        url = row[0]
                        print(f"\nProcessing YouTube URL: {url}")
                        process_youtube_link(url)
                        processed = True
                        if not args.all:
                            break  # Process only the first URL by default

                if not processed:
                    print("No valid YouTube URLs found in the CSV file.")
        else:
            print("No YouTube URL provided and no CSV file found.")
            print("Please provide a YouTube URL with --url or a CSV file with --csv")


if __name__ == "__main__":
    main()
