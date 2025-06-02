"""
YouTube Downloader Module

This module provides functionality for downloading audio from YouTube videos
using yt-dlp with appropriate error handling and file management.

Can be run as a standalone script to download YouTube audio from a CSV file.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import yt_dlp


def create_download_directory(output_path=None):
    """
    Create the downloads directory if it doesn't exist.

    Args:
        output_path: Optional custom path for downloads directory

    Returns:
        Path to the downloads directory
    """
    if output_path is None:
        # Get the path to the data directory
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
        )
        # Create downloads directory
        output_path = os.path.join(data_dir, "downloads")

    os.makedirs(output_path, exist_ok=True)
    return output_path


def check_ffmpeg_availability():
    """
    Check if ffmpeg is available on the system.

    Returns:
        Tuple of (ffmpeg_available, ffmpeg_path)
    """
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

    return ffmpeg_available, ffmpeg_path


def download_youtube_audio(youtube_url, output_path=None, uuid=None):
    """
    Download audio from a YouTube video using yt-dlp.
    If the video is already downloaded, it will not download it again.

    Args:
        youtube_url: URL of the YouTube video
        output_path: Directory to save the downloaded audio. If None, uses data/downloads
        uuid: Required unique identifier for the file. Will be used for naming the output file.

    Returns:
        Path to the downloaded audio file

    Raises:
        ValueError: If uuid is not provided
    """
    # Require UUID parameter
    if not uuid:
        raise ValueError(
            "UUID is required for downloading. Cannot proceed without a UUID."
        )

    print(f"Processing YouTube URL: {youtube_url} with UUID: {uuid}")

    # Use the downloads directory if no output path is specified
    if output_path is None:
        output_path = create_download_directory()

    try:
        # Check if the file is already downloaded
        for file in os.listdir(output_path):
            if uuid in file:
                existing_file = os.path.join(output_path, file)
                print(f"File already downloaded: {existing_file}")
                return existing_file

        # Check for ffmpeg availability
        ffmpeg_available, ffmpeg_path = check_ffmpeg_availability()

        # Set up yt-dlp options with UUID for naming
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(output_path, f"{uuid}.%(ext)s"),
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

            # Check for the downloaded file
            found_file = None

            # Check for the expected filename with wav extension if ffmpeg was available
            if ffmpeg_available:
                expected_file = os.path.join(output_path, f"{uuid}.wav")
                if os.path.exists(expected_file):
                    found_file = expected_file

            # If not found, search for any file with the UUID in the name
            if not found_file:
                for file in os.listdir(output_path):
                    if uuid in file:
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


def process_csv_for_downloads(csv_path, output_path=None, process_all=True):
    """
    Process a CSV file containing YouTube URLs and download audio for each URL.

    Args:
        csv_path: Path to the CSV file containing YouTube URLs
        output_path: Directory to save the downloaded audio files
        process_all: Whether to process all URLs in the CSV file or just the first one

    Returns:
        List of paths to the downloaded audio files
    """
    if output_path is None:
        output_path = create_download_directory()

    downloaded_files = []

    try:
        # Use pandas to read the CSV file
        df = pd.read_csv(csv_path)

        if df.empty:
            print("CSV file is empty.")
            return downloaded_files

        # Check if required columns exist
        if "link" not in df.columns:
            print("Error: CSV file must contain a 'link' column with YouTube URLs.")
            return downloaded_files

        if "uuid" not in df.columns:
            print(
                "Error: CSV file must contain a 'uuid' column with unique identifiers."
            )
            return downloaded_files

        # Process each row (or just the first if process_all is False)
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

                # Download the audio using UUID
                audio_file = download_youtube_audio(
                    url, output_path=output_path, uuid=uuid
                )
                if audio_file:
                    downloaded_files.append(audio_file)
            except ValueError as e:
                print(f"Error processing row {index}: {e}")
                continue

            if not process_all:
                break  # Process only the first URL if process_all is False

        print(f"\nDownloaded {len(downloaded_files)} files.")
        return downloaded_files

    except Exception as e:
        print(f"Error processing CSV file: {e}")
        return downloaded_files


def main():
    """
    Main function to download YouTube audio from command line or CSV file.
    """
    parser = argparse.ArgumentParser(description="Download audio from YouTube videos")
    parser.add_argument("--url", help="YouTube URL to download")
    parser.add_argument("--csv", help="CSV file containing YouTube URLs")
    parser.add_argument("--output", help="Directory to save the downloaded audio files")
    parser.add_argument(
        "--all", action="store_true", help="Process all URLs in the CSV file"
    )

    args = parser.parse_args()

    # Create output directory if specified
    output_path = None
    if args.output:
        output_path = args.output
        os.makedirs(output_path, exist_ok=True)
    else:
        output_path = create_download_directory()

    if args.url:
        # Download a single URL
        audio_file = download_youtube_audio(args.url, output_path=output_path)
        if audio_file:
            print(f"\nSuccessfully downloaded: {audio_file}")
    elif args.csv:
        # Process the CSV file
        process_csv_for_downloads(
            args.csv, output_path=output_path, process_all=args.all
        )
    else:
        # Use default CSV file if it exists
        default_csv_path = Path(__file__).parent.parent / "data" / "taqsim_ai.csv"
        if default_csv_path.exists():
            print(f"Using default CSV file: {default_csv_path}")
            process_csv_for_downloads(
                default_csv_path, output_path=output_path, process_all=args.all
            )
        else:
            print("No YouTube URL provided and no CSV file found.")
            print("Please provide a YouTube URL with --url or a CSV file with --csv")


if __name__ == "__main__":
    main()
