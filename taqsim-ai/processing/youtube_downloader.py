"""
YouTube Downloader Module

This module provides functionality for downloading audio from YouTube videos
using yt-dlp with appropriate error handling and file management.

Can be run as a standalone script to download YouTube audio from a CSV file.
"""

import os

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

    # Ensure output_path is provided
    if output_path is None:
        raise ValueError("output_path must be provided")

    try:
        # Check if the file already exists
        for file in os.listdir(output_path):
            if uuid in file:
                existing_file = os.path.join(output_path, file)
                print(f"File already exists: {existing_file}")
                return (
                    existing_file,
                    False,
                )  # False indicates file was not newly downloaded

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
                return found_file, True  # True indicates file was newly downloaded

            raise Exception(f"Downloaded file not found in {output_path}")

    except Exception as e:
        print(f"Error downloading YouTube video: {e}")
        print("This could be due to YouTube API changes or restrictions.")
        print("Check if the video is region-restricted or age-restricted")
        return None, False


# CSV processing has been moved to pipeline.py
