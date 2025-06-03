#!/usr/bin/env python3
"""
Audio Processing Pipeline

This script orchestrates the full audio processing pipeline:
1. Downloads YouTube audio using UUIDs
2. Removes silence from the beginning and end of audio files
3. Processes audio for classification (optional)

The pipeline processes a CSV file with 'link' and 'uuid' columns.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from silence_remover import create_processed_directory, process_audio_file
from youtube_downloader import create_download_directory, download_youtube_audio


def process_single_item(youtube_url, uuid, downloads_dir, processed_dir):
    """
    Process a single YouTube URL through the pipeline:
    1. Download audio
    2. Remove silence

    Args:
        youtube_url: URL of the YouTube video
        uuid: Unique identifier for the file
        downloads_dir: Directory to save downloaded audio
        processed_dir: Directory to save processed audio

    Returns:
        Tuple of (downloaded_file_path, processed_file_path) or (None, None) if failed
    """
    print(f"\nProcessing item with UUID: {uuid}")
    print(f"YouTube URL: {youtube_url}")

    # Step 1: Download audio
    downloaded_file, is_new_download = download_youtube_audio(
        youtube_url, output_path=downloads_dir, uuid=uuid
    )
    if not downloaded_file:
        print(f"Failed to download audio for UUID: {uuid}")
        return None, None

    # Report appropriate message based on whether file was newly downloaded or already existed
    if is_new_download:
        print(f"Successfully downloaded: {downloaded_file}")
    else:
        print(f"Using existing file: {downloaded_file}")

    # Step 2: Remove silence
    file_ext = os.path.splitext(downloaded_file)[1]
    processed_file = os.path.join(processed_dir, f"{uuid}{file_ext}")

    success = process_audio_file(downloaded_file, processed_file)
    if not success:
        print(f"Failed to process audio for UUID: {uuid}")
        return downloaded_file, None

    print(f"Successfully processed: {processed_file}")
    return downloaded_file, processed_file


def process_csv_file(
    csv_file, downloads_dir=None, processed_dir=None, process_all=False
):
    """
    Process all items in a CSV file through the pipeline.

    Args:
        csv_file: Path to the CSV file containing YouTube URLs and UUIDs
        downloads_dir: Directory to save downloaded audio
        processed_dir: Directory to save processed audio
        process_all: Whether to process all items or just the first one

    Returns:
        List of successfully processed items
    """
    try:
        # Create directories if not provided
        if downloads_dir is None:
            downloads_dir = create_download_directory()

        if processed_dir is None:
            processed_dir = create_processed_directory()

        # Read the CSV file
        df = pd.read_csv(csv_file)

        if df.empty:
            print("CSV file is empty.")
            return []

        # Check if required columns exist
        if "link" not in df.columns:
            print("Error: CSV file must contain a 'link' column with YouTube URLs.")
            return []

        if "uuid" not in df.columns:
            print(
                "Error: CSV file must contain a 'uuid' column with unique identifiers."
            )
            return []

        # Process each row
        processed_items = []

        # If process_all is True, process all rows. Otherwise, just take the first row.
        rows_to_process = (
            df.iterrows() if process_all else [next(df.iterrows(), (None, None))]
        )

        for index, row in rows_to_process:
            # Skip if we somehow got a None row (should only happen if df is empty)
            if row is None:
                break

            try:
                url = row["link"]
                uuid = row["uuid"]

                # Skip rows with missing UUID
                if pd.isna(uuid) or not uuid:
                    print(f"\nSkipping row {index}: Missing UUID for URL {url}")
                    continue

                # Add song name info if available
                song_info = (
                    f" - {row['song_name']}"
                    if "song_name" in df.columns and not pd.isna(row["song_name"])
                    else ""
                )
                print(f"\nProcessing YouTube URL: {url}{song_info} (UUID: {uuid})")

                # Process the item
                downloaded_file, processed_file = process_single_item(
                    url, uuid, downloads_dir, processed_dir
                )

                if processed_file:
                    processed_items.append(
                        {
                            "uuid": uuid,
                            "url": url,
                            "downloaded_file": downloaded_file,
                            "processed_file": processed_file,
                        }
                    )
                else:
                    print(f"Failed to process item with UUID: {uuid}")

            except Exception as e:
                print(f"Error processing row {index}: {e}")
                continue

        # If we only processed one item, let the user know how to process all
        if not process_all:
            print(
                "Processed first item only. Use --process-mode=all to process all items."
            )

        return processed_items

    except Exception as e:
        print(f"Error processing CSV file: {e}")
        return []


def main():
    """
    Main function to parse arguments and run the pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Run the full audio processing pipeline: download, silence removal, and processing."
    )
    parser.add_argument(
        "--csv", help="Path to CSV file containing YouTube URLs and UUIDs"
    )
    parser.add_argument("--url", help="Single YouTube URL to process")
    parser.add_argument("--uuid", help="UUID for the single URL (required with --url)")
    parser.add_argument(
        "--downloads-dir",
        help="Directory to save downloaded audio files",
    )
    parser.add_argument(
        "--processed-dir",
        help="Directory to save processed audio files",
    )
    parser.add_argument(
        "--process-mode",
        choices=["first", "all"],
        default="first",
        help="Whether to process only the first item or all items in the CSV (default: first)",
    )

    args = parser.parse_args()

    # Create directories if specified
    downloads_dir = None
    if args.downloads_dir:
        downloads_dir = args.downloads_dir
        os.makedirs(downloads_dir, exist_ok=True)
    else:
        downloads_dir = create_download_directory()

    processed_dir = None
    if args.processed_dir:
        processed_dir = args.processed_dir
        os.makedirs(processed_dir, exist_ok=True)
    else:
        processed_dir = create_processed_directory()

    if args.url:
        # Process a single URL
        if not args.uuid:
            print("Error: --uuid parameter is required when using --url")
            return

        process_single_item(args.url, args.uuid, downloads_dir, processed_dir)
    elif args.csv:
        # Process the CSV file
        process_csv_file(
            args.csv,
            downloads_dir=downloads_dir,
            processed_dir=processed_dir,
            process_all=(args.process_mode == "all"),
        )
    else:
        # Use default CSV file if it exists
        default_csv_path = Path(__file__).parent.parent / "data" / "taqsim_ai.csv"
        if default_csv_path.exists():
            print(f"Using default CSV file: {default_csv_path}")
            print(f"Processing mode: {args.process_mode}")
            process_csv_file(
                default_csv_path,
                downloads_dir=downloads_dir,
                processed_dir=processed_dir,
                process_all=(args.process_mode == "all"),
            )
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
