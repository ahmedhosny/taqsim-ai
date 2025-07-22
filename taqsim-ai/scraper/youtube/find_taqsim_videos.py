#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Find Taqsim Videos Script

This script searches YouTube for oud taqsim videos and saves the results in a CSV file
with the same format as taqsim_ai.csv. It filters videos based on specific criteria:
- Title contains "taqsim", "taksim", "تقاسيم", "تقسيم", or similar terms
- Optionally contains maqam name in title
- Only contains oud (instrumental, no other instruments or crowd noise)

Requirements:
- Google API client library: pip install google-api-python-client
- A YouTube Data API key (https://developers.google.com/youtube/v3/getting-started)

Usage:
python find_taqsim_videos.py --api-key YOUR_API_KEY --output new_taqsim_videos.csv --limit 200
"""

import argparse
import csv
import os
import re
import uuid

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# List of maqams to look for in video titles
MAQAMS = [
    "Bayati",
    "Rast",
    "Hijaz",
    "Nahawand",
    "Saba",
    "Kurd",
    "Ajam",
    "Sikah",
    "Husseini",
    "Huzam",
    "Jiharkah",
    "Nawa Athar",
    "Nikriz",
    "Segah",
    "Suzidil",
    "Shahnaz",
    "Bastanikar",
    "Farahfaza",
    "Suznak",
    "Mahur",
    "Ushaq",
    "Sazkar",
    "Rahat Al Arwah",
    "Hijaz Humayun",
    "Zanjaran",
    "Awj",
    "Mukhalif",
    "Nairuz",
    "Athar Kurd",
    "Shawq Afza",
]

# List of common oud performers
OUD_PERFORMERS = [
    "Naseer Shamma",
    "Munir Bashir",
    "Simon Shaheen",
    "Marcel Khalife",
    "Anouar Brahem",
    "Le Trio Joubran",
    "Dhafer Youssef",
    "Omar Bashir",
    "Rabih Abou-Khalil",
    "Charbel Rouhana",
    "Driss El Maloumi",
    "Necati Çelik",
    "Yurdal Tokcan",
    "Ara Dinkjian",
    "Mehmet Bitmez",
    "Khaled Arman",
    "Kenan Adnawi",
    "Haytham Safia",
    "Nizar Rohana",
    "Yair Dalal",
    "Erkan Oğur",
    "Samir Joubran",
    "Wissam Joubran",
    "Adel Salameh",
    "Issa Boulos",
    "Khaled Jubran",
    "Nabil Khemir",
    "Taiseer Elias",
    "Wisam Gibran",
    "Khyam Allami",
    "Mehmet Emin Bitmez",
    "Hesham Hamra",
    "Baha Yetkin",
]


def extract_maqam_from_title(title):
    """Extract maqam name from video title if present."""
    title_lower = title.lower()
    for maqam in MAQAMS:
        if maqam.lower() in title_lower:
            return maqam
    return "Unknown"


def extract_artist_from_title(title, channel_title=None):
    """Extract artist name from video title if present.

    First tries to match against known performers, then attempts to extract
    artist names using common patterns in video titles.
    If no match is found, uses the channel name as a fallback.

    Args:
        title: The video title
        channel_title: The YouTube channel name that posted the video

    Returns:
        The extracted artist name or channel name as fallback
    """
    # First check against our known list of performers
    for artist in OUD_PERFORMERS:
        if artist.lower() in title.lower():
            return artist

    # Try to extract artist name using common patterns
    # Pattern: "Artist Name - Taqsim"
    dash_split = title.split(" - ")
    if len(dash_split) > 1:
        potential_artist = dash_split[0].strip()
        # Avoid returning the entire title or very long strings
        if 3 <= len(potential_artist.split()) <= 5:
            return potential_artist

    # Pattern: "Taqsim by Artist Name"
    by_match = re.search(
        r"(?:by|performed by|playing by)\s+([^,|.]+)", title, re.IGNORECASE
    )
    if by_match:
        return by_match.group(1).strip()

    # Use channel name as fallback if available
    if channel_title and channel_title.strip():
        # Clean up channel name (remove common YouTube channel suffixes)
        channel_name = re.sub(
            r"\s*(?:Official|Channel|Music|TV|HD)\s*$",
            "",
            channel_title,
            flags=re.IGNORECASE,
        )
        return channel_name.strip()

    # If we can't determine the artist, return "Other"
    return "Other"


def determine_type(title):
    """Determine if the performance is Arabic, Turkish, etc."""
    title_lower = title.lower()
    if "turkish" in title_lower or "türk" in title_lower or "taksim" in title_lower:
        return "Turkish"
    elif "arabic" in title_lower or "عربي" in title_lower or "taqsim" in title_lower:
        return "Arabic"
    elif "armenian" in title_lower:
        return "Armenian"
    elif "persian" in title_lower or "iranian" in title_lower:
        return "Persian"
    else:
        return "Unknown"


def is_electric(title):
    """Determine if an electric oud is used based on title."""
    title_lower = title.lower()
    if "electric" in title_lower or "elektro" in title_lower:
        return "yes"
    return "no"


def is_vintage(title):
    """Determine if it's a vintage recording based on title."""
    title_lower = title.lower()
    if (
        "vintage" in title_lower
        or "old" in title_lower
        or "classic" in title_lower
        or "1920" in title_lower
        or "1930" in title_lower
        or "1940" in title_lower
        or "1950" in title_lower
        or "1960" in title_lower
    ):
        return "yes"
    return "no"


def search_youtube_videos(api_key, search_terms, max_results=200):
    """Search YouTube for videos matching the search terms."""
    youtube = build("youtube", "v3", developerKey=api_key)

    all_videos = []
    next_page_token = None

    # Keep track of video IDs we've already seen to avoid duplicates
    seen_video_ids = set()

    # Continue searching until we have enough results or no more pages
    while len(all_videos) < max_results:
        # Prepare search request
        search_request = youtube.search().list(
            q=search_terms,
            part="snippet",
            type="video",
            maxResults=50,  # Maximum allowed by API
            pageToken=next_page_token,
        )

        try:
            search_response = search_request.execute()

            # Process search results
            for item in search_response.get("items", []):
                video_id = item["id"]["videoId"]

                # Skip if we've already seen this video
                if video_id in seen_video_ids:
                    continue

                seen_video_ids.add(video_id)

                video_info = {
                    "id": video_id,
                    "title": item["snippet"]["title"],
                    "channel_title": item["snippet"]["channelTitle"],
                    "published_at": item["snippet"]["publishedAt"],
                }

                all_videos.append(video_info)

                # Stop if we have enough videos
                if len(all_videos) >= max_results:
                    break

            # Check if there are more pages
            next_page_token = search_response.get("nextPageToken")
            if not next_page_token:
                break

        except HttpError as e:
            print(f"An HTTP error occurred: {e}")
            break

    return all_videos[:max_results]


def filter_existing_videos(new_videos, existing_csv_path):
    """Filter out videos that already exist in the taqsim_ai.csv file."""
    existing_links = set()

    # Read existing CSV file if it exists
    if os.path.exists(existing_csv_path):
        with open(existing_csv_path, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if "link" in row and row["link"]:
                    # Extract video ID from YouTube URL
                    match = re.search(
                        r"(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\?]+)", row["link"]
                    )
                    if match:
                        existing_links.add(match.group(1))

    # Filter out videos that already exist
    filtered_videos = []
    for video in new_videos:
        if video["id"] not in existing_links:
            filtered_videos.append(video)

    return filtered_videos


def save_to_csv(videos, output_path):
    """Save the videos to a CSV file in the same format as taqsim_ai.csv."""
    with open(output_path, "w", encoding="utf-8", newline="") as csvfile:
        fieldnames = [
            "notes",
            "uuid",
            "link",
            "song_name",
            "artist",
            "maqam",
            "type",
            "electric",
            "vintage",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for video in videos:
            title = video["title"]
            channel_title = video["channel_title"]

            writer.writerow(
                {
                    "notes": "",  # Empty notes field
                    "uuid": str(uuid.uuid4()),  # Generate a random UUID
                    "link": f"https://www.youtube.com/watch?v={video['id']}",
                    "song_name": title,
                    "artist": extract_artist_from_title(title, channel_title),
                    "maqam": extract_maqam_from_title(title),
                    "type": determine_type(title),
                    "electric": is_electric(title),
                    "vintage": is_vintage(title),
                }
            )


def main():
    parser = argparse.ArgumentParser(
        description="Search YouTube for oud taqsim videos."
    )
    parser.add_argument("--api-key", required=True, help="YouTube Data API key")
    parser.add_argument(
        "--output", default="new_taqsim_videos.csv", help="Output CSV file path"
    )
    parser.add_argument(
        "--limit", type=int, default=500, help="Maximum number of videos to find"
    )
    parser.add_argument(
        "--existing",
        default="../../data/taqsim_ai.csv",
        help="Path to existing taqsim_ai.csv file",
    )

    args = parser.parse_args()

    # Search terms to find oud taqsim videos
    search_terms = "oud taqsim OR oud taksim OR تقاسيم عود OR تقسيم عود OR taqsim oud OR taksim oud OR تقاسيم عود OR تقسيم عود solo -ensemble -orchestra -concert"

    print(f"Searching YouTube for '{search_terms}'...")
    videos = search_youtube_videos(args.api_key, search_terms, args.limit)
    print(f"Found {len(videos)} videos.")

    # Filter out videos that already exist in taqsim_ai.csv
    existing_csv_path = os.path.join(os.path.dirname(args.output), args.existing)
    filtered_videos = filter_existing_videos(videos, existing_csv_path)
    print(
        f"After filtering out existing videos, {len(filtered_videos)} new videos remain."
    )

    # Save the results to CSV
    save_to_csv(filtered_videos, args.output)
    print(f"Saved {len(filtered_videos)} videos to {args.output}")


if __name__ == "__main__":
    main()
