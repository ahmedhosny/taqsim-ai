"""
Normalize embeddings at the artist level.

This script:
1. Loads embeddings from the original directory
2. Computes the mean embedding for each artist
3. Subtracts this mean from all embeddings of that artist's songs
4. Saves the normalized embeddings to a new directory
"""

import glob
import os

import numpy as np

# Import functions from embedding_visualizer_streamlit
from embedding_visualizer_streamlit import (
    EMBEDDINGS_DIR_PATH,
    get_metadata_from_csv,
    load_embeddings,
)

# Define the output directory for normalized embeddings
NORMALIZED_EMBEDDINGS_DIR = (
    "/Users/ahmedhosny/taqsim-ai/taqsim-ai/data/normalized_embeddings"
)


def normalize_embeddings_by_artist(all_embeddings, metadata):
    """
    Normalize embeddings by artist by subtracting the mean embedding for each artist.

    Args:
        all_embeddings: Dictionary of embeddings by video ID
        metadata: Dictionary of metadata by video ID

    Returns:
        Dictionary of normalized embeddings by video ID
    """
    print("Normalizing embeddings by artist...")

    # Create a mapping from video_id to artist
    video_to_artist = {}
    for video_id in all_embeddings.keys():
        if video_id in metadata:
            artist = metadata[video_id].get("artist", "Unknown")
        else:
            artist = "Unknown"
        video_to_artist[video_id] = artist

    # Group embeddings by artist
    artist_embeddings = {}
    for video_id, embeddings in all_embeddings.items():
        artist = video_to_artist.get(video_id, "Unknown")
        if artist not in artist_embeddings:
            artist_embeddings[artist] = []

        # Add all chunk embeddings for this video to the artist's collection
        for chunk_num, embedding in embeddings.items():
            # In the new structure, embedding is a numpy array, not a dictionary
            artist_embeddings[artist].append(embedding)

    # Compute mean embedding for each artist
    artist_means = {}
    for artist, embeddings_list in artist_embeddings.items():
        if embeddings_list:
            # Stack embeddings and compute mean
            stacked_embeddings = np.vstack(embeddings_list)
            artist_means[artist] = np.mean(stacked_embeddings, axis=0)

    # Create normalized embeddings
    normalized_embeddings = {}
    for video_id, embeddings in all_embeddings.items():
        artist = video_to_artist.get(video_id, "Unknown")
        if artist in artist_means:
            artist_mean = artist_means[artist]

            # Create a new dictionary for this video
            normalized_embeddings[video_id] = {}

            # Normalize each chunk embedding
            for chunk_num, embedding in embeddings.items():
                # Subtract the artist mean from the embedding
                normalized_embeddings[video_id][chunk_num] = embedding - artist_mean
        else:
            # If no mean was computed (e.g., only one embedding), keep original
            normalized_embeddings[video_id] = embeddings

    return normalized_embeddings


def save_normalized_embeddings(normalized_embeddings):
    """
    Save normalized embeddings to the output directory.

    Args:
        normalized_embeddings: Dictionary of normalized embeddings by video ID
    """
    print(f"Saving normalized embeddings to {NORMALIZED_EMBEDDINGS_DIR}...")

    # Create output directory if it doesn't exist
    os.makedirs(NORMALIZED_EMBEDDINGS_DIR, exist_ok=True)

    # Find all original embedding files to get their exact filenames
    original_files = {}
    for file_path in glob.glob(os.path.join(EMBEDDINGS_DIR_PATH, "*.npz")):
        basename = os.path.basename(file_path)
        parts = basename.split("_")
        if len(parts) >= 2:
            # The first part is the UUID
            video_id = parts[0]
            # The second part is the chunk number
            chunk_num = parts[1]
            chunk_key = f"chunk_{chunk_num}"
            # Store the mapping from (video_id, chunk_key) to the full filename
            original_files[(video_id, chunk_key)] = basename

    # Count of saved files
    saved_count = 0

    # Save each chunk's embedding to a separate NPZ file with the exact same filename
    for video_id, chunks in normalized_embeddings.items():
        for chunk_key, embedding in chunks.items():
            # Look up the original filename
            if (video_id, chunk_key) in original_files:
                original_filename = original_files[(video_id, chunk_key)]
                output_file = os.path.join(NORMALIZED_EMBEDDINGS_DIR, original_filename)
            else:
                # Fallback if original filename not found
                chunk_number = (
                    chunk_key.split("_")[1] if "_" in chunk_key else chunk_key
                )
                output_file = os.path.join(
                    NORMALIZED_EMBEDDINGS_DIR, f"{video_id}_{chunk_number}.npz"
                )

            # Save as NPZ file with 'embedding' key to match the original format
            np.savez(output_file, embedding=embedding)
            saved_count += 1

    print(
        f"Saved {saved_count} normalized embedding files for {len(normalized_embeddings)} videos."
    )


def main():
    """
    Main function to normalize and save embeddings.
    """
    print("Loading embeddings...")
    all_embeddings = load_embeddings(EMBEDDINGS_DIR_PATH)
    if not all_embeddings:
        print("No embeddings found.")
        return

    print(f"Loaded embeddings for {len(all_embeddings)} videos.")

    # Get metadata for all videos
    video_ids = list(all_embeddings.keys())
    print("Loading metadata...")
    metadata = get_metadata_from_csv(video_ids)
    print(f"Loaded metadata for {len(metadata)} videos.")

    # Normalize embeddings by artist
    normalized_embeddings = normalize_embeddings_by_artist(all_embeddings, metadata)

    # Save normalized embeddings
    save_normalized_embeddings(normalized_embeddings)

    print("Done!")


if __name__ == "__main__":
    main()
