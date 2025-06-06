"""
Taqsim Narrative Arc Visualization

This module provides a grid visualization of embeddings for each song/taqsim,
showing the narrative arc of each piece through its embedding trajectory.
"""

import os

import altair as alt
import pandas as pd
import streamlit as st
from embedding_visualizer_streamlit import (
    EMBEDDINGS_DIR_PATH,
    METADATA_CSV_PATH,
    get_metadata_from_csv,
    load_embeddings,
    prepare_embeddings_for_umap,
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP


def get_all_song_uuids_from_csv():
    """
    Get all unique song UUIDs from the metadata CSV.

    Returns:
        List of tuples (uuid, song_name, artist, maqam)
    """
    try:
        df = pd.read_csv(METADATA_CSV_PATH)
        if "uuid" not in df.columns:
            st.error("Metadata CSV is missing 'uuid' column.")
            return []

        # Get required columns, fill missing values
        df["song_name"] = df["song_name"].fillna("Unknown Song")
        df["artist"] = df["artist"].fillna("Unknown Artist")
        df["maqam"] = df["maqam"].fillna("Unknown Maqam")

        # Group by uuid to get unique songs
        unique_songs = []
        for uuid, group in df.groupby("uuid"):
            # Take the first row for each UUID
            row = group.iloc[0]
            unique_songs.append((uuid, row["song_name"], row["artist"], row["maqam"]))

        return unique_songs
    except Exception as e:
        st.error(f"Error reading song UUIDs from CSV: {e}")
        return []


def taqsim_narrative_arc_ui():
    """
    Streamlit UI for the taqsim narrative arc visualization
    """
    st.subheader("Taqsim Narrative Arc")

    # Add CSS to ensure all charts have the exact same width
    st.markdown(
        """
    <style>
    .stChart > div > div {
        width: 400px !important;
        height: 400px !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.write(
        "This visualization shows the narrative arc of each taqsim performance as a trajectory in the embedding space. "
        "Each plot represents one song, with points showing the 30-second segments in sequence."
    )

    # Sidebar options
    st.sidebar.title("Options")

    # Dimensionality reduction method
    reduction_method = st.sidebar.selectbox(
        "Dimensionality Reduction Method:",
        ["UMAP", "T-SNE", "PCA"],
        index=0,
    )

    # Embedding type
    embedding_type = st.sidebar.selectbox(
        "Embedding Type:",
        ["cls", "dist", "avg", "combined"],
        index=3,
    )

    # Exclude last chunk option
    exclude_last_chunk = st.sidebar.checkbox(
        "Exclude Last Segment",
        value=True,
    )

    # Get embeddings directory
    embeddings_dir = EMBEDDINGS_DIR_PATH
    if not os.path.exists(embeddings_dir):
        st.error(f"Embeddings directory does not exist: {embeddings_dir}")
        return

    # Load embeddings
    all_embeddings = load_embeddings(embeddings_dir)
    if not all_embeddings:
        st.error("No embeddings found.")
        return

    # Prepare embeddings for dimensionality reduction
    embeddings_array, video_ids, chunk_numbers = prepare_embeddings_for_umap(
        all_embeddings,
        embedding_type=embedding_type,
        exclude_last_chunk=exclude_last_chunk,
    )

    if len(embeddings_array) == 0:
        st.error("No valid embeddings found for the selected options.")
        return

    # Apply dimensionality reduction
    st.info(f"Applying {reduction_method} to {len(embeddings_array)} embeddings...")

    # Initialize the appropriate reducer
    if reduction_method == "UMAP":
        reducer = UMAP(n_components=2, random_state=42)
    elif reduction_method == "T-SNE":
        reducer = TSNE(n_components=2, random_state=42)
    else:  # PCA
        reducer = PCA(n_components=2)

    # Apply the reduction
    reduced_embeddings = reducer.fit_transform(embeddings_array)

    # Create a DataFrame with the reduced embeddings
    data = {
        "video_id": video_ids,
        "chunk_number": chunk_numbers,
        "x": reduced_embeddings[:, 0],
        "y": reduced_embeddings[:, 1],
    }
    all_data_df = pd.DataFrame(data)

    # Get unique video IDs from the reduced data
    unique_video_ids = list(all_data_df["video_id"].unique())

    # Get metadata for these video IDs
    metadata = get_metadata_from_csv(unique_video_ids)

    # Get unique songs with metadata
    all_songs = []
    for video_id in all_data_df["video_id"].unique():
        found = False
        for uuid, song_data in metadata.items():
            if uuid == video_id:
                song_name = song_data.get("song_name", f"Unknown ({uuid[:8]})")
                artist = song_data.get("artist", "Unknown")
                maqam = song_data.get("maqam", "Unknown")
                all_songs.append((uuid, song_name, artist, maqam))
                found = True
                break
        if not found:
            all_songs.append(
                (video_id, f"Unknown ({video_id[:8]})", "Unknown", "Unknown")
            )

    # Sort songs by name
    all_songs.sort(key=lambda x: x[1])

    # Use all songs for display
    songs_to_display = all_songs

    # Progress bar
    progress_bar = st.progress(0)

    # We'll use individual scales for each chart instead of global scales

    # Create a grid layout using Streamlit columns instead of Altair concatenation
    # Determine how many columns to use
    cols_per_row = 1

    # Group songs into rows
    for i in range(0, len(songs_to_display), cols_per_row):
        # Create a row with the specified number of columns
        cols = st.columns(cols_per_row)

        # Fill each column with a chart
        for col_idx in range(cols_per_row):
            song_idx = i + col_idx
            if song_idx < len(songs_to_display):
                # Get song data
                uuid, song_name, artist, maqam = songs_to_display[song_idx]

                # Filter data for this song
                song_df = all_data_df[all_data_df["video_id"] == uuid].copy()

                if len(song_df) == 0:
                    continue

                # Sort by chunk number
                song_df = song_df.sort_values("chunk_number")

                # Calculate individual min/max for this song's data points
                x_min, x_max = song_df["x"].min(), song_df["x"].max()
                y_min, y_max = song_df["y"].min(), song_df["y"].max()

                # Add a small padding (5%) to the ranges for better visualization
                x_padding = (x_max - x_min) * 0.05 if x_max > x_min else 0.1
                y_padding = (y_max - y_min) * 0.05 if y_max > y_min else 0.1

                # Create scatter plot with no color encoding - using a single color
                scatter = (
                    alt.Chart(song_df)
                    .mark_circle(size=100, color="#1f77b4")
                    .encode(
                        x=alt.X(
                            "x",
                            scale=alt.Scale(
                                domain=[x_min - x_padding, x_max + x_padding], nice=True
                            ),
                        ),
                        y=alt.Y(
                            "y",
                            scale=alt.Scale(
                                domain=[y_min - y_padding, y_max + y_padding], nice=True
                            ),
                        ),
                        tooltip=["chunk_number", "x", "y"],
                    )
                )

                # Create lines with the same axis scales
                lines = (
                    alt.Chart(song_df)
                    .mark_line(opacity=0.5)
                    .encode(
                        x=alt.X(
                            "x",
                            scale=alt.Scale(
                                domain=[x_min - x_padding, x_max + x_padding], nice=True
                            ),
                        ),
                        y=alt.Y(
                            "y",
                            scale=alt.Scale(
                                domain=[y_min - y_padding, y_max + y_padding], nice=True
                            ),
                        ),
                        order="chunk_number:Q",
                    )
                )

                # Create text labels positioned above the dots
                text = (
                    alt.Chart(song_df)
                    .mark_text(
                        fontSize=10,
                        dy=-10,  # Offset text upward by 10 pixels
                        align="center",
                        baseline="bottom",
                    )
                    .encode(
                        x=alt.X(
                            "x",
                            scale=alt.Scale(
                                domain=[x_min - x_padding, x_max + x_padding], nice=True
                            ),
                        ),
                        y=alt.Y(
                            "y",
                            scale=alt.Scale(
                                domain=[y_min - y_padding, y_max + y_padding], nice=True
                            ),
                        ),
                        text="chunk_number",
                    )
                )

                # Combine layers
                chart = (scatter + lines + text).properties(
                    width=400,
                    height=400,
                    title=f"{song_name}\n{artist} | {maqam}\n{uuid[:8]}",
                )

                # Display in the appropriate column
                with cols[col_idx]:
                    st.altair_chart(chart)

                # Update progress bar
                current_progress = min(0.999, (song_idx + 1) / len(songs_to_display))
                progress_bar.progress(current_progress)

    # Complete the progress
    progress_bar.progress(1.0)
    progress_bar.empty()


if __name__ == "__main__":
    taqsim_narrative_arc_ui()
