import os  # Added import

import altair as alt  # Added for future use with Altair charts
import pandas as pd
import streamlit as st


def data_explorer_page():
    st.subheader("Data Explorer")
    st.write("This page allows you to explore the taqsim dataset metadata.")

    # Path to the CSV file, relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assumes 'data' folder is in the same directory as this script.
    csv_path = os.path.join(script_dir, "data", "taqsim_ai.csv")

    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)

        # Display basic info about the dataset
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Taqsim", df.shape[0])
        with col2:
            if "artist" in df.columns:
                st.metric("Unique Artists", df["artist"].nunique())
            else:
                st.metric("Unique Artists", "N/A")
        with col3:
            if "maqam" in df.columns:
                st.metric("Unique Maqams", df["maqam"].nunique())
            else:
                st.metric("Unique Maqams", "N/A")

        # Add search/filter functionality
        st.subheader("Filter Data")
        search_term = st.text_input("Search by any field:")

        # Filter the dataframe if search term is provided
        if search_term:
            filtered_df = df[
                df.astype(str).apply(
                    lambda row: row.str.contains(search_term, case=False).any(), axis=1
                )
            ]
        else:
            filtered_df = df

        # Display the filtered dataframe
        st.dataframe(filtered_df, use_container_width=True)

        # Check if relevant columns exist for visualizations
        if "artist" in df.columns:
            # Artist distribution
            st.write("Artist Distribution")
            artist_counts_series = filtered_df["artist"].value_counts().head(10)
            plot_df_artist = artist_counts_series.reset_index()
            # plot_df_artist will have columns named 'artist' (from original series index) and 'count' (name of value_counts series)
            chart_artist = (
                alt.Chart(plot_df_artist)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "artist:N",
                        sort=alt.EncodingSortField(
                            field="count", op="sum", order="descending"
                        ),
                    ),
                    y=alt.Y("count:Q", title="Number of Taqasim"),
                    tooltip=["artist", "count"],
                )
                .properties(title="Top 10 Artists by Number of Taqasim")
            )
            st.altair_chart(chart_artist, use_container_width=True)

        if "maqam" in df.columns:
            # Maqam distribution
            st.write("Maqam Distribution")
            maqam_counts_series = filtered_df["maqam"].value_counts()
            plot_df_maqam = maqam_counts_series.reset_index()
            # plot_df_maqam will have columns named 'maqam' and 'count'
            chart_maqam = (
                alt.Chart(plot_df_maqam)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "maqam:N",
                        sort=alt.EncodingSortField(
                            field="count", op="sum", order="descending"
                        ),
                    ),
                    y=alt.Y("count:Q", title="Number of Taqasim"),
                    tooltip=["maqam", "count"],
                )
                .properties(title="Maqam Distribution by Number of Taqasim")
            )
            st.altair_chart(chart_maqam, use_container_width=True)

    except Exception as e:
        st.error(f"Error reading the CSV file: {e}")
        st.info(f"Make sure the file exists at: {csv_path}")

        # Show path to expected CSV file location
        st.code(csv_path)
