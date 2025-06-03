import pandas as pd
import streamlit as st
from embedding_visualizer_streamlit import embedding_visualizer_ui

# Set page configuration
st.set_page_config(
    page_title="Taqsim x AI",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add a title and description
st.title("Taqsim x AI")


# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    [
        "Home",
        "Data Explorer",
        "Embeddings Visualization",
    ],
)

# Home page
if page == "Home":
    # Display project overview
    st.subheader("Project Overview")
    st.markdown("""
    This dashboard provides interactive visualizations and tools for exploring the Taqsim AI project.
    Use the sidebar to navigate between different features and analysis tools.
    """)
    st.write("""
    Taqsim x AI is a project focused on analyzing and understanding Arabic music through machine learning.
    This dashboard provides tools to explore audio data, visualize embeddings, and analyze musical patterns.
    """)


# Data Explorer page
elif page == "Data Explorer":
    st.subheader("Data Explorer")
    st.write("This page allows you to explore the taqsim dataset metadata.")

    # Path to the CSV file
    csv_path = "/Users/ahmedhosny/taqsim-ai/data/taqsim_ai.csv"

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
            artist_counts = filtered_df["artist"].value_counts().head(10)
            st.bar_chart(artist_counts)

        if "maqam" in df.columns:
            # Maqam distribution
            st.write("Maqam Distribution")
            maqam_counts = filtered_df["maqam"].value_counts()
            st.bar_chart(maqam_counts)

    except Exception as e:
        st.error(f"Error reading the CSV file: {e}")
        st.info(f"Make sure the file exists at: {csv_path}")

        # Show path to expected CSV file location
        st.code(csv_path)


# Embeddings Visualization page
elif page == "Embeddings Visualization":
    st.header("Embeddings Visualization")
    st.write("This page will provide interactive visualizations of audio embeddings.")
    embedding_visualizer_ui()


# Footer
st.markdown("---")
st.markdown("© 2025 Taqsim x AI Project")
