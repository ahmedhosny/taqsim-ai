import streamlit as st
from data_explorer_ui import data_explorer_page
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
    data_explorer_page()


# Embeddings Visualization page
elif page == "Embeddings Visualization":
    st.header("Embeddings Visualization")
    st.write(
        "This page will provide interactive visualizations of audio embeddings."
        "Each dot represent a 30 second segments of the taqsim, with the numbering representing"
        " the segment order."
    )
    embedding_visualizer_ui()


# Footer
st.markdown("---")
st.markdown("© 2025 Taqsim x AI Project")
