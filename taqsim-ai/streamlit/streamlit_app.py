import streamlit as st
from data_explorer_ui import data_explorer_page
from embedding_visualizer_streamlit import embedding_visualizer_ui
from homepage import render_home_page
from taqsim_narrative_arc_new import taqsim_narrative_arc_ui

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
        "Taqsim Narrative Arc",
    ],
)

# Home page
if page == "Home":
    render_home_page()


# Data Explorer page
elif page == "Data Explorer":
    data_explorer_page()


# Embeddings Visualization page
elif page == "Embeddings Visualization":
    embedding_visualizer_ui()


# Taqsim Narrative Arc page
elif page == "Taqsim Narrative Arc":
    taqsim_narrative_arc_ui()


# Footer
st.markdown("---")
st.markdown("© 2025 Taqsim x AI Project")
