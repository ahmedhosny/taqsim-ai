import streamlit as st


def render_home_page():
    # st.set_page_config should be called in the main app script
    st.header("Taqsim Atlas: Mapping the Improvised Landscape of the Taqsim")

    st.header("What Is a Taqsim?")
    st.write("""
    A taqsim is a form of solo instrumental improvisation central to Arabic, Turkish, and broader Middle Eastern musical traditions. Performed on instruments like the oud, ney, or violin, a taqsim unfolds a maqam (modal system) in real time, with no fixed sheet music or prescribed structure. Instead, it draws on deep stylistic intuition, technical fluency, and cultural tradition to tell a story through sound. A taqsim is both deeply personal and highly coded: it adheres to the grammar of maqamat, while offering space for individual expression, modulation, and interpretation.

    Yet despite its cultural richness, taqsim remains under-archived, under-analyzed, and poorly understood—especially in digital and educational spaces. Because taqasim are improvised and rarely notated, we lack systematic tools to study or preserve them, let alone compare or teach them across traditions, artists, and styles.
    """)

    st.header("Why This Project Matters?")
    st.write("""
    Taqsim is a living oral tradition, and without thoughtful documentation and analysis, much of its expressive nuance is at risk of being lost. Traditional methods—sheet music transcription or genre categorization—fail to capture the dynamism, temporal structure, and improvisational narrative that make each taqsim unique.

    This project aims to reimagine how we archive, study, and teach taqsim using modern tools—especially Artificial Intelligence (AI) and interactive data visualization. By analyzing patterns across thousands of taqasim, we can begin to uncover latent structures: narrative arcs, characteristic motifs, modulation patterns, and stylistic fingerprints. These insights can be used by educators, students, musicians, and researchers to deepen their understanding of maqam music in a way that respects its complexity and individuality.
    """)

    st.header("What the Project Is?")
    st.write("""
    This is a data-driven, musician-informed platform for understanding taqsim. At its core, the project does three things:

    - Build the largest database of oud taqasim to date, aggregating performances from YouTube, and eventually, social platforms like Instagram and TikTok.

    - Develop an AI-powered system to extract musical fingerprints from each taqsim: unbiased embeddings that help us compare taqasim independent of the oud, artist, or recording setup.

    - Produce expert-guided analyses of taqasim, combining AI outputs with musician annotation to interpret structure, motifs, and modulation events.

    These components converge in an interactive public website where taqasim are visualized, explored, and explained.
    """)

    st.header("Aims of the Project")

    st.subheader("1. Archive and Enrich")
    st.write("""
    Create an open, searchable, and growing repository of taqasim.

    Normalize and clean audio data to reduce confounding factors like recording quality and instrument variation.
    """)

    st.subheader("2. Analyze and Understand")
    st.write("""
    Use AI to generate embeddings (fingerprints) of each taqsim to map similarities and differences.

    Apply dimensionality reduction (e.g., UMAP) to visualize the "taqsim space" and cluster performances by maqam, artist, or style.

    Study narrative arcs of individual taqasim by combining silence detection, cadence recognition, and musical knowledge.
    """)

    st.subheader("3. Interpret and Educate")
    st.write("""
    Collaborate with expert musicians to label key musical events (modulations, motifs, climaxes).

    Offer side-by-side comparisons of taqasim by the same artist in different maqamat or of different structures (linear, cyclical, arched).

    Document and explain stylistic differences (e.g., Turkish vs. Arabic taqsim traditions).
    """)

    st.subheader("4. Share and Engage")
    st.write("""
    Build a public-facing platform featuring:

    “Artist in Focus” pages: tracing stylistic themes across taqasim by one performer.

    “Taqsim in Focus” analyses: annotated listening experiences highlighting structure and narrative.

    Search and filter functionality to explore taqasim by maqam, artist, region, and more.

    Expand dataset collection through community-driven contributions and automated scraping from social platforms.
    """)

    st.header("Deliverables")
    st.write("""
    - A publicly accessible website at [URL TBD], titled Taqsim Atlas (or similar).

    - A library of taqasim audio and metadata: the most comprehensive online resource of its kind.

    - UMAP-style interactive visualization of the taqsim space, colorable by maqam, artist, or stylistic features.

    - Annotated audio players showing the narrative arc and key moments in selected taqasim.

    - Regularly updated blog and research notes explaining methodology, findings, and musical context.

    - Tools for educators and learners: e.g., glossary of maqamat, video explainers, or lesson plans.
    """)

    st.header("Use and Impact")
    st.write("""
    This project will serve a wide range of communities:

    - Musicians and educators, who will gain new tools to teach and explore taqsim.

    - Students, especially in diaspora communities, who often lack access to traditional instruction.

    - Researchers in ethnomusicology and ML, who will find a rich, well-structured dataset to work with.

    - Audiences, who will enjoy a new way to listen and learn about a sophisticated art form.
    """)
