#!/usr/bin/env python3
"""
Embedding Extractor

This module handles the extraction of embeddings from audio chunks using
the MAEST model. It provides functions to extract embeddings and save them
to disk in a compressed numpy format.
"""

import os

import numpy as np


def create_embeddings_directory(output_path=None):
    """
    Create a directory for storing embeddings.
    Uses the same directory structure as audio chunks.

    Args:
        output_path: Optional custom path for the output directory

    Returns:
        Path to the created embeddings directory
    """
    if output_path is None:
        # Get the path to the data directory, same as in audio_chunker
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
        )
        # Create embeddings directory next to audio_chunks
        output_path = os.path.join(data_dir, "embeddings")

    os.makedirs(output_path, exist_ok=True)
    print(f"Created embeddings directory: {output_path}")
    return output_path


def extract_embeddings_from_maest(
    chunks, model_name="mtg-upf/discogs-maest-30s-pw-129e", transformer_block=6
):
    """
    Extract embeddings from the MAEST model for each audio chunk.

    Args:
        chunks: List of audio chunks as numpy arrays
        model_name: Name of the MAEST model to use
        transformer_block: Which transformer block to extract embeddings from (default: 6)

    Returns:
        Dictionary with chunk indices as keys and embeddings as values
    """
    print(f"Loading MAEST model for embedding extraction: {model_name}")

    try:
        # Try to import the MAEST model directly
        try:
            # First try to import from maest package if installed
            from maest import get_maest

            model = get_maest(arch=model_name.split("/")[-1])
            print("Successfully loaded MAEST model from maest package")
            direct_maest = True
        except ImportError:
            # If that fails, use the transformers model
            from transformers import AutoFeatureExtractor, AutoModel

            model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                model_name, trust_remote_code=True
            )
            print("Using transformers model for embedding extraction")
            direct_maest = False

        # Set model to evaluation mode
        model.eval()

        embeddings_dict = {}

        import torch

        with torch.no_grad():
            for i, chunk in enumerate(chunks):
                print(f"Extracting embeddings for chunk {i + 1}/{len(chunks)}...")

                if direct_maest:
                    # Use the direct MAEST interface
                    _, embeddings = model(chunk, transformer_block=transformer_block)
                    # Convert to numpy array
                    embeddings = embeddings.cpu().numpy()
                else:
                    # Use the transformers interface
                    inputs = feature_extractor(
                        chunk, sampling_rate=16000, return_tensors="pt"
                    )
                    outputs = model(**inputs, output_hidden_states=True)

                    # Get embeddings from the specified transformer block
                    hidden_states = outputs.hidden_states[transformer_block]

                    # Extract CLS token, DIST token, and average of rest of tokens
                    cls_token = hidden_states[0, 0].cpu().numpy()  # CLS token
                    dist_token = hidden_states[0, 1].cpu().numpy()  # DIST token
                    rest_tokens = (
                        hidden_states[0, 2:].mean(dim=0).cpu().numpy()
                    )  # Average of rest

                    # Stack them together
                    embeddings = np.vstack([cls_token, dist_token, rest_tokens])

                # Store embeddings for this chunk
                embeddings_dict[i + 1] = embeddings

        return embeddings_dict

    except Exception as e:
        print(f"Error extracting embeddings: {e}")
        import traceback

        traceback.print_exc()
        return {}


def save_embeddings_to_file(embeddings_dict, output_file="embeddings.npz"):
    """
    Save embeddings to a numpy compressed file.

    Args:
        embeddings_dict: Dictionary with chunk indices as keys and embeddings as values
        output_file: Path to the output file

    Returns:
        Boolean indicating success or failure
    """
    try:
        # Convert integer keys to strings for np.savez_compressed
        string_dict = {f"chunk_{key}": value for key, value in embeddings_dict.items()}
        np.savez_compressed(output_file, **string_dict)
        print(f"Saved embeddings to {output_file}")
        return True
    except Exception as e:
        print(f"Error saving embeddings: {e}")
        import traceback

        traceback.print_exc()
        return False


def extract_and_save_embeddings(chunk_paths, embeddings_dir, uuid):
    """
    Extract embeddings from audio chunks and save them to separate files.
    Each chunk's embeddings are saved in a separate NPZ file with the same naming
    as the audio chunk it was generated from.
    If embeddings already exist for a chunk, they will be reused instead of re-extracting.

    Args:
        chunk_paths: List of paths to audio chunk files
        embeddings_dir: Directory to save embeddings
        uuid: Unique identifier for the audio file

    Returns:
        Tuple containing (list of paths to the saved embeddings files, boolean indicating if any were reused)
        or (None, False) if extraction failed
    """
    if not chunk_paths:
        print("No audio chunks provided for embedding extraction")
        return None, False

    try:
        import os.path

        import librosa

        # Create a list to store paths to saved embedding files
        embedding_files = []
        all_reused = True  # Track if all embeddings were reused

        # Process each chunk individually
        for i, chunk_path in enumerate(chunk_paths):
            # Get the base filename without extension
            chunk_basename = os.path.basename(chunk_path)
            chunk_name = os.path.splitext(chunk_basename)[0]

            # Check if embedding already exists for this chunk
            embedding_file = os.path.join(embeddings_dir, f"{chunk_name}.npz")
            if os.path.exists(embedding_file):
                print(f"Found existing embedding for chunk: {chunk_name}, reusing it")
                embedding_files.append(embedding_file)
                continue

            # If we reach here, at least one embedding needs to be created
            all_reused = False

            print(f"Processing chunk {i + 1}/{len(chunk_paths)}: {chunk_name}")

            # Load audio chunk
            audio, sr = librosa.load(chunk_path, sr=16000, mono=True)

            # Extract embedding for this single chunk
            embeddings_dict = extract_embeddings_from_maest([audio])

            if not embeddings_dict or 1 not in embeddings_dict:
                print(f"Failed to extract embeddings for chunk: {chunk_name}")
                continue

            # Get the embedding for this chunk (key is 1 since we only processed one chunk)
            embedding = embeddings_dict[1]

            # Create a dictionary with a single entry for this chunk
            chunk_dict = {"embedding": embedding}

            # Save to a file with the same name as the chunk
            np.savez_compressed(embedding_file, **chunk_dict)

            print(f"Saved embeddings to {embedding_file}")
            embedding_files.append(embedding_file)

        if embedding_files:
            if all_reused:
                print(f"Using {len(embedding_files)} existing embedding files")
            else:
                print(f"Successfully processed {len(embedding_files)} embedding files")
            return embedding_files, all_reused
        else:
            print("No embeddings were successfully extracted and saved")
            return None, False

    except Exception as e:
        print(f"Error in embedding extraction pipeline: {e}")
        import traceback

        traceback.print_exc()
        return None, False
