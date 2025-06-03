"""
Chunk Stitcher

This script:
1. Takes a directory containing audio chunks (.wav files)
2. Reads them in order based on their numbering
3. Takes the first second of each chunk (except the last one)
4. Stitches them together with the last chunk
5. Saves the result as a new .wav file

This is used to verify that the overlapping chunk generation is working correctly.
"""

import argparse
import os
import re

import numpy as np
import soundfile as sf


def stitch_audio_chunks(chunks_dir, output_file=None, sample_rate=16000):
    """
    Stitch audio chunks together by taking the first second of each chunk
    (except the last one) and then adding the last chunk in full.

    Args:
        chunks_dir: Directory containing the audio chunks
        output_file: Path to save the stitched audio file
        sample_rate: Sample rate of the audio files

    Returns:
        Path to the stitched audio file
    """
    print(f"Looking for audio chunks in: {chunks_dir}")

    # Get all .wav files in the directory
    wav_files = [f for f in os.listdir(chunks_dir) if f.endswith(".wav")]

    if not wav_files:
        print(f"No .wav files found in {chunks_dir}")
        return None

    # Extract the numbering from filenames and sort
    def get_chunk_number(filename):
        # Extract the number at the end of the filename (before .wav)
        match = re.search(r"_(\d+)\.wav$", filename)
        if match:
            return int(match.group(1))
        return 0

    wav_files.sort(key=get_chunk_number)
    print(f"Found {len(wav_files)} audio chunks")

    # Read all audio chunks
    chunks = []
    for i, wav_file in enumerate(wav_files):
        file_path = os.path.join(chunks_dir, wav_file)
        audio, sr = sf.read(file_path)

        if i < len(wav_files) - 1:
            # For all except the last chunk, take only the first second
            samples_per_second = sr
            if len(audio) >= samples_per_second:
                chunks.append(audio[:samples_per_second])
            else:
                print(
                    f"Warning: {wav_file} is shorter than 1 second, using entire file"
                )
                chunks.append(audio)
        else:
            # For the last chunk, take the entire audio
            chunks.append(audio)
            print(f"Added full last chunk: {wav_file} ({len(audio) / sr:.2f}s)")

    # Stitch chunks together
    stitched_audio = np.concatenate(chunks)

    # Determine output file path if not provided
    if output_file is None:
        # Extract base name from the first file (removing the numbering)
        base_name = re.sub(r"_\d+\.wav$", "", wav_files[0])
        output_file = os.path.join(chunks_dir, f"{base_name}_stitched.wav")

    # Save stitched audio
    sf.write(output_file, stitched_audio, sample_rate, "PCM_16")
    print(f"Stitched audio saved to: {output_file}")
    print(f"Total duration: {len(stitched_audio) / sample_rate:.2f} seconds")

    return output_file


def main():
    """Parse command line arguments and stitch audio chunks."""
    parser = argparse.ArgumentParser(
        description="Stitch audio chunks to verify overlapping chunk generation"
    )
    parser.add_argument(
        "--chunks_dir",
        required=True,
        help="Directory containing audio chunks (.wav files)",
    )
    parser.add_argument(
        "--output_file", default=None, help="Path to save the stitched audio file"
    )
    parser.add_argument(
        "--sample_rate", type=int, default=16000, help="Sample rate of the audio files"
    )

    args = parser.parse_args()

    stitch_audio_chunks(
        chunks_dir=args.chunks_dir,
        output_file=args.output_file,
        sample_rate=args.sample_rate,
    )


if __name__ == "__main__":
    main()
