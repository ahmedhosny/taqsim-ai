"""
Maqam Classifier using XGBoost

This script builds an XGBoost model to predict maqam types (Rast, Nahawand, Hijaz, Bayati)
based on the first chunk embeddings of audio files.

The script:
1. Loads the taqsim_ai.csv metadata file
2. Filters for the specified maqams
3. Loads the first chunk embeddings for each song
4. Performs a stratified 80:20 train-test split
5. Trains an XGBoost classifier
6. Evaluates the model on the test set
7. Generates a confusion matrix with accuracy percentages
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)  # Go one level up
DATA_DIR = os.path.join(PARENT_DIR, "data")
METADATA_CSV_PATH = os.path.join(DATA_DIR, "taqsim_ai.csv")
EMBEDDINGS_DIR_PATH = os.path.join(DATA_DIR, "embeddings")

# Target maqams for classification
TARGET_MAQAMS = ["Rast", "Nahawand", "Hijaz", "Bayati"]

# Valid embedding types
VALID_EMBEDDING_TYPES = ["cls", "dist", "avg", "combined"]


def load_metadata():
    """Load and filter metadata for target maqams."""
    print("Loading metadata...")
    df = pd.read_csv(METADATA_CSV_PATH)

    # Filter for target maqams
    filtered_df = df[df["maqam"].isin(TARGET_MAQAMS)]

    # Check if we have data for all target maqams
    available_maqams = filtered_df["maqam"].unique()
    print(f"Available maqams: {available_maqams}")

    missing_maqams = set(TARGET_MAQAMS) - set(available_maqams)
    if missing_maqams:
        print(
            f"Warning: The following maqams are missing from the dataset: {missing_maqams}"
        )

    # Count samples per maqam
    maqam_counts = filtered_df["maqam"].value_counts()
    print("Samples per maqam:")
    for maqam, count in maqam_counts.items():
        print(f"  {maqam}: {count}")

    return filtered_df


def get_first_chunk_embedding(uuid, embedding_type="combined"):
    """Get the embedding for the first chunk of a given UUID."""
    # Find all embedding files for this UUID
    pattern = os.path.join(EMBEDDINGS_DIR_PATH, f"{uuid}_*.npz")
    embedding_files = glob.glob(pattern)

    if not embedding_files:
        return None

    # Extract chunk numbers from filenames
    chunk_info = []
    for file_path in embedding_files:
        filename = os.path.basename(file_path)
        parts = filename.split("_")

        # Find the first numeric part which is the chunk number
        for i, part in enumerate(parts):
            if part.isdigit():
                chunk_number = int(part)
                chunk_info.append((chunk_number, file_path))
                break

    if not chunk_info:
        return None

    # Sort by chunk number and get the first one
    chunk_info.sort()
    first_chunk_file = chunk_info[0][1]

    # Load the embedding
    with np.load(first_chunk_file) as data:
        embedding = data["embedding"]

        # If the embedding has shape [3, 768], it contains cls, dist, and avg embeddings
        if embedding.shape[0] == 3 and len(embedding.shape) == 2:
            # Select the appropriate embedding based on the type
            if embedding_type == "cls":
                embedding = embedding[0]  # First row is CLS token
            elif embedding_type == "dist":
                embedding = embedding[1]  # Second row is DIST token
            elif embedding_type == "avg":
                embedding = embedding[2]  # Third row is AVG token
            elif embedding_type == "combined":
                embedding = embedding.flatten()  # Flatten all three for combined
            else:
                # Default to combined if invalid type
                embedding = embedding.flatten()

    return embedding


def prepare_dataset(metadata_df, embedding_type="combined"):
    """Prepare dataset with embeddings as features and maqams as labels."""
    print("Preparing dataset...")
    X = []
    y = []
    uuids = []

    for _, row in metadata_df.iterrows():
        uuid = row["uuid"]
        maqam = row["maqam"]

        embedding = get_first_chunk_embedding(uuid, embedding_type)
        if embedding is not None:
            # Ensure the embedding is flattened if it's still multi-dimensional
            if len(embedding.shape) > 1:
                embedding = embedding.flatten()

            X.append(embedding)
            y.append(maqam)
            uuids.append(uuid)

    print(f"Dataset prepared with {len(X)} samples")

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    return X, y, uuids


def stratified_train_test_split_by_maqam(X, y, test_size=0.5, random_state=42):
    """Perform stratified train-test split ensuring each maqam has the same proportion."""
    # Encode labels for stratification
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Perform stratified split
    X_train, X_test, y_train_encoded, y_test_encoded, train_indices, test_indices = (
        train_test_split(
            X,
            y_encoded,
            np.arange(len(y)),
            test_size=test_size,
            random_state=random_state,
            stratify=y_encoded,
        )
    )

    # Convert encoded labels back to original
    y_train = le.inverse_transform(y_train_encoded)
    y_test = le.inverse_transform(y_test_encoded)

    # Verify stratification
    train_distribution = pd.Series(y_train).value_counts(normalize=True)
    test_distribution = pd.Series(y_test).value_counts(normalize=True)

    print("Train distribution:")
    for maqam, prop in train_distribution.items():
        print(f"  {maqam}: {prop:.2f} ({(prop * len(y_train)):.0f} samples)")

    print("Test distribution:")
    for maqam, prop in test_distribution.items():
        print(f"  {maqam}: {prop:.2f} ({(prop * len(y_test)):.0f} samples)")

    return X_train, X_test, y_train, y_test


def train_xgboost_model(X_train, y_train):
    """Train an XGBoost classifier."""
    print("Training XGBoost model...")

    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    # Create and train the model
    model = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=len(TARGET_MAQAMS),
        learning_rate=0.1,
        max_depth=5,
        n_estimators=100,
        random_state=42,
        verbosity=1,  # Add some verbosity to see training progress
    )

    model.fit(X_train, y_train_encoded)

    return model, le


def evaluate_model(model, X_test, y_test, label_encoder):
    """Evaluate the model on the test set."""
    print("Evaluating model...")

    # Encode true labels
    y_test_encoded = label_encoder.transform(y_test)

    # Make predictions
    y_pred_encoded = model.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall accuracy: {accuracy:.4f}")

    # Classification report
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    print("Classification Report:")
    print(report)

    return y_test, y_pred, label_encoder.classes_


def plot_confusion_matrix(
    y_true, y_pred, class_names, output_filename="maqam_confusion_matrix.png"
):
    """Plot confusion matrix with accuracy percentages."""
    print("Generating confusion matrix...")

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_names)

    # Calculate accuracy for each class
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    class_accuracies = np.diag(cm_normalized)

    # Create annotation text with counts and percentages
    cm_text = np.zeros_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            cm_text[i, j] = f"{cm[i, j]}\n({cm_normalized[i, j]:.1%})"

    # Plot with more room for title
    plt.figure(figsize=(10, 10))  # Increased height even more

    # Create the heatmap without the color bar
    sns.heatmap(
        cm,
        annot=cm_text,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,  # Remove the color bar
    )

    # Add labels
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # Add title with more padding
    embedding_type = output_filename.split("_")[-1].split(".")[0]
    plt.title(
        f"Confusion Matrix with Accuracy Percentages\nEmbedding Type: {embedding_type}",
        pad=30,  # Add padding between title and plot
    )

    # Add class accuracies as text
    for i, accuracy in enumerate(class_accuracies):
        plt.text(
            i + 0.5,
            -0.1,
            f"{class_names[i]}: {accuracy:.1%}",
            ha="center",
            va="center",
            fontsize=10,
            transform=plt.gca().transData,
        )

    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Confusion matrix saved as '{output_filename}'")
    plt.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train an XGBoost model to predict maqam types."
    )
    parser.add_argument(
        "--embedding-type",
        type=str,
        choices=VALID_EMBEDDING_TYPES,
        default="combined",
        help="Type of embedding to use: cls, dist, avg, or combined (default: combined)",
    )
    return parser.parse_args()


def main():
    """Main function to run the maqam classification pipeline."""
    # Parse command line arguments
    args = parse_args()
    embedding_type = args.embedding_type
    print(f"Using embedding type: {embedding_type}")

    # Load and filter metadata
    metadata_df = load_metadata()

    # Prepare dataset
    X, y, uuids = prepare_dataset(metadata_df, embedding_type)

    # Split data
    X_train, X_test, y_train, y_test = stratified_train_test_split_by_maqam(X, y)

    # Train model
    model, label_encoder = train_xgboost_model(X_train, y_train)

    # Evaluate model
    y_true, y_pred, class_names = evaluate_model(model, X_test, y_test, label_encoder)

    # Plot confusion matrix
    output_filename = f"maqam_confusion_matrix_{embedding_type}.png"
    plot_confusion_matrix(y_true, y_pred, class_names, output_filename)

    print(f"Maqam classification complete using {embedding_type} embeddings!")


if __name__ == "__main__":
    main()
