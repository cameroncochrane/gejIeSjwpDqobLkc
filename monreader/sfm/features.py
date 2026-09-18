# CCDS Specific imports:
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer

# Imports from monreader:
from monreader.sfm.config import RAW_DATA_DIR, SFM_PROCESSED_DATA_DIR
from monreader.utils.mrp_functions import load_images_to_df, convert_images_to_grayscale, resize_images_in_df, add_grayscale_channel, normalize_images, encode_labels, shuffle_data, create_validation_set

# Library imports:
import gc
import cv2
import numpy as np
import pandas as pd
import pickle

# Main script
app = typer.Typer()
@app.command()
def main(
    input_path: Path = RAW_DATA_DIR,
    output_path: Path = SFM_PROCESSED_DATA_DIR / "sfm_processed_data_gray.pkl"
):
    RAW_TRAIN_DATA_DIR = input_path / "training"
    RAW_TEST_DATA_DIR = input_path / "testing"

    # Train set to intermediate conversion:
    raw_training = load_images_to_df(image_dir=RAW_TRAIN_DATA_DIR,
                                label=False,
                                label_from_subdir=True,
                                split='training')

    gray_training = convert_images_to_grayscale(raw_training)

    del raw_training #GC for memory efficiency

    print(f"Shape: {gray_training.shape}")
    print(f"Image data-type: {type(gray_training["image"][0])}, Shape: {gray_training["image"][0].shape}")
    image_bytes = gray_training["image"].map(lambda image: image.nbytes).sum()
    print(f"Image data: {image_bytes / 1024**2:.2f} MB")
    print(f"DataFrame reported: {gray_training.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Test set raw to intermediate conversion:
    raw_testing = load_images_to_df(image_dir=RAW_TEST_DATA_DIR,
                                label=False,
                                label_from_subdir=True,
                                split='testing')

    gray_testing = convert_images_to_grayscale(raw_testing)

    del raw_testing #GC for memory efficiency

    print(f"Shape: {gray_testing.shape}")
    print(f"Image data-type: {type(gray_testing["image"][0])}, Shape: {gray_testing["image"][0].shape}")
    image_bytes = gray_testing["image"].map(lambda image: image.nbytes).sum()
    print(f"Image data: {image_bytes / 1024**2:.2f} MB")
    print(f"DataFrame reported: {gray_testing.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


    # Downsize images x0.25:
    gray_training = resize_images_in_df(gray_training, 0.25)
    gray_testing = resize_images_in_df(gray_testing, 0.25)

    # Add grayscale channel (N x H x W) -> (N x H x W x 1):
    gray_training = add_grayscale_channel(gray_training)
    gray_testing = add_grayscale_channel(gray_testing)

    # Normalize pixel values: [0-255] -> [0-1]:
    X_train = normalize_images(X_train)
    X_test = normalize_images(X_test)

    # Encode labels to binary:
    y_train = encode_labels(y_train)
    y_test = encode_labels(y_test)

    # Convert to numpy arrays:
    X_train = np.stack(X_train.to_numpy())
    X_test = np.stack(X_test.to_numpy())
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    # Shuffle training data and create validation sets:
    X_train, y_train = shuffle_data(X_train, y_train)
    X_train, X_val, y_train, y_val = create_validation_set(X_train, y_train, val_size=0.2, random_state=13, stratify=True)

    # Save the processed data to the output path:
    processed_data = {
        "X_train": np.asarray(X_train, dtype=np.float32),
        "y_train": np.asarray(y_train),
        "X_val": np.asarray(X_val, dtype=np.float32),
        "y_val": np.asarray(y_val),
        "X_test": np.asarray(X_test, dtype=np.float32),
        "y_test": np.asarray(y_test),
    }

    with open(output_path, "wb") as file:
        pickle.dump(processed_data, file)

if __name__ == "__main__":
    app()
