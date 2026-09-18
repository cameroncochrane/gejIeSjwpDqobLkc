# CCDS Specific imports:
from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

# Imports from monreader:
from monreader.sfm.config import RAW_DATA_DIR, SFM_PROCESSED_DATA_DIR
from monreader.utils.mrp_functions import count_frames_per_clip

# Library imports:
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Main script
app = typer.Typer()
@app.command()
def main(
    input_path: Path = RAW_DATA_DIR,
    output_path: Path = None
):
    # Counting the amount to judge class balance and existing train/test splits
    data_count_df = count_frames_per_clip(RAW_DATA_DIR)
    print(data_count_df.groupby(["split", "label"])["frame_count"].agg(["count", "mean", "min", "max"]))

    # Raw image analysis:
    # 1. View raw image and its dimensions:
    image_name = '0001_000000001.jpg'
    img = cv2.imread(RAW_DATA_DIR/"training/notflip"/image_name)
    print("Raw image dimensions (pixels): ", img.shape)
    print("Raw image datatype: ", type(img))

    plt.figure(figsize=(12, 7))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


    # 2. Convert the image to grayscale and view it:
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("Grayscale image dimensions (pixels): ", gray_img.shape)
    print("Grayscale image datatype: ", type(gray_img))

    plt.figure(figsize=(12, 7))
    plt.imshow(gray_img, cmap="gray") #Default imshow colour is green. Specify gray here
    plt.axis("off")
    plt.show()

    # 3. Viewing pixel values for a random 25x25 patch from an image
    patch_size = 25
    h, w = gray_img.shape

    # Random top-left corner for 20x20 patch
    start_row = np.random.randint(0, h - patch_size + 1)
    start_col = np.random.randint(0, w - patch_size + 1)

    patch = gray_img[start_row:start_row + patch_size, start_col:start_col + patch_size]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Full image with selected patch highlighted
    ax1.imshow(gray_img, cmap='gray')
    ax1.axis('off')
    rect = plt.Rectangle((start_col, start_row), patch_size, patch_size,
                        fill=False, edgecolor='red', linewidth=2)
    ax1.add_patch(rect)

    # 20x20 patch with pixel annotations
    ax2.imshow(patch, cmap='gray')
    ax2.set_title(f'Random {patch_size}x{patch_size} patch')
    ax2.set_xticks(range(patch_size))
    ax2.set_yticks(range(patch_size))
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])

    for i in range(patch_size):
        for j in range(patch_size):
            ax2.text(j, i, int(patch[i, j]), ha="center", va="center", color="red", fontsize=6)

    plt.tight_layout()
    plt.show()

    print(f"Patch location (row, col): ({start_row}, {start_col})")
    print(f"Patch shape: {patch.shape}")
    print(f"Pixel value range in patch: {patch.min()} - {patch.max()}")
    print(f"Data type: {patch.dtype}")

    # Hard to plot the whole image as theyare quite large and detailed. We are viewing a small area here.

if __name__ == "__main__":
    app()
