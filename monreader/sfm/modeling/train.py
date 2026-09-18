# CCDS Specific imports:
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer

# Imports from monreader:
from monreader.sfm.config import SFM_MODELS_DIR, PROCESSED_DATA_DIR
from monreader.utils import train_sagemaker

# Library imports:
import runpy

# Main script
app = typer.Typer()
@app.command()
def main(
    # Main parameters e.g. model, job, and data names are defined in monreader.utils.train_sagemaker.py
):
    # Execute the 'train_sagemaker.py' script located in the utils folder
    runpy.run_module("monreader.utils.train_sagemaker", run_name="__main__")
    

if __name__ == "__main__":
    app()
