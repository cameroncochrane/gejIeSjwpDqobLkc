# CCDS Specific imports:
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer

# Imports from monreader:
from monreader.sfm.config import MODELS_DIR, PROCESSED_DATA_DIR, SFM_MODELS_DIR
from monreader.utils.mrp_functions import load_model, load_pickle_data, evaluate_model, plot_training_history

# Main script
app = typer.Typer()
@app.command()
def main(
    data_path: Path = PROCESSED_DATA_DIR / "sfm_processed_data_gray.pkl",
    model_directory: Path = SFM_MODELS_DIR / "aws_trained" / "monreader_model"
):
    return None


if __name__ == "__main__":
    app()
