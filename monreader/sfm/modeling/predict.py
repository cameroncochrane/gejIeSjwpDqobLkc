# CCDS Specific imports:
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer

# Imports from monreader:
from monreader.sfm.config import SFM_PROCESSED_DATA_DIR, SFM_MODELS_DIR
from monreader.utils.mrp_functions import load_model, load_pickle_data, evaluate_model, plot_training_history

# Library imports:
import numpy as np
import pandas as pd
import tensorflow as tf

# Main script
app = typer.Typer()
@app.command()
def main(
    data_dir: Path = SFM_PROCESSED_DATA_DIR,
    model_directory: Path = SFM_MODELS_DIR / "aws_trained" / "monreader_model" 
):
    
    pkl_data = load_pickle_data(data_dir)
    data = pkl_data["sfm_processed_data_gray"]
    X_test = data['X_test']
    y_test = data['y_test']

    train_model = False

    if train_model == False:
        model, model_history =  load_model("final_model", model_directory)
        plot_training_history(model_history, metric='loss')


    model_to_eval = model if train_model == False else model

    # Guard against accidentally passing the training History object
    if isinstance(model_to_eval, tf.keras.callbacks.History):
        model_to_eval = model

    # Ensure rank is known and the arrays are concrete numpy arrays
    X_eval = np.asarray(X_test, dtype=np.float32)
    y_eval = np.asarray(y_test, dtype=np.int32).reshape(-1)

    eval_results = evaluate_model(
        model_to_eval,
        X_eval,
        y_eval,
        class_names=('notflip', 'flip'),
        plot_name="final_model"
    ) # Automatically saves evaluation results to the figures directory



if __name__ == "__main__":
    app()
