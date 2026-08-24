import os
from pathlib import Path
import pickle
import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report, f1_score, ConfusionMatrixDisplay

import tensorflow as tf


#############################################################################################################
# Loading data + models:

def load_pickle_data(data_dir):
    """
    Load all pickle files from a directory into a dictionary.

    Parameters
    ----------
    data_dir : str or pathlib.Path
        Directory containing the pickle files. Only files with a `.pkl`
        extension directly within this directory are loaded.

    Returns
    -------
    dict
        Dictionary mapping each pickle file's stem (filename without the
        `.pkl` extension) to its deserialized contents.

    Notes
    -----
    Files are loaded in the order returned by `Path.glob()`. If multiple
    files have the same stem, the last loaded file overwrites earlier
    entries.

    Prints
    ------
    None
        Prints the number of loaded files and their dictionary keys.
    
    Example
    -------
    >>> SFM_DATA_DIR = "a/directory/is/here/"
    >>> pkl_data = load_pickle_data(SFM_DATA_DIR)
    """
    data_dir = Path(data_dir)
    pkl_data = {}

    for pkl_file in data_dir.glob("*.pkl"):
        with pkl_file.open("rb") as file:
            pkl_data[pkl_file.stem] = pickle.load(file)

    print(f"Loaded {len(pkl_data)} pickle files:")
    print(list(pkl_data))

    return pkl_data

def load_model(model_name, model_dir):
    """
    Load a Keras model and its associated training history from disk.

    Parameters
    ----------
    model_name : str
        Base name of the model (without extension).
    model_dir : str or Path
        Directory containing the model (.keras) and history (_history.pkl) files.

    Returns
    -------
    tuple
        (model, history_dict) where model is the loaded Keras model and 
        history_dict is the loaded training history (or None if not found).
    
    Example
    -------
    >>> SFM_MODELS_DIR = "a/directory/containing/models/and/history"
    >>> trained_model,trained_model_history =  load_model("model_1", SFM_MODELS_DIR)
    """

    model_dir = Path(model_dir)
    model_path = model_dir / f"{model_name}.keras"
    history_path = model_dir / f"{model_name}_history.pkl"

    model = tf.keras.models.load_model(model_path)
    print(f"Loaded model from '{model_path}'")

    history_dict = None
    if history_path.exists():
        with open(history_path, "rb") as f:
            history_dict = pickle.load(f)
        print(f"Loaded history from '{history_path}'")
    else:
        print(f"No history file found at '{history_path}'")

    return model, history_dict


#############################################################################################################
# Data shuffling + generation:

def shuffle_training_data(X_train, y_train, random_state=13):
    """
    Randomly shuffle training samples and their corresponding labels.

    The same permutation is applied to both inputs so that each feature
    sample remains paired with its original label.

    Parameters
    ----------
    X_train : array-like
        Training features or images. The first dimension represents samples.
    y_train : array-like
        Training labels corresponding to `X_train`.
    random_state : int or None, default=13
        Seed used to make the shuffle reproducible. Set to ``None`` for
        non-deterministic shuffling.

    Returns
    -------
    X_shuffled : array-like
        Shuffled training features.
    y_shuffled : array-like
        Labels shuffled using the same permutation as `X_shuffled`.

    Raises
    ------
    ValueError
        If `X_train` and `y_train` contain different numbers of samples.

    Example
    -------
    >>> X_train, y_train = shuffle_training_data(X_train, y_train)
    """
    
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must contain the same number of samples.")

    return shuffle(X_train, y_train, random_state=random_state)

def create_validation_set(X_train,y_train,val_size=0.2,random_state=13,stratify=True):
    """
    Split training data into training and validation subsets.

    The split is reproducible when ``random_state`` is set. When
    ``stratify=True``, both subsets preserve approximately the same
    class distribution as the original training data. The validation
    subset is shuffled after splitting.

    Parameters
    ----------
    X_train : array-like
        Training features or images. The first dimension must represent
        individual samples.
    y_train : array-like
        Class labels corresponding to ``X_train``.
    val_size : float or int, default=0.2
        Proportion or absolute number of samples assigned to the
        validation set.
    random_state : int or None, default=13
        Random seed used to make the split and validation shuffle
        reproducible. Use ``None`` for non-deterministic behavior.
    stratify : bool, default=True
        Whether to preserve the class distribution in both subsets.

    Returns
    -------
    X_train_new : array-like
        Features assigned to the new training set.
    X_val : array-like
        Features assigned to the validation set.
    y_train_new : array-like
        Labels corresponding to ``X_train_new``.
    y_val : array-like
        Labels corresponding to ``X_val``.

    Raises
    ------
    ValueError
        If the input arrays contain different numbers of samples or if
        stratification cannot be performed.

    Examples
    --------
    >>> X_train, X_val, y_train, y_val = create_validation_set(
    ...     X_train, y_train, val_size=0.2, random_state=13
    ... )
    """
    stratify_labels = y_train if stratify else None

    X_train_new, X_val, y_train_new, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        stratify=stratify_labels,
        random_state=random_state
    )

    X_val, y_val = shuffle(
        X_val,
        y_val,
        random_state=random_state
    )

    return X_train_new, X_val, y_train_new, y_val

def enhance_image(image,gamma=0.9,clahe_clip=10.0,clahe_grid=(16, 16),sharpen_amount=1.0):
    """
    Enhance a normalized grayscale image.

    Parameters
    ----------
    image : np.ndarray
        Image of shape (H, W, 1), expected range [0, 1].

    gamma : float
        < 1 brightens image.
        > 1 darkens image.

    CLAHE — Contrast Limited Adaptive Histogram Equalization

    clahe_clip : float
        Strength of local contrast enhancement.

    clahe_grid : tuple
        Size of local regions used by CLAHE.

    sharpen_amount : float
        Strength of unsharp masking.

    Returns
    -------
    np.ndarray
        Enhanced float32 image in range [0, 1].
    """

    # Remove final channel dimension
    img = np.squeeze(image)
    # Convert [0,1] float -> [0,255] uint8 for OpenCV
    img = np.clip(img * 255, 0, 255).astype(np.uint8)

    
    # 1. Gamma / exposure
    img_float = img.astype(np.float32) / 255.0
    img_float = np.power(img_float, gamma)
    img = np.clip(img_float * 255,0,255).astype(np.uint8)

    # 2. Local contrast: CLAHE
    clahe = cv2.createCLAHE(clipLimit=clahe_clip,tileGridSize=clahe_grid)
    img = clahe.apply(img)

    # 3. Mild sharpening
    blurred = cv2.GaussianBlur(img,(0, 0),sigmaX=1.0)
    img = cv2.addWeighted(img,1 + sharpen_amount,blurred,-sharpen_amount,0)

    # Convert back to float32 [0,1]
    img = img.astype(np.float32) / 255.0

    return img[..., np.newaxis]

def enhance_dataset(X, **kwargs):
    """
    Apply image enhancement to a batch of images.

    Parameters
    ----------
    X : array-like
        Batch of images to enhance. Shape should be (N, H, W) or (N, H, W, C).
    **kwargs : dict
        Keyword arguments to pass to enhance_image(). Common options:
        - gamma : float, default=1.0
            Gamma correction value for exposure adjustment.
        - clahe_clip : float, default=2.0
            Clip limit for CLAHE (Contrast Limited Adaptive Histogram Equalization).
        - clahe_grid : tuple, default=(8, 8)
            Tile grid size for CLAHE.
        - sharpen_amount : float, default=0.0
            Sharpening intensity.

    Returns
    -------
    np.ndarray
        Enhanced images as float32 array with shape (N, H, W, 1) and values in [0, 1].

    Examples
    --------
    >>> X = np.random.rand(10, 224, 224, 1)  # 10 images
    >>> enhanced = enhance_dataset(X, gamma=1.2, clahe_clip=3.0)
    """
    return np.stack([
        enhance_image(image, **kwargs)
        for image in X
    ]).astype(np.float32)


#############################################################################################################
# Model evaluation:

def plot_training_history(history, metric='loss'):
    """
    Plots training and validation curves for a given metric from a Keras History dict.

    Parameters
    ----------
    history : dict
        Training history dictionary (e.g. History.history or a loaded history dict).
        Expected to contain keys like 'loss', 'val_loss', 'accuracy', 'val_accuracy', etc.
    metric : str, default='loss'
        The metric to plot. Common options: 'loss', 'accuracy', 'precision', 'recall', 'f1_score'.

    Returns
    -------
    None
        Displays a matplotlib figure with training and validation curves.

    Raises
    ------
    ValueError
        If the specified metric or its validation counterpart is not found in the history dictionary.

    Example
    -------
    >>> plot_training_history(trained_model.history, metric='loss')
    >>> plot_training_history(trained_model_history, metric='accuracy')
    """
    val_metric = f"val_{metric}"

    if metric not in history or val_metric not in history:
        raise ValueError(f"Metric '{metric}' not found in training history.")

    plt.figure(figsize=(8, 5))
    plt.plot(history[metric], marker='o', label=f'Training {metric.capitalize()}')
    plt.plot(history[val_metric], marker='o', label=f'Validation {metric.capitalize()}')
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.title(f'Training vs Validation {metric.capitalize()}')
    plt.legend()
    plt.grid(True)
    plt.show()

def evaluate_model(model,X_test,y_test,class_names=("Class 0", "Class 1"),plot_name=""):
    """
    Evaluate a binary Keras classifier using integer labels (0/1).
    """

    # Ensure labels are compatible with Dense(1, sigmoid)
    y_test_eval = np.asarray(y_test).reshape(-1, 1)

    # Keras evaluation
    results = model.evaluate(
        X_test,
        y_test_eval,
        verbose=0
    )

    metric_names = model.metrics_names

    print("Test set evaluation:")
    for name, value in zip(metric_names, results):
        print(f"  {name}: {value}")

    # Predicted probabilities
    y_pred_probs = model.predict(X_test, verbose=0).ravel()

    # Convert probabilities -> classes
    y_pred = (y_pred_probs >= 0.5).astype(int)

    # sklearn prefers 1D labels
    y_test_1d = np.asarray(y_test).ravel()

    print("\nClassification Report:")
    print(
        classification_report(
            y_test_1d,
            y_pred,
            target_names=class_names
        )
    )

    macro_f1 = f1_score(
        y_test_1d,
        y_pred,
        average="macro"
    )

    print(f"Macro F1-score: {macro_f1:.4f}")

    cm = confusion_matrix(
        y_test_1d,
        y_pred
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix ({plot_name})")
    plt.show()

    return {
        "test_loss_metrics": dict(zip(metric_names, results)),
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "predicted_probabilities": y_pred_probs
    }


#############################################################################################################
# Saving models + history objects:

def save_model(file_path, model, history=None):
    """
    Save a Keras model (and optionally its training history) to disk.

    Parameters
    ----------
    file_path : str
        Destination path for the model file. Saved in the native Keras
        format (`.keras`). Parent directories are created automatically
        if they do not exist.
    model : keras.Model
        Trained Keras model to save.
    history : keras.callbacks.History or dict, optional
        Training history to save alongside the model. If provided, it is
        pickled to a `.pkl` file with the same base name as `file_path`.

    Returns
    -------
    None
        This function writes the model (and optionally history) to disk
        and prints confirmation messages.
    
    Example
    --------
    >>> model_path = "the/path/to/be/saved/to # Don't need to add .pkl or keras at the end.
    >>> save_model(model_path, model_1, model_1.history)
    """
    file_path = str(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Ensure correct extension for native Keras format
    if not file_path.endswith(".keras"):
        file_path = os.path.splitext(file_path)[0] + ".keras"

    model.save(file_path)
    print(f"Saved model to '{file_path}'")

    # Print size of saved model file in MB
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"Model size: {size_mb:.2f} MB")

    # Print estimated serialized history size
    if history is not None:
        history_dict = history.history if hasattr(history, "history") else history
        history_size_mb = len(pickle.dumps(history_dict)) / (1024 * 1024)
        print(f"History size: {history_size_mb:.4f} MB")

    if history is not None:
        history_dict = history.history if hasattr(history, "history") else history
        history_path = os.path.splitext(file_path)[0] + "_history.pkl"

        with open(history_path, "wb") as f:
            pickle.dump(history_dict, f)

        print(f"Saved history to '{history_path}'")


