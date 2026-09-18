import os
from pathlib import Path
import pickle
import json
import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report, f1_score, ConfusionMatrixDisplay

import tensorflow as tf

from monreader.sfm.config import SFM_FIGURES_DIR

#############################################################################################################
# Count + Load raw images
def count_frames_per_clip(data_dir) -> pd.DataFrame:
    """
    Count frames per clip in the dataset.

    Expects: data_dir / <split> / <label> / <VideoID>_<FrameNumber>.jpg
    e.g.     data/raw / training  / flip   / 0001_000000020.jpg

    VideoID numbering resets inside each (split, label) folder, so a clip is
    only uniquely identified by the (split, label, video_id) triple, not by
    video_id alone.

    Returns one row per clip: split, label, video_id, frame_count,
    min_frame, max_frame (raw frame numbers aren't contiguous within a clip,
    so max_frame - min_frame + 1 can be larger than frame_count).
    """
    data_dir = Path(data_dir)
    rows = []
    for split_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for label_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            counts, frame_min, frame_max = {}, {}, {}
            for img_path in label_dir.glob("*.jpg"):
                video_id, _, frame_str = img_path.stem.partition("_")
                frame_number = int(frame_str)
                counts[video_id] = counts.get(video_id, 0) + 1
                frame_min[video_id] = min(frame_min.get(video_id, frame_number), frame_number)
                frame_max[video_id] = max(frame_max.get(video_id, frame_number), frame_number)
            for video_id, frame_count in counts.items():
                rows.append({
                    "split": split_dir.name,
                    "label": label_dir.name,
                    "video_id": video_id,
                    "frame_count": frame_count,
                    "min_frame": frame_min[video_id],
                    "max_frame": frame_max[video_id],
                })
    return (
        pd.DataFrame(rows)
        .sort_values(["split", "label", "video_id"])
        .reset_index(drop=True)
    )

def load_images_to_df(image_dir, label=None, label_from_subdir=True, split=None):
    """
    Load images into a DataFrame using cv2.imread.

    Images are converted from OpenCV's BGR format to RGB.

    Example:
    ---------
    >>> raw_training = load_images_to_df(image_dir=TRAIN_DATA_DIR,label=False,label_from_subdir=True,split='training')

    >>> raw_testing = load_images_to_df(image_dir=TEST_DATA_DIR,label=False,label_from_subdir=True,split='testing')
    """
    ALLOWED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif'}
    records = []
    image_dir = Path(image_dir)

    if not image_dir.is_dir():
        return pd.DataFrame(records)

    def _load_from_dir(directory, lbl):
        for root_dir, _, files in os.walk(directory):
            for fname in sorted(files):
                path = os.path.join(root_dir, fname)

                if Path(fname).suffix.lower() not in ALLOWED_EXTS:
                    continue

                image = cv2.imread(path, cv2.IMREAD_COLOR)
                if image is None:
                    continue

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                records.append({
                    'image': image,
                    'label': lbl,
                    'split': split
                })

    if label_from_subdir:
        subdirs = [
            d for d in sorted(os.listdir(image_dir))
            if (image_dir / d).is_dir()
        ]

        if subdirs:
            for subdir in subdirs:
                _load_from_dir(image_dir / subdir, subdir)
        else:
            _load_from_dir(image_dir, label)
    else:
        _load_from_dir(image_dir, label)

    return pd.DataFrame(records)


#############################################################################################################
# Data shuffling + processing:

def shuffle_data(X_sample, y_sample, random_state=13):
    """
    Randomly shuffle samples and their corresponding labels.

    The same permutation is applied to both inputs so that each feature
    sample remains paired with its original label.

    Parameters
    ----------
    X_sample : array-like
        Training features or images. The first dimension represents samples.
    y_sample : array-like
        Labels corresponding to `X_sample`.
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
        If `X_sample` and `y_sample` contain different numbers of samples.

    Example
    -------
    >>> X_train, y_train = shuffle_data(X_train, y_train)
    """
    
    if len(X_sample) != len(y_sample):
        raise ValueError("X_sample and y_sample must contain the same number of samples.")

    return shuffle(X_sample, y_sample, random_state=random_state)

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

def convert_images_to_grayscale(df):
    """
    Return a copy of df with images converted from RGB to grayscale.
    """
    grayscale_df = df.copy()
    grayscale_df["image"] = grayscale_df["image"].map(
        lambda image: cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    )
    return grayscale_df

def resize_image(image, scale_factor=0.5):
    """
    Resize a grayscale image by a fixed scale factor while preserving aspect ratio.
    """
    height, width = image.shape[:2]
    new_width = max(1, int(width * scale_factor))
    new_height = max(1, int(height * scale_factor))

    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

def resize_images_in_df(df, scale_factor=0.5):
    """
    Return a copy of df with all images resized by scale_factor.
    """
    resized_df = df.copy()
    resized_df["image"] = resized_df["image"].map(
        lambda image: resize_image(image, scale_factor)
    )
    return resized_df

def add_grayscale_channel(df):
    """
    Return a copy of the DataFrame with grayscale images shaped as
    (height, width, 1) for TensorFlow models. 
    
    Try to use after resizing the image. Hasn't been tried without resizing.
    """
    channel_df = df.copy()
    channel_df["image"] = channel_df["image"].map(
        lambda image: image[..., np.newaxis]
        if image.ndim == 2
        else image
    )
    return channel_df

def normalize_images(images):
    """Normalize image pixel values to the range [0, 1] as float32."""
    return images.map(lambda image: image.astype(np.float32) / 255.0)

def encode_labels(labels):
    """Encode 'flip' as 1 and all other labels as 0."""
    return labels.map(lambda label: 1 if label == "flip" else 0)

def series_to_numpy(series, stack=False):
    """
    Convert a pandas Series to a NumPy array.

    Set stack=True when each Series element is a NumPy array, such as an image.

    Example:
    ---------
    >>> X_train = series_to_numpy(X_train, stack=True)
    >>> X_test = series_to_numpy(X_test, stack=True)

    >>> y_train = series_to_numpy(y_train)
    >>> y_test = series_to_numpy(y_test)
    """
    values = series.to_numpy()
    return np.stack(values) if stack else values


#############################################################################################################
# Model evaluation:

def plot_training_history(history, metric='loss', plot_name="", save_dir=SFM_FIGURES_DIR):
    """
    Plots training and validation curves for a given metric from a Keras History dict.

    Parameters
    ----------
    history : dict
        Training history dictionary (e.g. History.history or a loaded history dict).
        Expected to contain keys like 'loss', 'val_loss', 'accuracy', 'val_accuracy', etc.
    metric : str, default='loss'
        The metric to plot. Common options: 'loss', 'accuracy', 'precision', 'recall', 'f1_score'.
    plot_name : str, optional
        Identifier (e.g. model name) included in the saved chart's filename.
    save_dir : str or Path, optional
        Directory the chart is saved to. Defaults to SFM_FIGURES_DIR. Set to
        None to skip saving and only display the chart.

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
    >>> plot_training_history(trained_model_history, metric='accuracy', plot_name='model_3_2_3')
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

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{plot_name}_{metric}_history.png" if plot_name else f"{metric}_history.png"
        plt.savefig(save_dir / filename, bbox_inches="tight")
        print(f"Saved plot to '{save_dir / filename}'")

    plt.show()

def evaluate_model(model,X_test,y_test,class_names=("Class 0", "Class 1"),plot_name="",save_dir=SFM_FIGURES_DIR):
    """
    Evaluate a binary Keras classifier using integer labels (0/1).

    Parameters
    ----------
    save_dir : str or Path, optional
        Directory the confusion matrix chart is saved to. Defaults to
        SFM_FIGURES_DIR. Set to None to skip saving and only display it.
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

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{plot_name}_confusion_matrix.png" if plot_name else "confusion_matrix.png"
        plt.savefig(save_dir / filename, bbox_inches="tight")
        print(f"Saved plot to '{save_dir / filename}'")

    plt.show()

    return {
        "test_loss_metrics": dict(zip(metric_names, results)),
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "predicted_probabilities": y_pred_probs
    }

# Add function here regarding full execution  of eval functions above with model/history input error handling

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


#############################################################################################################
# Loading pkl data + models:

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
        Directory containing the model (.keras) and history
        (_history.pkl or _history.json) files.

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
    history_pkl_path = model_dir / f"{model_name}_history.pkl"
    history_json_path = model_dir / f"{model_name}_history.json"

    model = tf.keras.models.load_model(model_path)
    print(f"Loaded model from '{model_path}'")

    history_dict = None
    if history_pkl_path.exists():
        with open(history_pkl_path, "rb") as f:
            history_dict = pickle.load(f)
        print(f"Loaded history from '{history_pkl_path}'")
    elif history_json_path.exists():
        with open(history_json_path, "r") as f:
            history_dict = json.load(f)
        print(f"Loaded history from '{history_json_path}'")
    else:
        print(f"No history file found at '{history_pkl_path}' or '{history_json_path}'")

    return model, history_dict

