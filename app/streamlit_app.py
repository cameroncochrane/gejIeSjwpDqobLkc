"""
streamlit_app.py

MonReader - Streamlit showcase for the best trained SFM (Single Frame Method)
page-flip detector: model_3_2_3, a custom CNN trained on grayscale frames.

Preprocessing note
-------------------
A Sobel edge-detection preprocessing variant was tried during development
(see notebooks/SFM_NB_4.ipynb, notebooks/SFM_NB_5.ipynb and
references/project_summary.md) but was dropped - it discarded information the
model needed and hurt performance. The final model was trained on plain
grayscale frames instead, and this app reproduces exactly that pipeline:
grayscale -> resize to 270x480 -> normalize to [0, 1].

Run locally (from the repo root):
    pip install -r app/requirements.txt
    streamlit run app/streamlit_app.py

See app/README.md for Streamlit Community Cloud deployment steps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

# =====================================================================
# CONFIGURATION
# =====================================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

MODEL_JOB_NAME = "tensorflow-monreader-model-20260912-141534"
MODEL_FILE_NAME = "model_3_2_3.keras"
MODEL_PATH = PROJECT_ROOT / "models" / "sfm" / "aws_trained" / MODEL_JOB_NAME / MODEL_FILE_NAME

TEST_SET_PATH = APP_DIR / "assets" / "sfm_test_set.npz"

IMG_HEIGHT = 480
IMG_WIDTH = 270

CLASS_NAMES = ("notflip", "flip")  # label 0, label 1 - matches training's encode_labels()
PREDICTION_THRESHOLD = 0.5


# =====================================================================
# CACHED LOADERS
# =====================================================================

@st.cache_resource(show_spinner="Loading trained model...")
def load_model() -> tf.keras.Model:
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data(show_spinner="Loading held-out test set...")
def load_test_set() -> tuple[np.ndarray, np.ndarray]:
    with np.load(TEST_SET_PATH) as npz:
        return npz["images"], npz["labels"]


@st.cache_data(show_spinner="Running the model over the test set...")
def predict_test_set(_model: tf.keras.Model, images_uint8: np.ndarray, labels: np.ndarray) -> np.ndarray:
    x = images_uint8.astype(np.float32)[..., np.newaxis] / 255.0
    return _model.predict(x, verbose=0).ravel()


# =====================================================================
# PREPROCESSING (for a user-uploaded image)
# =====================================================================

def preprocess_uploaded_image(pil_image: Image.Image) -> np.ndarray:
    """
    Reproduce the SFM training pipeline for one arbitrary image:
    grayscale -> resize to (IMG_WIDTH, IMG_HEIGHT) -> normalize to [0, 1].

    Training frames were 1080x1920 and resized by a fixed 25% scale factor,
    which lands on exactly (270, 480). A user photo can be any size/aspect
    ratio, so it's resized directly to that target instead - see the caveat
    shown in the "Try your own image" tab.
    """
    grayscale = pil_image.convert("L")
    resized = grayscale.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.float32) / 255.0


def predict_single(model: tf.keras.Model, normalized_image: np.ndarray) -> float:
    batch = normalized_image[np.newaxis, ..., np.newaxis]
    return float(model.predict(batch, verbose=0).ravel()[0])


# =====================================================================
# METRICS DISPLAY
# =====================================================================

def render_confusion_matrix(cm: np.ndarray, class_names=CLASS_NAMES):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    return fig


# =====================================================================
# PAGE SETUP
# =====================================================================

st.set_page_config(page_title="MonReader - Page Flip Detector", page_icon="📖", layout="wide")

st.title("📖 MonReader — Page Flip Detector")
st.caption("Single-Frame Method (SFM): a custom CNN classifies whether one photo shows a page mid-flip.")

if not MODEL_PATH.exists():
    st.error(f"Model file not found at `{MODEL_PATH}`. Make sure it's committed to the repo.")
    st.stop()

if not TEST_SET_PATH.exists():
    st.error(f"Test set asset not found at `{TEST_SET_PATH}`. Make sure it's committed to the repo.")
    st.stop()

model = load_model()

with st.sidebar:
    st.header("About this model")
    st.markdown(
        f"""
**Model:** `{MODEL_FILE_NAME}` (`model_3_2_3`)

**Training job:** `{MODEL_JOB_NAME}`

**Input:** {IMG_HEIGHT}×{IMG_WIDTH} grayscale image

**Params:** {model.count_params():,}
        """
    )
    st.markdown(
        "A **Sobel edge-detection** preprocessing variant was tried during "
        "development but was dropped - it discarded information the model "
        "needed and hurt performance. This model was trained on plain "
        "**grayscale** frames instead, which worked far better."
    )
    with st.expander("Model architecture"):
        summary_lines: list[str] = []
        model.summary(print_fn=summary_lines.append)
        st.code("\n".join(summary_lines), language=None)

tab_eval, tab_upload = st.tabs(["🧪 Test Set Evaluation", "🖼️ Try Your Own Image"])

# ---------------------------------------------------------------
# Tab 1: Test set evaluation
# ---------------------------------------------------------------
with tab_eval:
    st.subheader("Live evaluation on the held-out SFM test set")
    st.write(
        "Runs `model_3_2_3` over all 597 held-out test frames (never seen during "
        "training) and reports the results below."
    )

    images_uint8, y_true = load_test_set()
    y_prob = predict_test_set(model, images_uint8, y_true)
    y_pred = (y_prob >= PREDICTION_THRESHOLD).astype(int)

    accuracy = float(np.mean(y_pred == y_true))
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    auc = roc_auc_score(y_true, y_prob)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy:.1%}")
    col2.metric("Macro F1", f"{macro_f1:.1%}")
    col3.metric("AUC", f"{auc:.3f}")
    col4.metric("Test images", f"{len(y_true)}")

    col_cm, col_report = st.columns([1, 1.4])
    with col_cm:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y_true, y_pred)
        st.pyplot(render_confusion_matrix(cm))
    with col_report:
        st.markdown("**Classification report**")
        report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
        st.code(report, language=None)

    st.divider()
    st.markdown("**Browse individual predictions**")
    st.caption(
        "Images are shown exactly as the model sees them: grayscale, "
        f"{IMG_WIDTH}×{IMG_HEIGHT}, downscaled from the original phone-video frame."
    )

    filter_choice = st.radio(
        "Filter", ["All", "Correct only", "Incorrect only"], horizontal=True, key="eval_filter"
    )
    correct_mask = y_pred == y_true
    if filter_choice == "Correct only":
        candidate_idx = np.where(correct_mask)[0]
    elif filter_choice == "Incorrect only":
        candidate_idx = np.where(~correct_mask)[0]
    else:
        candidate_idx = np.arange(len(y_true))

    if len(candidate_idx) == 0:
        st.info("No images match this filter.")
    else:
        max_examples = min(24, len(candidate_idx))
        n_examples = st.slider("Examples to show", 4, max_examples, min(12, max_examples), key="eval_n")
        shown_idx = candidate_idx[:n_examples]

        n_cols = 6
        cols = st.columns(n_cols)
        for position, idx in enumerate(shown_idx):
            true_name = CLASS_NAMES[y_true[idx]]
            pred_name = CLASS_NAMES[y_pred[idx]]
            confidence = y_prob[idx] if y_pred[idx] == 1 else 1 - y_prob[idx]
            mark = "✅" if y_pred[idx] == y_true[idx] else "❌"
            with cols[position % n_cols]:
                st.image(images_uint8[idx], use_container_width=True)
                st.caption(f"{mark} True: {true_name} | Pred: {pred_name} ({confidence:.0%})")

# ---------------------------------------------------------------
# Tab 2: Upload your own image
# ---------------------------------------------------------------
with tab_upload:
    st.subheader("Try the model on your own photo")
    st.write(
        "Upload a photo and the model will predict whether it shows a page "
        "mid-flip (`flip`) or not (`notflip`)."
    )
    st.caption(
        "The model was trained only on frames extracted from portrait-orientation "
        f"phone video (resized to {IMG_WIDTH}×{IMG_HEIGHT} grayscale). Your photo is "
        "resized directly to that size regardless of its original aspect ratio, so "
        "portrait photos similar to a page-flip video frame will get the most "
        "reliable predictions."
    )

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        original_image = Image.open(uploaded_file)
        normalized = preprocess_uploaded_image(original_image)
        probability = predict_single(model, normalized)
        predicted_label = CLASS_NAMES[int(probability >= PREDICTION_THRESHOLD)]
        confidence = probability if probability >= PREDICTION_THRESHOLD else 1 - probability

        col_original, col_processed = st.columns(2)
        with col_original:
            st.image(original_image, caption="Original upload", use_container_width=True)
        with col_processed:
            st.image(normalized, caption=f"What the model sees ({IMG_WIDTH}×{IMG_HEIGHT} grayscale)", use_container_width=True)

        if predicted_label == "flip":
            st.success(f"Prediction: **FLIP** — {confidence:.1%} confidence")
        else:
            st.warning(f"Prediction: **NOTFLIP** — {confidence:.1%} confidence")
    else:
        st.info("Upload a JPG or PNG image above to get a prediction.")
