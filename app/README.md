# MonReader Streamlit App

A showcase UI for the best trained SFM (Single Frame Method) model, `model_3_2_3`
(job `tensorflow-monreader-model-20260912-141534`). Two tabs:

- **Test Set Evaluation** — runs the model live over the full 597-image held-out
  test set and reports accuracy, macro F1, AUC, a confusion matrix, a classification
  report, and a browsable gallery of individual predictions.
- **Try Your Own Image** — upload a photo and get a live `flip` / `notflip` prediction.

See [`references/project_summary.md`](../references/project_summary.md) for the full
modeling history. In short: a Sobel edge-detection preprocessing variant was tried
and **dropped** (it discarded information the model needed); the final model was
trained on plain grayscale frames, and this app's preprocessing matches that exactly.

## Why there's a bundled `assets/sfm_test_set.npz`

`data/` is gitignored project-wide (see the root `.gitignore`), so the processed
test set isn't available in a fresh clone or on Streamlit Community Cloud. This app
ships its own compact copy of just the test split — grayscale images as uint8
(`(597, 480, 270)`) plus labels — derived losslessly from
`data/processed/sfm/sfm_processed_data_gray.pkl`. It was generated once with:

```python
import pickle, numpy as np

with open("data/processed/sfm/sfm_processed_data_gray.pkl", "rb") as f:
    data = pickle.load(f)

images_uint8 = (data["X_test"].squeeze(-1) * 255.0).round().astype(np.uint8)
labels = data["y_test"].astype(np.int8)
np.savez_compressed("app/assets/sfm_test_set.npz", images=images_uint8, labels=labels)
```

Re-run that (with an updated `sfm_processed_data_gray.pkl`) only if the test split
itself ever changes. The trained model file is **not** duplicated here — the app
reads it directly from `models/sfm/aws_trained/<job_name>/model_3_2_3.keras`, which
is already tracked in git (unlike `data/`, `models/` is not gitignored).

## Run locally

From the repo root:

```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

This needs the model file at
`models/sfm/aws_trained/tensorflow-monreader-model-20260912-141534/model_3_2_3.keras`
to be present, which it already is in this repo.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (the model file and `app/` must be committed — check
   with `git status` / `git ls-files app/ models/`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in, then **New app**.
3. Pick this repository and branch.
4. Set **Main file path** to `app/streamlit_app.py`.
5. Streamlit Cloud looks for a `requirements.txt` next to the main file first, so
   `app/requirements.txt` should be picked up automatically. If it isn't, set the
   Python dependencies file path explicitly under **Advanced settings**.
6. Deploy. The first build installs TensorFlow and friends, so it can take a few
   minutes.

No AWS credentials or secrets are needed for the deployed app — it only reads the
already-trained `.keras` file and the bundled test-set asset from the repo; it never
calls SageMaker/S3 (that's only `train_sagemaker.py`, a separate local-only workflow).

## Notes / limitations

- The model has a fixed input size (270×480 grayscale) inherited from the training
  pipeline's 1080×1920 source frames resized by 25%. A user-uploaded photo is
  resized directly to 270×480 regardless of its original aspect ratio, so results
  are most reliable on portrait-orientation photos similar to a phone-video frame
  of a page being flipped.
- The 34MB `app/assets/sfm_test_set.npz` is the only sizeable binary this app adds
  to the repo; everything else it depends on (the model) was already tracked.
