# MonReader — Project Summary so far

## 1. Problem & Goal

- **Goal:** detect whether a page is being flipped, from smartphone video of someone flipping through a document.
- Source data: short video clips, each labelled `flip` / `notflip`, exported frame-by-frame to disk as `VideoID_FrameNumber.jpg`.
- **Success metric:** F1 score (chosen over plain accuracy).
- Two candidate modelling strategies identified from the start (NB_1):
  1. **SFM — Single Frame Method:** classify every individual frame independently.
  2. **CM — (Multiple Frame) Clip Method:** aggregate a clip's frames and classify the whole clip.
- Decision: pursue **SFM first** (simpler, more training rows), treat **CM as a separate, later pipeline**.

## 2. Initial EDA (NB_1)

- Images are 1920×1080 RGB (`.jpg`), read via OpenCV (BGR) and converted to RGB.
- Frame counts per clip vary a lot (training: flip clips avg ~18 frames, notflip avg ~24 frames; testing clips much shorter, ~5–6 frames avg).
- **Class imbalance** present — more `flip` clips than `notflip` in training.
- Early call: convert to **grayscale** — colour shouldn't matter for detecting a flip, and it shrinks the data a lot.

## 3. SFM Data Processing (SFM_NB_2)

- Treated every frame as its own independent row (not grouped by clip) — the working assumption being that sequential frames of a clip still each carry useful, slightly-different signal.
- Memory was the main constraint throughout — raw data was ~17–18GB in memory.
  - Converting to grayscale dropped training set memory from ~14GB → ~5GB.
  - Resized images to 25% scale (visually still fine, big further memory/compute saving).
  - Deleted intermediate raw variables (`gc.collect()`) after each transform stage.
- Final processing steps: grayscale → resize (25%) → add explicit channel dim `(H, W, 1)` → normalize pixels to `[0, 1]` float32 → binary-encode labels (`flip`=1) → save as pickles.
- Output: `X_train/y_train/X_test/y_test` saved to `data/processed/sfm/`.

## 4. First Model & Overfitting (SFM_NB_3)

- Shuffled data before splitting (important for SFM specifically — otherwise sequential near-duplicate frames from the same clip cluster together and bias training/validation).
- Built a starter CNN (reused from a prior CV project) — `model_1`.
- **Result: clear overfitting** — training/validation loss diverged early. This became the central problem to solve for the next several notebooks.

## 5. Diagnosing & Addressing Overfitting (SFM_NB_4)

- **Output layer/loss mismatch found:** was using `Dense(2, softmax)` + one-hot labels for what's actually a binary problem — switched to `Dense(1, sigmoid)` + `binary_crossentropy`.
- Sanity-checked the core assumption: *can a single frame actually show a page flip?* Manual inspection of sample images confirmed yes — a single frame captures a page mid-motion, at an angle, usually near image centre — so SFM is viable in principle; the problem is model capacity/generalization, not the framing.
- Tried **image enhancement** to make edges more distinct before augmentation:
  - Brightness/contrast/CLAHE tuning — didn't feel worth the complexity.
  - **Sobel edge-detection filter** — suppresses flat background, highlights page edges/text/hand boundaries. This became `model_2_1`'s input.
- **Moved training to AWS SageMaker** (`ml.g5.xlarge`) at this point — 5–10× faster than local hardware. From here on, model *architecture* is defined/explained in the notebooks but actual training happens on AWS, with results downloaded back for local evaluation. (Training script: `train_sagemaker.py`.)
- Added **data augmentation** (rotation, translation, zoom, contrast, Gaussian noise — brightness deliberately excluded for the Sobel-image models, since it artificially lifts the black background) — `model_2_2`.
- Still overfitting / poor class separation. Suspected cause: `GlobalAveragePooling2D()` discards spatial location — and location (where the hand/page-edge is) is likely important for this problem.

## 6. Rebuilding the Architecture (SFM_NB_5)

- Removed `GlobalAveragePooling2D()`, ran ablations: no-DA vs with-DA (`model_2_3`, `model_2_4`) — still not distinguishing classes well; predicted probabilities had collapsed toward one class.
- Sanity-checked labels/class counts to rule out a data bug — data was fine.
- Floated (not yet tried) idea: feed grayscale **and** Sobel as two separate channels, to recover information lost in the Sobel transform.
- **`model_3`: architecture rethink** — deeper hierarchical CNN, `SpatialDropout2D` (drops whole feature channels, not individual pixels), L2 regularization for the small dataset. Overfitting reduced but predictions still weak on Sobel-processed images.
- **Key pivot: went back to plain grayscale (pre-Sobel) images** with the `model_3` architecture → `model_3_2`. This immediately worked much better (F1 ≈ 70%), **confirming the suspicion that Sobel filtering was discarding useful information**, not just noise.
- Noticed early stopping (patience=5) was cutting training short before it converged — increased patience to 10 → `model_3_2_2` → **F1 jumped to ~98%**.
- Added `ReduceLROnPlateau` alongside early stopping to smooth out erratic validation loss → `model_3_2_3` (most recent run) → **F1 ≈ 99%, test accuracy ≈ 99%**, no signs of overfitting (best weights at epoch 29). **This is the current best SFM model.**

## 7. Current Data/Model Assets

- Two processed SFM datasets exist in `data/processed/sfm/`:
  - `sfm_processed_data_sobel.pkl` — Sobel-enhanced, **experimental, not used for the final model**.
  - `sfm_processed_data_gray.pkl` — plain grayscale, **used to train the final best SFM model** (`model_3_2_3`).
- All AWS-trained model runs (architectures, histories, metrics) are kept under `models/sfm/aws_trained/<job_name>/` for traceability/reproducibility.

## 8. Codebase Restructure (supporting engineering work, done alongside the notebooks)

- Split the `monreader` package into three parts to keep SFM and CM as genuinely independent pipelines:
  - `monreader/sfm/` — single-frame pipeline (config, dataset, features, plots, modeling).
  - `monreader/cm/` — whole-clip pipeline (same shape, currently scaffolding only).
  - `monreader/utils/` — shared AWS scripts (S3 upload/download, SageMaker training workflow).
- Reinforced Cookiecutter Data Science (CCDS) compatibility — fixed several path/import bugs introduced by the split and verified both pipelines run end-to-end (see `monreader/README.md` for the structural breakdown).

## 9. Where CM (Clip Method) Stands

- Currently in the plnning and theory learning stage.
- Is it needed based on how well the final SFM model performs?