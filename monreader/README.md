# monreader

Source code for the MonReader project, split into two independent modeling approaches. Both were considered in [`notebooks/NB_1.ipynb`](../notebooks/NB_1.ipynb), which frames the choice as:

1. **Single Frame Method (SFM)** — treat every extracted frame as its own row and predict `flip` / `notflip` from a single image.
2. **Multiple Frame Method (MFM)**, implemented here as **CM** — aggregate all the frames belonging to one clip and predict `flip` / `notflip` for the whole clip.

## `sfm/` — single-frame classification

The working, end-to-end pipeline. Development is documented across [`notebooks/SFM_NB_2.ipynb`](../notebooks/SFM_NB_2.ipynb) through [`notebooks/SFM_NB_5.ipynb`](../notebooks/SFM_NB_5.ipynb): raw frames are loaded, converted to grayscale, resized/normalized, and saved to `data/processed/sfm/` before being trained (on AWS SageMaker, see `utils/` below) and evaluated. See the top-level [README.md](../README.md) for a note on the two processed datasets this pipeline produced.

- `config.py` — paths and shared configuration.
- `dataset.py` — loading/generating the SFM dataset.
- `features.py` — feature/image preprocessing for modeling.
- `plots.py` — visualizations (training curves, sample images, etc.).
- `modeling/train.py` — model training entry point.
- `modeling/predict.py` — inference with a trained SFM model.

## `cm/` — whole-clip classification

The counterpart to `sfm/` for the Multiple Frame Method: instead of classifying individual frames, a clip's frames are aggregated and classified together. This module currently mirrors the `sfm/` module's structure as unimplemented scaffolding (`dataset.py`, `features.py`, `plots.py`, `modeling/train.py`, `modeling/predict.py`) — no CM notebook or trained model exists yet, and `sfm/` remains the only modeling approach that has been carried through to a trained model.

## `utils/` — AWS scripts

Helper scripts used by the `sfm/` (and, eventually, `cm/`) training workflow to interact with AWS, since models are trained remotely on SageMaker rather than locally:

- `s3_data.py` — `upload_to_s3` / `download_from_s3` helpers for moving processed datasets and model artifacts to/from an S3 bucket.
- `train_sagemaker.py` — packages this project, uploads processed training data and source code to S3, submits and monitors a SageMaker TensorFlow training job, then downloads and extracts the resulting trained model. Can be run locally (it detects whether it's running inside SageMaker or on a local machine, e.g. via `python train_sagemaker.py`) or is invoked by the SageMaker training job itself.
