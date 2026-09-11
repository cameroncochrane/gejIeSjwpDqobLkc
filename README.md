# MonReader

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A mobile document digitization application powered by computer vision

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   │   ├── sfm        <- Processed data for the single-frame (SFM) model, see note below.
│   │   └── cm         <- Processed data for the whole-clip (CM) model.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│   ├── sfm            <- Models trained by the sfm pipeline.
│   └── cm             <- Models trained by the cm pipeline.
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         monreader and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── monreader   <- Source code for use in this project. See monreader/README.md for details.
    │
    ├── __init__.py             <- Makes monreader a Python module
    │
    ├── sfm                     <- Single-frame classification pipeline
    │   ├── __init__.py
    │   ├── config.py           <- Store useful variables and configuration
    │   ├── dataset.py          <- Scripts to download or generate data
    │   ├── features.py         <- Code to create features for modeling
    │   ├── plots.py            <- Code to create visualizations
    │   └── modeling
    │       ├── __init__.py
    │       ├── predict.py      <- Code to run model inference with trained models
    │       └── train.py        <- Code to train models
    │
    ├── cm                      <- Whole-clip classification pipeline (mirrors sfm/ above)
    │   └── ...
    │
    └── utils                   <- AWS scripts (S3, SageMaker training) shared by sfm/ and cm/
        ├── __init__.py
        ├── s3_data.py
        └── train_sagemaker.py
```

## Processed Data (SFM)

`data/processed/sfm/` contains two pickled datasets produced while developing the single-frame (SFM) model (see [`monreader/README.md`](monreader/README.md) and [`notebooks/SFM_NB_4.ipynb`](notebooks/SFM_NB_4.ipynb) / [`notebooks/SFM_NB_5.ipynb`](notebooks/SFM_NB_5.ipynb) for the full walkthrough):

- **`sfm_processed_data_sobel.pkl`** — grayscale frames with a Sobel edge-detection filter applied on top, to emphasise page edges over flat background/page content. This was an experimental variant and was **not** used to train the final best SFM model.
- **`sfm_processed_data_gray.pkl`** — the standard grayscale frames (resized and normalized), *before* Sobel enhancement. This is the dataset that was used to train the final best SFM model.

--------

