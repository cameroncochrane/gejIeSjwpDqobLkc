"""
train_sagemaker.py

Single-file TensorFlow + Amazon SageMaker training workflow.

LOCAL EXECUTION
---------------
Run from Windows:

    python train_sagemaker.py

The script will:

1. Authenticate using your existing AWS credentials/profile.
2. Upload processed training data to S3.
3. Package itself as source.tar.gz.
4. Upload the source package to S3.
5. Create a SageMaker TensorFlow training job.
6. Wait for the job to finish.
7. Download model.tar.gz.
8. Extract the trained Keras model locally.

Run as this when you don't want to re-upload data:

    python train_sagemaker.py --skip-data-upload


SAGEMAKER EXECUTION
-------------------
When SageMaker runs this same script, it detects the SageMaker
environment and runs train_model() instead of submitting another job.


"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import tarfile
import tempfile
import time

from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


# =====================================================================
# USER CONFIGURATION
# =====================================================================

REGION = "eu-west-2"

# Existing S3 bucket.
# Do NOT include "s3://"
S3_BUCKET = "amazon-sagemaker-934912969749-eu-west-2-40xm5p8enanggl"

# Everything created by this script will go underneath this prefix.
S3_PREFIX = "tensorflow-training"

# SageMaker EXECUTION ROLE, not your IAM user ARN.
SAGEMAKER_ROLE_ARN = "arn:aws:iam::934912969749:role/service-role/AmazonSageMakerAdminIAMExecutionRole"

# GPU instance
INSTANCE_TYPE = "ml.g5.xlarge"
INSTANCE_COUNT = 1

# EBS storage attached to the training instance.
VOLUME_SIZE_GB = 50

# Kill the training job if it exceeds this duration.
MAX_RUNTIME_SECONDS = 4 * 60 * 60

# TensorFlow Deep Learning Container.
#
# This example uses the current SageMaker TensorFlow 2.21 GPU DLC.
TRAINING_IMAGE_URI = f"763104351884.dkr.ecr.{REGION}.amazonaws.com/tensorflow-training:2.21.0-gpu-py312-cu129-amzn2023-sagemaker"

PROJECT_NAME = "tensorflow-monreader-model"

# Local paths are relative to this Python file.
PROJECT_ROOT = Path().resolve().parents[0]

LOCAL_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "sfm"
LOCAL_MODEL_DIR = PROJECT_ROOT / "models" / "sfm"

MODEL_NAME = "model_2_1_sm"

MODEL_FILENAME = MODEL_NAME + ".keras"
HISTORY_FILENAME = MODEL_NAME + "_history.json"

# ---------------------------------------------------------------------
# Expected processed dataset
# ---------------------------------------------------------------------
#
# data/processed/sfm/
# └── sfm_processed_data.pkl
#
# If validation arrays aren't present, validation_split is used.


DATA_FILE = "sfm_processed_data.pkl"

# ---------------------------------------------------------------------
# Training parameters
# ---------------------------------------------------------------------

EPOCHS = 10
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.20  # Shouldn't need this

# How frequently your Windows script checks SageMaker.
STATUS_POLL_SECONDS = 20

# CloudWatch log group SageMaker publishes container stdout/stderr to.
LOG_GROUP_NAME = "/aws/sagemaker/TrainingJobs"

# How many recent log lines to print if a job fails.
FAILURE_LOG_TAIL_LINES = 40


# =====================================================================
# SAGEMAKER DETECTION
# =====================================================================

def running_inside_sagemaker() -> bool:
    """SageMaker exposes these environment variables inside the training container."""
    return "SM_MODEL_DIR" in os.environ and "SM_CHANNEL_TRAINING" in os.environ


# =====================================================================
# MODEL DEFINITION
# =====================================================================

def build_model(input_shape):
    """DEFINE THE TENSORFLOW MODEL HERE."""

    import tensorflow as tf

    model = tf.keras.Sequential([
        # Input + Layer 1
        tf.keras.Conv2D(32, 3, padding="same", activation="relu", input_shape=input_shape),
        tf.keras.BatchNormalization(),

        # Layer 2
        tf.keras.Conv2D(32, 3, activation="relu"),
        tf.keras.MaxPooling2D(),
        tf.keras.Dropout(0.2),

        # Layer 3
        tf.keras.Conv2D(128, 3, padding="same", activation="relu"),
        tf.keras.BatchNormalization(),
        tf.keras.MaxPooling2D(),
        tf.keras.Dropout(0.2),

        tf.keras.GlobalAveragePooling2D(),

        # Layer 4
        tf.keras.Dense(32, activation="relu"),
        tf.keras.BatchNormalization(),
        tf.keras.Dropout(0.3),

        # Output
        tf.keras.Dense(1, activation="sigmoid"),
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="auc")])

    return model


# =====================================================================
# SAGEMAKER TRAINING CODE
# =====================================================================

def load_training_data(training_dir: Path):
    """Load the processed pickle dataset supplied through the SageMaker 'training' channel."""

    data_path = training_dir / DATA_FILE

    if not data_path.exists(): raise FileNotFoundError(f"Training data file not found: {data_path}")

    with open(data_path, "rb") as file: data = pickle.load(file)

    if isinstance(data, dict):
        try: X_train = data["X_train"]; y_train = data["y_train"]
        except KeyError as error: raise KeyError("The processed dataset must contain 'X_train' and 'y_train'.") from error
        X_val = data.get("X_val"); y_val = data.get("y_val")
    elif isinstance(data, (tuple, list)) and len(data) >= 2:
        X_train, y_train = data[:2]
        X_val = data[2] if len(data) > 2 else None
        y_val = data[3] if len(data) > 3 else None
    else:
        raise ValueError("Expected sfm_processed_data.pkl to contain a dataset dictionary or tuple.")

    return X_train, y_train, X_val, y_val

def train_model():
    """This function runs INSIDE the SageMaker TensorFlow container."""

    import numpy as np
    import tensorflow as tf

    # ---------------------------------------------------------------
    # Hyperparameters passed to the training container
    # ---------------------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)

    # SageMaker may provide additional arguments.
    args, _ = parser.parse_known_args()

    # ---------------------------------------------------------------
    # SageMaker locations
    # ---------------------------------------------------------------

    training_dir = Path(os.environ["SM_CHANNEL_TRAINING"])
    model_dir = Path(os.environ["SM_MODEL_DIR"])
    output_dir = Path(os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))

    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Environment information
    # ---------------------------------------------------------------

    print("=" * 70)
    print("SAGEMAKER TENSORFLOW TRAINING")
    print("=" * 70)

    print(f"TensorFlow version: {tf.__version__}")
    print(f"Training directory: {training_dir}")
    print(f"Model directory: {model_dir}")

    gpus = tf.config.list_physical_devices("GPU")

    print(f"Detected GPUs: {gpus}")

    if not gpus: print("WARNING: TensorFlow did not detect a GPU.")

    # ---------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------

    X_train, y_train, X_val, y_val = load_training_data(training_dir)

    print()
    print("Dataset")
    print("-------")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")

    if X_val is not None:
        print(f"X_val:   {X_val.shape}")
        print(f"y_val:   {y_val.shape}")

    print()

    # ---------------------------------------------------------------
    # Build model
    # ---------------------------------------------------------------

    model = build_model(input_shape=X_train.shape[1:])

    model.summary()

    # ---------------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------------

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]

    # ---------------------------------------------------------------
    # Train
    # ---------------------------------------------------------------

    fit_kwargs = {"x": X_train, "y": y_train, "epochs": args.epochs, "batch_size": args.batch_size, "callbacks": callbacks, "verbose": 2}

    if X_val is not None: fit_kwargs["validation_data"] = (X_val, y_val)
    else: fit_kwargs["validation_split"] = VALIDATION_SPLIT

    history = model.fit(**fit_kwargs)

    # ---------------------------------------------------------------
    # Save final model
    # ---------------------------------------------------------------
    #
    # SageMaker automatically collects everything under
    # /opt/ml/model and packages it into model.tar.gz.
    #

    model_path = model_dir / MODEL_FILENAME

    model.save(model_path)

    print()
    print(f"Model saved to: {model_path}")

    # ---------------------------------------------------------------
    # Save training history
    # ---------------------------------------------------------------

    serializable_history = {key: [float(value) for value in values] for key, values in history.history.items()}

    history_path = model_dir / HISTORY_FILENAME

    with open(history_path, "w", encoding="utf-8") as file: json.dump(serializable_history, file, indent=4)

    print(f"Training history saved to: {history_path}")

    print()
    print("Training complete.")


# =====================================================================
# LOCAL AWS UTILITIES
# =====================================================================

def create_boto3_session(profile_name=None):
    """Create a boto3 session using the same AWS credential system used by the AWS CLI."""
    import boto3
    kwargs = {"region_name": REGION}
    if profile_name: kwargs["profile_name"] = profile_name
    return boto3.Session(**kwargs)

def validate_local_configuration():
    """Fail early if important configuration hasn't been changed."""
    errors = []
    if S3_BUCKET == "YOUR-S3-BUCKET": errors.append("Set S3_BUCKET at the top of the script.")
    if "YOUR-SAGEMAKER-EXECUTION-ROLE" in SAGEMAKER_ROLE_ARN: errors.append("Set SAGEMAKER_ROLE_ARN at the top of the script.")
    if not LOCAL_DATA_DIR.exists(): errors.append(f"Data directory does not exist: {LOCAL_DATA_DIR}")

    required_files = [LOCAL_DATA_DIR / DATA_FILE]
    for path in required_files:
        if not path.exists(): errors.append(f"Required training file missing: {path}")

    if errors: raise RuntimeError("Configuration errors:\n" + "\n".join(f" - {error}" for error in errors))

def generate_job_name():
    """Produce a unique SageMaker-compatible training job name."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    base = re.sub(r"[^a-z0-9-]", "-", f"{PROJECT_NAME}-{timestamp}".lower()).strip("-")
    return base[:63]


# =====================================================================
# S3 UPLOAD
# =====================================================================

def upload_directory_to_s3(s3_client, local_directory: Path, bucket: str, prefix: str):
    """Recursively upload a local directory to S3."""

    files = [path for path in local_directory.rglob("*") if path.is_file()]

    if not files: raise RuntimeError(f"No files found in {local_directory}")

    print()
    print(f"Uploading {len(files)} data file(s) to S3...")

    for path in files:
        relative_path = path.relative_to(local_directory).as_posix()
        s3_key = f"{prefix.rstrip('/')}/{relative_path}"
        print(f"  {relative_path}")
        s3_client.upload_file(str(path), bucket, s3_key)

    return f"s3://{bucket}/{prefix.rstrip('/')}/"


# =====================================================================
# PACKAGE TRAINING SCRIPT
# =====================================================================

def package_and_upload_source(s3_client, job_name: str):
    """
    Package this script as source.tar.gz and upload it to S3.

    SageMaker Training Toolkit will unpack it and execute this same
    Python file inside the TensorFlow container.
    """

    script_path = Path(__file__).resolve()
    s3_key = f"{S3_PREFIX}/code/{job_name}/source.tar.gz"

    with tempfile.TemporaryDirectory() as temp_dir:
        archive_path = Path(temp_dir) / "source.tar.gz"

        with tarfile.open(archive_path, "w:gz") as archive: archive.add(script_path, arcname=script_path.name)

        print()
        print("Uploading training code...")

        s3_client.upload_file(str(archive_path), S3_BUCKET, s3_key)

    return f"s3://{S3_BUCKET}/{s3_key}"


# =====================================================================
# AWS CONSOLE LINKS
# =====================================================================

def build_console_links(job_name: str):
    """Build direct console URLs for the training job and its CloudWatch log group."""
    training_job_url = f"https://{REGION}.console.aws.amazon.com/sagemaker/home?region={REGION}#/jobs/{job_name}"
    log_group_url = f"https://{REGION}.console.aws.amazon.com/cloudwatch/home?region={REGION}#logsV2:log-groups/log-group/{LOG_GROUP_NAME.replace('/', '$252F')}"
    return training_job_url, log_group_url


# =====================================================================
# CLOUDWATCH LOGS
# =====================================================================

def get_log_streams(logs_client, job_name: str):
    """Return the CloudWatch log streams for a training job, if any exist yet."""
    try:
        response = logs_client.describe_log_streams(logGroupName=LOG_GROUP_NAME, logStreamNamePrefix=f"{job_name}/", orderBy="LogStreamName")
        return response.get("logStreams", [])
    except logs_client.exceptions.ResourceNotFoundException:
        return []

def stream_new_log_events(logs_client, job_name: str, stream_tokens: dict):
    """Print any new CloudWatch log lines since the last call. `stream_tokens` is mutated in place to track position per stream."""
    for stream in get_log_streams(logs_client, job_name):
        stream_name = stream["logStreamName"]
        label = stream_name.split("/")[-1]

        kwargs = {"logGroupName": LOG_GROUP_NAME, "logStreamName": stream_name, "startFromHead": True}
        if stream_name in stream_tokens: kwargs["nextToken"] = stream_tokens[stream_name]

        response = logs_client.get_log_events(**kwargs)

        for event in response.get("events", []): print(f"[{label}] {event['message']}")

        stream_tokens[stream_name] = response["nextForwardToken"]

def print_recent_log_tail(logs_client, job_name: str, max_lines: int = FAILURE_LOG_TAIL_LINES):
    """Print the most recent CloudWatch log lines for a job, to aid debugging on failure."""
    streams = get_log_streams(logs_client, job_name)

    if not streams:
        print("No CloudWatch log streams were found for this job.")
        return

    print()
    print(f"Last {max_lines} log line(s) per instance:")
    print("-" * 70)

    for stream in streams:
        stream_name = stream["logStreamName"]
        label = stream_name.split("/")[-1]
        response = logs_client.get_log_events(logGroupName=LOG_GROUP_NAME, logStreamName=stream_name, limit=max_lines, startFromHead=False)
        for event in response.get("events", []): print(f"[{label}] {event['message']}")


# =====================================================================
# CREATE SAGEMAKER TRAINING JOB
# =====================================================================

def submit_training_job(sm_client, job_name: str, training_data_uri: str, source_uri: str, epochs: int, batch_size: int):
    """Submit the SageMaker CreateTrainingJob request."""

    output_uri = f"s3://{S3_BUCKET}/{S3_PREFIX}/output"

    print()
    print("=" * 70)
    print("SUBMITTING SAGEMAKER TRAINING JOB")
    print("=" * 70)

    print(f"Job:       {job_name}")
    print(f"Instance:  {INSTANCE_TYPE}")
    print(f"Data:      {training_data_uri}")
    print(f"Code:      {source_uri}")
    print(f"Output:    {output_uri}")
    print()

    response = sm_client.create_training_job(
        TrainingJobName=job_name,

        # -----------------------------------------------------------
        # Script-mode parameters
        # -----------------------------------------------------------

        HyperParameters={"sagemaker_program": Path(__file__).name, "sagemaker_submit_directory": source_uri, "sagemaker_container_log_level": "20", "sagemaker_region": REGION, "epochs": str(epochs), "batch-size": str(batch_size)},

        # -----------------------------------------------------------
        # TensorFlow Docker container
        # -----------------------------------------------------------

        AlgorithmSpecification={"TrainingImage": TRAINING_IMAGE_URI, 
                                "TrainingInputMode": "File", 
                                "EnableSageMakerMetricsTimeSeries": True,
                                "MetricDefinitions": [
        { # Metric definitions for model training:
            "Name": "train:loss",
            "Regex": r"loss: ([0-9\.]+)"
        },
        {
            "Name": "validation:loss",
            "Regex": r"val_loss: ([0-9\.]+)"
        },
        {
            "Name": "validation:accuracy",
            "Regex": r"val_accuracy: ([0-9\.]+)"
        }]},

        # -----------------------------------------------------------
        # SageMaker execution role
        # -----------------------------------------------------------

        RoleArn=SAGEMAKER_ROLE_ARN,

        # -----------------------------------------------------------
        # Training-data channel
        # -----------------------------------------------------------

        InputDataConfig=[{"ChannelName": "training", "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": training_data_uri, "S3DataDistributionType": "FullyReplicated"}}, "InputMode": "File"}],

        # -----------------------------------------------------------
        # Model output
        # -----------------------------------------------------------

        OutputDataConfig={"S3OutputPath": output_uri},

        # -----------------------------------------------------------
        # Training hardware
        # -----------------------------------------------------------

        ResourceConfig={"InstanceType": INSTANCE_TYPE, "InstanceCount": INSTANCE_COUNT, "VolumeSizeInGB": VOLUME_SIZE_GB},

        # -----------------------------------------------------------
        # Safety limit
        # -----------------------------------------------------------

        StoppingCondition={"MaxRuntimeInSeconds": MAX_RUNTIME_SECONDS},

        Tags=[{"Key": "Project", "Value": PROJECT_NAME}],
    )

    print("Training job successfully submitted.")

    print(f"ARN: {response['TrainingJobArn']}")

    training_job_url, log_group_url = build_console_links(job_name)

    print(f"Console:   {training_job_url}")
    print(f"Logs:      {log_group_url}")


# =====================================================================
# MONITOR JOB
# =====================================================================

def print_training_time_summary(response: dict):
    """Print how long the job actually ran / was billed for."""
    training_seconds = response.get("TrainingTimeInSeconds")
    billable_seconds = response.get("BillableTimeInSeconds")

    if training_seconds is not None: print(f"Training time: {training_seconds}s ({training_seconds / 60:.1f} min)")
    if billable_seconds is not None: print(f"Billable time: {billable_seconds}s ({billable_seconds / 60:.1f} min) on {INSTANCE_COUNT}x {INSTANCE_TYPE}")

def wait_for_training_job(sm_client, logs_client, job_name: str):
    """Poll SageMaker until the job reaches a terminal state, streaming CloudWatch logs and detailed status as they arrive."""
    from botocore.exceptions import ClientError

    print()
    print("Monitoring training job...")
    print("Press Ctrl+C if you want to stop monitoring.")
    print("The SageMaker job will continue running in AWS.")
    print()

    previous_status = None
    previous_secondary_status = None
    stream_tokens = {}
    logs_enabled = True

    while True:
        response = sm_client.describe_training_job(TrainingJobName=job_name)
        status = response["TrainingJobStatus"]
        secondary_status = response.get("SecondaryStatus")

        if status != previous_status or secondary_status != previous_secondary_status:
            timestamp = datetime.now().strftime("%H:%M:%S")
            transitions = response.get("SecondaryStatusTransitions", [])
            message = transitions[-1].get("StatusMessage") if transitions else None
            detail = f" - {message}" if message else ""
            print(f"[{timestamp}] {status} / {secondary_status}{detail}")
            previous_status = status
            previous_secondary_status = secondary_status

        if logs_enabled:
            try:
                stream_new_log_events(logs_client, job_name, stream_tokens)
            except ClientError as error:
                logs_enabled = False
                print(f"WARNING: Unable to read CloudWatch logs ({error.response['Error']['Code']}: {error.response['Error']['Message']}).")
                print("Falling back to status-only polling. Grant logs:DescribeLogStreams and logs:GetLogEvents to see live training output.")

        if status == "Completed":
            print()
            print_training_time_summary(response)
            return response

        if status in {"Failed", "Stopped"}:
            failure_reason = response.get("FailureReason", "No failure reason returned.")
            print()
            print("=" * 70)
            print(f"TRAINING JOB {status.upper()}")
            print("=" * 70)
            print(f"FailureReason: {failure_reason}")
            if logs_enabled:
                try:
                    print_recent_log_tail(logs_client, job_name)
                except ClientError as error:
                    print(f"WARNING: Unable to fetch CloudWatch log tail ({error.response['Error']['Code']}: {error.response['Error']['Message']}).")
            raise RuntimeError(f"SageMaker job {status}:\n{failure_reason}")

        time.sleep(STATUS_POLL_SECONDS)


# =====================================================================
# DOWNLOAD MODEL
# =====================================================================

def parse_s3_uri(s3_uri: str):
    """
    Convert:
        s3://bucket/path/file
    into:
        bucket
        path/file
    """

    parsed = urlparse(s3_uri)

    if parsed.scheme != "s3": raise ValueError(f"Not an S3 URI: {s3_uri}")

    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    return bucket, key


def safe_extract_tar(archive_path: Path, destination: Path):
    """Safely extract a tar archive, rejecting path traversal."""

    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()

    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if target != destination_resolved and destination_resolved not in target.parents: raise RuntimeError("Unsafe path detected inside model archive.")
        archive.extractall(destination)


def download_and_extract_model(s3_client, job_name: str, training_description: dict):
    """Download SageMaker's model.tar.gz and extract it locally."""

    artifact_uri = training_description["ModelArtifacts"]["S3ModelArtifacts"]

    print()
    print("=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print(f"Model artifact: {artifact_uri}")

    bucket, key = parse_s3_uri(artifact_uri)

    job_model_dir = LOCAL_MODEL_DIR / job_name

    job_model_dir.mkdir(parents=True, exist_ok=True)

    archive_path = job_model_dir / "model.tar.gz"

    print()
    print("Downloading model artifact...")

    s3_client.download_file(bucket, key, str(archive_path))

    print(f"Downloaded to: {archive_path}")

    print()
    print("Extracting model...")

    safe_extract_tar(archive_path, job_model_dir)

    print(f"Extracted to: {job_model_dir}")

    keras_models = list(job_model_dir.rglob("*.keras"))

    if keras_models:
        print()
        print("Keras model:")
        for model_path in keras_models: print(f"  {model_path}")
    else:
        print("WARNING: No .keras model was found in the artifact.")

    return job_model_dir


# =====================================================================
# LOCAL ORCHESTRATION
# =====================================================================

def local_workflow():
    """Runs on your Windows PC."""

    parser = argparse.ArgumentParser(description="Submit TensorFlow training to Amazon SageMaker.")
    parser.add_argument("--profile", default=None, help="Optional AWS CLI profile. Otherwise the standard boto3/AWS_PROFILE credential chain is used.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--no-wait", action="store_true", help="Submit the job and exit immediately instead of waiting and downloading the model.")
    parser.add_argument("--skip-data-upload", action="store_true", help="Reuse the existing dataset already stored under the configured S3 data prefix.")

    args = parser.parse_args()

    validate_local_configuration()

    # Import boto3 locally only.
    # The SageMaker training process does not require this path.
    try:
        session = create_boto3_session(args.profile)
    except ImportError as error:
        raise RuntimeError("boto3 is not installed locally.\n\n Install it with:\n pip install boto3") from error

    s3 = session.client("s3")
    sm = session.client("sagemaker")
    sts = session.client("sts")
    logs = session.client("logs")

    # ---------------------------------------------------------------
    # Verify AWS identity
    # ---------------------------------------------------------------

    identity = sts.get_caller_identity()

    print("=" * 70)
    print("AWS IDENTITY")
    print("=" * 70)

    print(f"Account: {identity['Account']}")
    print(f"Identity: {identity['Arn']}")
    print(f"Region: {REGION}")

    # ---------------------------------------------------------------
    # Create unique job name
    # ---------------------------------------------------------------

    job_name = generate_job_name()

    # ---------------------------------------------------------------
    # Upload dataset
    # ---------------------------------------------------------------

    data_prefix = f"{S3_PREFIX}/data"
    training_data_uri = f"s3://{S3_BUCKET}/{data_prefix}/"

    if not args.skip_data_upload:
        training_data_uri = upload_directory_to_s3(s3_client=s3, local_directory=LOCAL_DATA_DIR, bucket=S3_BUCKET, prefix=data_prefix)
    else:
        print()
        print("Skipping data upload.")
        print(f"Using: {training_data_uri}")

    # ---------------------------------------------------------------
    # Package and upload this script
    # ---------------------------------------------------------------

    source_uri = package_and_upload_source(s3_client=s3, job_name=job_name)

    # ---------------------------------------------------------------
    # Submit job
    # ---------------------------------------------------------------

    submit_training_job(sm_client=sm, job_name=job_name, training_data_uri=training_data_uri, source_uri=source_uri, epochs=args.epochs, batch_size=args.batch_size)

    # ---------------------------------------------------------------
    # Optional detached execution
    # ---------------------------------------------------------------

    if args.no_wait:
        print(); print("Job submitted."); print("The training job will continue running in AWS."); print(); print("Job name:"); print(f"    {job_name}"); return

    # ---------------------------------------------------------------
    # Wait for completion
    # ---------------------------------------------------------------

    try:
        description = wait_for_training_job(sm_client=sm, logs_client=logs, job_name=job_name)
    except KeyboardInterrupt:
        print(); print(); print("Local monitoring stopped."); print("The SageMaker training job is still running."); print(f"Job name: {job_name}"); return

    # ---------------------------------------------------------------
    # Download model
    # ---------------------------------------------------------------

    local_model_path = download_and_extract_model(s3_client=s3, job_name=job_name, training_description=description)

    print()
    print("=" * 70)
    print("WORKFLOW COMPLETE")
    print("=" * 70)

    print(f"Local model directory:\n{local_model_path}")


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    if running_inside_sagemaker():
        # Running on the AWS GPU instance.
        train_model()
    else:
        # Running on your Windows machine.
        local_workflow()
