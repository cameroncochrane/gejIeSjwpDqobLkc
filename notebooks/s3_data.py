from pathlib import Path
import boto3


def upload_to_s3(local_path, bucket_name, s3_key, profile_name=None):
    """
    Upload a local file to an S3 bucket.

    Parameters
    ----------
    local_path : str | Path
        Path to the local file.
    bucket_name : str
        Name of the destination S3 bucket.
    s3_key : str
        Object key (path/name) to use inside the bucket.
        Example: "data/X_train.pkl"
    profile_name : str, optional
        AWS CLI profile to use. If None, boto3 uses the default
        AWS credential chain.
    """
    local_path = Path(local_path)

    if not local_path.is_file():
        raise FileNotFoundError(f"File not found: {local_path}")

    session = boto3.Session(profile_name=profile_name)
    s3 = session.client("s3")

    s3.upload_file(
        Filename=str(local_path),
        Bucket=bucket_name,
        Key=s3_key
    )

    print(f"Uploaded: {local_path} -> s3://{bucket_name}/{s3_key}")


def download_from_s3(bucket_name, s3_key, local_path, profile_name=None):
    """
    Download an object from S3 to the local machine.

    Parameters
    ----------
    bucket_name : str
        Name of the source S3 bucket.
    s3_key : str
        Object key inside the bucket.
        Example: "data/X_train.pkl"
    local_path : str | Path
        Local path where the downloaded file should be saved.
    profile_name : str, optional
        AWS CLI profile to use. If None, boto3 uses the default
        AWS credential chain.
    """
    local_path = Path(local_path)

    # Create parent directories if necessary
    local_path.parent.mkdir(parents=True, exist_ok=True)

    session = boto3.Session(profile_name=profile_name)
    s3 = session.client("s3")

    s3.download_file(
        Bucket=bucket_name,
        Key=s3_key,
        Filename=str(local_path)
    )

    print(f"Downloaded: s3://{bucket_name}/{s3_key} -> {local_path}")