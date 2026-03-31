import os
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

R2_ENDPOINT    = os.getenv("R2_ENDPOINT_URL")
R2_ACCESS_KEY  = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_KEY  = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET      = os.getenv("R2_BUCKET_NAME")
R2_PUBLIC_URL  = os.getenv("R2_PUBLIC_URL", "")

_r2_client = None


def get_r2_client():
    global _r2_client
    if _r2_client is None:
        if not all([R2_ENDPOINT, R2_ACCESS_KEY, R2_SECRET_KEY, R2_BUCKET]):
            raise RuntimeError("Missing R2 environment variables. Check R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME in .env")
        _r2_client = boto3.client(
            "s3",
            endpoint_url=R2_ENDPOINT,
            aws_access_key_id=R2_ACCESS_KEY,
            aws_secret_access_key=R2_SECRET_KEY,
            region_name="auto",
        )
    return _r2_client


def upload_video_to_r2(local_path: str, object_key: str) -> str:
    """
    Upload a video file from local_path to R2 under object_key.
    Returns the public URL of the uploaded file.
    """
    client = get_r2_client()
    with open(local_path, "rb") as f:
        client.put_object(
            Bucket=R2_BUCKET,
            Key=object_key,
            Body=f,
            ContentType="video/mp4",
        )
    if R2_PUBLIC_URL:
        return f"{R2_PUBLIC_URL.rstrip('/')}/{object_key}"
    return f"{R2_ENDPOINT}/{R2_BUCKET}/{object_key}"


def delete_from_r2(object_key: str) -> bool:
    """Delete an object from R2. Returns True on success."""
    try:
        client = get_r2_client()
        client.delete_object(Bucket=R2_BUCKET, Key=object_key)
        return True
    except ClientError as e:
        print(f"R2 delete error: {e}")
        return False


def build_r2_key(camera_name: str, filename: str) -> str:
    """Build a consistent R2 object key: recordings/<camera_name>/<filename>"""
    import re
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", camera_name)
    return f"recordings/{safe_name}/{filename}"
