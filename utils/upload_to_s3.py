import os
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
import logging

logger = logging.getLogger(__name__)

def upload_to_s3(local_file_path, s3_path):
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    bucket_name = os.getenv("AWS_STORAGE_BUCKET_NAME")
    try:
        logger.info(f"Uploading {local_file_path} to s3://{bucket_name}/{s3_path}")
        s3_client.upload_file(local_file_path, bucket_name, s3_path)
        logger.info("Upload successful")
    except (NoCredentialsError, PartialCredentialsError) as e:
        logger.error(f"Credentials error: {e}")
        raise e
    except ClientError as e:
        logger.error(f"Authorization error: {e}")
        raise e