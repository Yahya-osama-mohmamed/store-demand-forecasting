"""Create the S3 bucket (if needed) and upload the Lambda model artifacts.

S3 here is the artifact store that decouples training from serving:
retraining locally + re-running this script rolls the Lambda forward with
zero code changes (it reads the latest artifacts on next cold start).

Run:  python -m aws.upload_model <bucket-name> [region]
"""

import sys
from pathlib import Path

import boto3

from aws.create_budget_guard import require_budget

BUILD_DIR = Path(__file__).resolve().parent / "build"


def main() -> None:
    require_budget()
    if len(sys.argv) < 2:
        sys.exit("usage: python -m aws.upload_model <bucket-name> [region]")
    bucket = sys.argv[1]
    region = sys.argv[2] if len(sys.argv) > 2 else "eu-central-1"

    for f in ("model_xgb.json", "artifacts.json"):
        if not (BUILD_DIR / f).exists():
            sys.exit(f"missing {f} — run: python -m aws.export_for_lambda")

    s3 = boto3.client("s3", region_name=region)
    existing = [b["Name"] for b in s3.list_buckets()["Buckets"]]
    if bucket not in existing:
        kwargs = {"Bucket": bucket}
        if region != "us-east-1":
            kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
        s3.create_bucket(**kwargs)
        s3.put_public_access_block(  # private bucket, belt and braces
            Bucket=bucket,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls": True, "IgnorePublicAcls": True,
                "BlockPublicPolicy": True, "RestrictPublicBuckets": True,
            })
        print(f"created private bucket {bucket} in {region}")

    for f in ("model_xgb.json", "artifacts.json"):
        s3.upload_file(str(BUILD_DIR / f), bucket, f"models/{f}")
        print(f"uploaded s3://{bucket}/models/{f}")


if __name__ == "__main__":
    main()
