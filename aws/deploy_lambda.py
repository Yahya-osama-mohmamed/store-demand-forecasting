"""Create/update the Lambda function + its least-privilege IAM role.

Run:  python -m aws.deploy_lambda <bucket-name> [region]
"""

import json
import sys
import time
from pathlib import Path

import boto3

from aws.create_budget_guard import require_budget

HERE = Path(__file__).resolve().parent
FUNCTION_NAME = "demand-forecast"
ROLE_NAME = "LambdaForecastingRole"


def ensure_role(iam, bucket: str) -> str:
    policy = json.loads((HERE / "iam_policy.json").read_text())
    policy["Statement"][0]["Resource"] = f"arn:aws:s3:::{bucket}/models/*"
    trust = json.loads((HERE / "trust_policy.json").read_text())

    try:
        role = iam.get_role(RoleName=ROLE_NAME)["Role"]
        print(f"role {ROLE_NAME} exists")
    except iam.exceptions.NoSuchEntityException:
        role = iam.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust),
            Description="Least-privilege role for the forecast Lambda",
        )["Role"]
        print(f"created role {ROLE_NAME}")
        time.sleep(10)  # IAM propagation

    iam.put_role_policy(RoleName=ROLE_NAME, PolicyName="forecast-s3-logs",
                        PolicyDocument=json.dumps(policy))
    return role["Arn"]


def main() -> None:
    require_budget()
    if len(sys.argv) < 2:
        sys.exit("usage: python -m aws.deploy_lambda <bucket-name> [region]")
    bucket = sys.argv[1]
    region = sys.argv[2] if len(sys.argv) > 2 else "eu-central-1"

    zip_path = HERE / "lambda_package.zip"
    if not zip_path.exists():
        sys.exit("missing lambda_package.zip — run: bash aws/build_lambda_package.sh")

    iam = boto3.client("iam")
    lam = boto3.client("lambda", region_name=region)
    role_arn = ensure_role(iam, bucket)

    config = dict(
        Runtime="python3.11",
        Role=role_arn,
        Handler="lambda_function.lambda_handler",
        Timeout=10,
        MemorySize=512,
        Environment={"Variables": {"MODEL_BUCKET": bucket,
                                   "MODEL_PREFIX": "models"}},
    )
    # Packages over 50MB zipped must be referenced from S3, not inlined
    s3 = boto3.client("s3", region_name=region)
    code_key = "lambda/lambda_package.zip"
    s3.upload_file(str(zip_path), bucket, code_key)
    print(f"uploaded code to s3://{bucket}/{code_key} "
          f"({zip_path.stat().st_size / 1e6:.1f} MB)")
    code_ref = {"S3Bucket": bucket, "S3Key": code_key}
    try:
        lam.get_function(FunctionName=FUNCTION_NAME)
        lam.update_function_code(FunctionName=FUNCTION_NAME, **code_ref)
        waiter = lam.get_waiter("function_updated_v2")
        waiter.wait(FunctionName=FUNCTION_NAME)
        lam.update_function_configuration(FunctionName=FUNCTION_NAME, **config)
        print(f"updated function {FUNCTION_NAME}")
    except lam.exceptions.ResourceNotFoundException:
        lam.create_function(FunctionName=FUNCTION_NAME, Code=code_ref, **config)
        lam.get_waiter("function_active_v2").wait(FunctionName=FUNCTION_NAME)
        print(f"created function {FUNCTION_NAME} (512MB / 10s timeout)")

    print("smoke-invoke:")
    resp = lam.invoke(FunctionName=FUNCTION_NAME,
                      Payload=json.dumps({"body": json.dumps(
                          {"date": "2018-07-15", "store": 2, "item": 15})}))
    print(resp["Payload"].read().decode()[:300])


if __name__ == "__main__":
    main()
