"""Tear down the serverless inference stack (keeps the budget alert).

Note: this stack bills nothing while idle — teardown is optional by cost
logic. Redeploy any time with the 6 commands in DEPLOYMENT_LOG.md.

Run:  python -m aws.teardown <bucket-name> [region]
"""

import sys

import boto3

FUNCTION = "demand-forecast"
API_NAME = "demand-forecast-api"
ROLE = "LambdaForecastingRole"


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: python -m aws.teardown <bucket-name> [region]")
    bucket = sys.argv[1]
    region = sys.argv[2] if len(sys.argv) > 2 else "eu-central-1"

    apigw = boto3.client("apigatewayv2", region_name=region)
    lam = boto3.client("lambda", region_name=region)
    s3 = boto3.client("s3", region_name=region)
    iam = boto3.client("iam")

    for api in apigw.get_apis()["Items"]:
        if api["Name"] == API_NAME:
            apigw.delete_api(ApiId=api["ApiId"])
            print(f"deleted API {API_NAME} ({api['ApiId']})")

    try:
        lam.delete_function(FunctionName=FUNCTION)
        print(f"deleted lambda {FUNCTION}")
    except lam.exceptions.ResourceNotFoundException:
        print("lambda already gone")

    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket):
            objs = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            if objs:
                s3.delete_objects(Bucket=bucket, Delete={"Objects": objs})
        s3.delete_bucket(Bucket=bucket)
        print(f"emptied + deleted bucket {bucket}")
    except Exception as e:
        print(f"bucket: {e}")

    try:
        for p in iam.list_role_policies(RoleName=ROLE)["PolicyNames"]:
            iam.delete_role_policy(RoleName=ROLE, PolicyName=p)
        iam.delete_role(RoleName=ROLE)
        print(f"deleted role {ROLE}")
    except iam.exceptions.NoSuchEntityException:
        print("role already gone")


if __name__ == "__main__":
    main()
