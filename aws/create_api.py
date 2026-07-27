"""Expose the Lambda via an API Gateway HTTP API (simpler + cheaper than REST).

Run:  python -m aws.create_api [region]
Prints the public invoke URL for POST /forecast.
"""

import sys

import boto3

from aws.create_budget_guard import require_budget

FUNCTION_NAME = "demand-forecast"
API_NAME = "demand-forecast-api"


def main() -> None:
    require_budget()
    region = sys.argv[1] if len(sys.argv) > 1 else "eu-central-1"
    lam = boto3.client("lambda", region_name=region)
    apigw = boto3.client("apigatewayv2", region_name=region)
    sts = boto3.client("sts")
    account = sts.get_caller_identity()["Account"]

    fn_arn = lam.get_function(FunctionName=FUNCTION_NAME)["Configuration"]["FunctionArn"]

    existing = [a for a in apigw.get_apis()["Items"] if a["Name"] == API_NAME]
    if existing:
        api = existing[0]
        print(f"API {API_NAME} exists")
    else:
        api = apigw.create_api(
            Name=API_NAME, ProtocolType="HTTP",
            Target=fn_arn, RouteKey="POST /forecast",
        )
        print(f"created HTTP API {API_NAME}")

    # Allow API Gateway to invoke the function
    try:
        lam.add_permission(
            FunctionName=FUNCTION_NAME,
            StatementId="apigw-invoke",
            Action="lambda:InvokeFunction",
            Principal="apigateway.amazonaws.com",
            SourceArn=f"arn:aws:execute-api:{region}:{account}:{api['ApiId']}/*",
        )
    except lam.exceptions.ResourceConflictException:
        pass  # permission already granted

    print(f"\nendpoint: {api['ApiEndpoint']}/forecast")
    print('test:  curl -X POST <endpoint> -d \'{"start_date":"2018-07-01","periods":14,"store":2,"item":15}\'')


if __name__ == "__main__":
    main()
