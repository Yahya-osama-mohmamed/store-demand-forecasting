"""GUARDRAIL #1 — the $0.01 budget alert. Run this BEFORE anything else.

Creates (or verifies) an AWS Budget that emails you the moment estimated
charges exceed one cent. Every other script in this folder refuses to run
until this budget exists — cost safety is enforced, not suggested.

Run:  python -m aws.create_budget_guard your@email.com
"""

import sys

import boto3

BUDGET_NAME = "zero-spend-guard"


def budget_exists() -> bool:
    """True / False, or raises PermissionError when the IAM user cannot view
    budgets at all (root hasn't enabled 'IAM access to Billing')."""
    import botocore.exceptions
    sts = boto3.client("sts")
    account = sts.get_caller_identity()["Account"]
    budgets = boto3.client("budgets")
    try:
        # Any cost budget counts — console templates use their own names
        found = budgets.describe_budgets(AccountId=account).get("Budgets", [])
        return len(found) > 0
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "AccessDeniedException":
            raise PermissionError("cannot view budgets from this IAM user")
        raise


def require_budget() -> None:
    """Imported by every deploy script — hard-stops if the guard is missing.

    If the IAM user cannot view budgets (a root-level billing setting), the
    deploy may proceed ONLY with an explicit manual confirmation that the
    budget was created in the console: set BUDGET_CONFIRMED=1.
    """
    import os
    try:
        if budget_exists():
            return
        sys.exit(
            f"REFUSING TO DEPLOY: budget '{BUDGET_NAME}' not found.\n"
            f"Run first:  python -m aws.create_budget_guard your@email.com"
        )
    except PermissionError:
        if os.environ.get("BUDGET_CONFIRMED") == "1":
            print("budget guard: cannot view budgets from this IAM user — "
                  "proceeding on BUDGET_CONFIRMED=1 (created via console)")
            return
        sys.exit(
            "REFUSING TO DEPLOY: this IAM user cannot view budgets, so the "
            "$0 guard cannot be verified.\n"
            "Create a 'Zero spend budget' in the Billing Console (root), "
            "then re-run with BUDGET_CONFIRMED=1 set."
        )


def main() -> None:
    if len(sys.argv) < 2 or "@" not in sys.argv[1]:
        sys.exit("usage: python -m aws.create_budget_guard your@email.com")
    email = sys.argv[1]

    sts = boto3.client("sts")
    account = sts.get_caller_identity()["Account"]
    budgets = boto3.client("budgets")

    if budget_exists():
        print(f"budget '{BUDGET_NAME}' already exists — guard is active")
        return

    budgets.create_budget(
        AccountId=account,
        Budget={
            "BudgetName": BUDGET_NAME,
            "BudgetLimit": {"Amount": "0.01", "Unit": "USD"},
            "TimeUnit": "MONTHLY",
            "BudgetType": "COST",
        },
        NotificationsWithSubscribers=[{
            "Notification": {
                "NotificationType": "ACTUAL",
                "ComparisonOperator": "GREATER_THAN",
                "Threshold": 100.0,  # 100% of $0.01
                "ThresholdType": "PERCENTAGE",
            },
            "Subscribers": [{"SubscriptionType": "EMAIL", "Address": email}],
        }],
    )
    print(f"created budget '{BUDGET_NAME}' ($0.01/month) with alert to {email}")


if __name__ == "__main__":
    main()
