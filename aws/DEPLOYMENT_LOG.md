# Deployment Log — Serverless Inference on AWS (executed 2026-07-27, eu-central-1)

Complete record of the live deployment and teardown. Evidence:
[`screenshots/aws_deployment_evidence.png`](screenshots/aws_deployment_evidence.png)
+ [`screenshots/deployment_evidence.json`](screenshots/deployment_evidence.json)
(captured from the live resources via AWS APIs, including real endpoint calls).

## Guardrails honored

- ✅ Zero-spend budget verified via the Budgets API before deployment
- ✅ Lambda over SageMaker (no hourly-billing endpoint can exist by design)
- ✅ Private S3 bucket (all public access blocked)
- ✅ Least-privilege IAM: `s3:GetObject` on `models/*` + own logs — nothing else

## Resources created (account 451071929298)

| Resource | Identifier | Notes |
|---|---|---|
| S3 bucket | `demand-forecast-models-451071929298` | model artifacts (`models/`) + code zip (`lambda/`), private |
| IAM role | `LambdaForecastingRole` | inline policy `forecast-s3-logs` |
| Lambda | `demand-forecast` | python3.11, 512MB, 10s, 52.6MB package (numpy + xgboost-cpu) |
| API Gateway | `demand-forecast-api` (`ayckev03m9`) | HTTP API, `POST /forecast`, public |

## The engineering story

The trained artifact is a sklearn Pipeline (custom FeatureEngineer +
ColumnTransformer + XGBoost) whose pickle needs sklearn+pandas+joblib —
over Lambda's 250MB unzipped limit. Solution: `export_for_lambda.py`
decomposed the pipeline into an XGBoost booster JSON (2.2MB) + learned
statistics (0.14MB), served by a numpy+xgboost-cpu-only handler.
**`parity_check.py` proved equivalence before deploy: max abs diff 0.000007
across 500 random samples spanning 2013–2020.**

Packaging detail: xgboost pulls scipy, pushing the zip to 58.7MB — pruned
bundled test suites (−64MB unzipped) and deployed via S3 code reference
(direct upload caps at 50MB zipped). Final: 52.6MB zipped / 175.7MB unzipped.

## What was verified live (public endpoint)

`POST https://ayckev03m9.execute-api.eu-central-1.amazonaws.com/forecast`

| Request | Response |
|---|---|
| 2018-07-15, store 2, item 15 (summer Sunday) | **185.59 units** |
| 2018-01-08, store 2, item 15 (winter Monday) | **67.47 units** |
| 14-day horizon from 2018-07-01 | 14 points, model=XGBoost, expected SMAPE 12.33 |

The AWS numbers match the local FastAPI/pipeline **exactly** — guaranteed by
the pre-deploy parity check. Seasonality sanity assert (summer > winter)
passed. Warm-cache confirmed: cold start ~2s (S3 fetch + parse), warm
invocations ~10ms without touching S3.

## Teardown record (same session, at user request)

Deleted: API Gateway `demand-forecast-api` → Lambda `demand-forecast` →
S3 bucket (emptied) → IAM role policy + role. Budget alert kept. Note:
this stack costs $0 while idle — teardown was by explicit request, not
cost necessity.

## Redeploy from scratch (~5 min)

```bash
python -m aws.export_for_lambda && python -m aws.parity_check
python -m aws.upload_model  <bucket>
bash aws/build_lambda_package.sh
python -m aws.deploy_lambda <bucket>
python -m aws.create_api                 # prints the new public URL
python -m aws.test_invoke   <url>
```
