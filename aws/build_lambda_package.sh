#!/usr/bin/env bash
# Build the Lambda deployment ZIP: handler + numpy + xgboost-cpu only.
# The full sklearn/pandas stack would blow Lambda's 250MB unzipped limit —
# see export_for_lambda.py for how we avoid needing it at all.
#
# Run from the project root:  bash aws/build_lambda_package.sh
set -euo pipefail
cd "$(dirname "$0")"

rm -rf package lambda_package.zip
mkdir -p package

# manylinux wheels so the binaries match the Lambda (Amazon Linux) runtime
pip install \
  --platform manylinux2014_x86_64 \
  --implementation cp --python-version 3.11 --only-binary=:all: \
  --target package \
  -r requirements-lambda.txt

cp lambda_function.py package/
cd package && zip -qr ../lambda_package.zip . && cd ..

echo "lambda_package.zip: $(du -m lambda_package.zip | cut -f1) MB (zipped)"
echo "unzipped size     : $(du -sm package | cut -f1) MB (limit: 250 MB)"
