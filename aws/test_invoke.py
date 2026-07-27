"""End-to-end test of the deployed API Gateway endpoint.

Run:  python -m aws.test_invoke https://xxxx.execute-api.region.amazonaws.com/forecast
"""

import sys

import requests


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: python -m aws.test_invoke <endpoint-url>")
    url = sys.argv[1]

    cases = [
        {"date": "2018-07-15", "store": 2, "item": 15},          # summer Sunday
        {"date": "2018-01-08", "store": 2, "item": 15},          # winter Monday
        {"start_date": "2018-07-01", "periods": 14, "store": 2, "item": 15},
    ]
    for payload in cases:
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        points = data["forecast"]
        print(f"{payload} -> {len(points)} point(s), "
              f"first: {points[0]}, model: {data['model']}")

    summer = requests.post(url, json=cases[0], timeout=30).json()["forecast"][0]
    winter = requests.post(url, json=cases[1], timeout=30).json()["forecast"][0]
    assert summer["predicted_sales"] > winter["predicted_sales"], \
        "seasonality sanity failed"
    print("OK — endpoint healthy, seasonality sanity passed "
          f"(summer {summer['predicted_sales']} > winter {winter['predicted_sales']})")


if __name__ == "__main__":
    main()
