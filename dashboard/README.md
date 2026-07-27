# Demand Planning Control Tower — Power BI Dashboard

## How to open

1. Double-click `DemandPlanning/DemandPlanning.pbip` (opens in Power BI Desktop).
2. On first open, click **Refresh now** in the yellow banner (or Home → Refresh).
   PBIP projects store the model definition as text; the data itself loads from
   the CSVs in `data/` on first refresh (~30-60 seconds — the history table has
   913k rows).
3. Save once after refreshing — Power BI caches the data locally after that.

## Pages & the story they tell

| Page | Business question |
|---|---|
| **Executive Overview** | How is demand trending and can we trust the forecast? 5-year trend, growth, the two seasonal rhythms (July peak, weekend uplift), and the headline SMAPE. |
| **Seasonality & Planning** | When to stock and staff? Store × month planning grid, weekday/weekend split by month, compounding YoY growth, item ranking. |
| **Forecast Explorer** | What will sell next quarter? Pick any store & item: 2017 history beside the 90-day forecast, order plan by month, forecast by store. |
| **Forecast Accuracy** | Why trust the plan? Held-out quarter actual-vs-forecast overlay, bias, error by store, candidate model comparison, SHAP drivers. |

Slicers on Store/Item filter **all** fact tables at once (history, future forecast,
and test window) via the star-schema relationships.

## Data

All data in `data/*.csv` is **model-generated**: the future forecast contains
45,000 predictions (90 days × 500 store-item series) from the trained XGBoost
pipeline, and the test-forecast table holds per-row errors for the held-out
quarter (SMAPE 12.33).

The semantic model loads the CSVs by absolute path. If you move the project,
update the paths: open Transform Data in Power BI Desktop, or edit the
`File.Contents("...")` paths in `DemandPlanning.SemanticModel/model.bim`.
