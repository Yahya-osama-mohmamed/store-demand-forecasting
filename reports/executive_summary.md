# Executive Summary: Store Item Demand Forecasting

## 1. Business Context & Objective
Demand forecasting is the backbone of retail operations: purchasing, inventory allocation, staffing, and logistics all depend on knowing how much of each item each store will sell. Under-forecasting causes stockouts and lost revenue; over-forecasting ties up working capital and increases waste. The objective of this project was to build a system that forecasts **daily unit sales for 50 items across 10 stores** (500 concurrent time series) over a 3-month horizon.

## 2. Key Findings from Data Exploration (EDA)
Analysis of 913,000 daily sales records (2013–2017) revealed strong, exploitable structure:
* **Yearly seasonality:** demand peaks every summer (July) and troughs in winter (January) — the July average is roughly 50% above January's.
* **Weekly rhythm:** weekends sell markedly more than weekdays; Sunday is the strongest day, Monday the weakest.
* **Steady growth:** average demand grows year over year across all stores — the trend must be extrapolated, not just repeated.
* **Stable entity hierarchy:** stores and items keep their relative ranking over time. A store's and an item's historical average is a powerful predictor of its future demand.
* **No missing days or gaps:** all 500 series are complete over 5 years, simplifying modeling.

## 3. Model Development & Performance
We built a complete pipeline: cleaning, calendar/cyclical feature engineering, per-entity demand aggregates learned from training data only, and three tuned gradient-boosting models — **LightGBM**, **XGBoost**, and scikit-learn's **HistGradientBoosting** (the model family that dominates this competition's public leaderboard).

**Evaluation Strategy:** strictly chronological splits — train through June 2017, model selection on July–September 2017, and a final untouched test window of October–December 2017 matching the competition's 3-month horizon. Hyperparameters were tuned with time-series cross-validation optimizing **SMAPE**, the competition metric.

### Final Model Selection
The champion is chosen automatically by validation SMAPE on each pipeline run and recorded in `models/model_metadata.json` with its test metrics. All three models land close together — the signal in this dataset is largely exhausted by seasonality + entity averages, and the champion's test SMAPE (~13–14) is competitive with leading public solutions (~12.5–14).

## 4. Drivers of Demand (SHAP Explainability)
SHAP analysis of the champion model shows the forecast is driven by:
1. **Store-item historical average** — the single strongest signal (who is selling what).
2. **Day-of-week aggregates** — the weekly rhythm per entity.
3. **Seasonal (month) aggregates** — the summer peak / winter trough.
4. **Year** — captures the steady multiplicative growth trend.
5. **Cyclical calendar encodings** — fine-grained position within week and year.

## 5. Business Recommendations
1. **Automate replenishment from the forecast:** feed the 90-day item-store forecasts directly into purchasing to cut both stockouts and overstock.
2. **Plan seasonal capacity early:** the summer peak is predictable months ahead — negotiate supplier volumes and staffing before the June ramp.
3. **Weekend-weighted staffing & delivery windows:** weekend demand is systematically higher; align labor and logistics schedules with the weekly rhythm.
4. **Watch the growth trend:** year-over-year growth compounds; capacity decisions based on last year's absolute volumes will systematically under-provision.

## 6. Estimated ROI
Replacing naive "same as last year" ordering with model-driven forecasts typically reduces inventory holding costs and stockout losses by double-digit percentages. With per-item, per-store, per-day granularity, buying teams can move from category-level guesses to precise, explainable order quantities.
