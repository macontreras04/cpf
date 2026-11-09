# Savings Deposits with XGBoost: Core / Non-Core and Forecasting

Quantile XGBoost models for the behaviour of non-maturity savings deposits (ALM / IRRBB / liquidity), applied to ten years of daily balances of two savings products.

- **Part 1, core vs non-core** (`core_noncore_model.py`): estimates the share of today's balance that stays even at the worst point of the next 12 months, with 99% and 95% confidence.
- **Part 2, forecasting** (`forecast_part2.py`): 1- to 12-month quantile forecasts of the balance, backtested against a simple "no change" forecast.

Author: Miguel Contreras, CQF · Market Risk

## Data

`data/savings_deposits_indexed.csv` has 2,677 daily rows from 2016-01-04 to 2026-09-30:

| Column | Description |
|---|---|
| `date` | Business day (ISO format) |
| `savings_at_sight_index` | Savings at sight balance, indexed to 100 on 2016-01-04 |
| `savings_other_index` | Savings other balance, indexed to 100 on 2016-01-04 |

The data comes from a real bank. The institution, currency and actual balances are withheld: each series is divided by its own first value and multiplied by 100. Both models work on ratios and log changes, so indexing does not change their results, apart from tiny rounding differences.

The file is provided as received, including 3 duplicated dates and a few one-day spikes. The cleaning step in `core_noncore_model.py` handles them and records every change in the `Cleaning` sheet of the output.

## Part 1: core vs non-core

For each date *t* and horizon *h* (1 day to 12 months), the target is the worst-point retention ratio:

```
y_h(t) = min(B[t+1], ..., B[t+h]) / B[t]
Core_t = B_t · q̂_α(y_12m | X_t)      α = 1% → 99% confidence
```

- **Model:** one quantile XGBoost per horizon (`reg:quantileerror`), predicting the 1%, 5%, 10% and 50% quantiles together.
- **Features:** rolling and EWMA volatility, drawdowns, worst recent falls and seasonality, all observable at *t*.
- **Overfitting control:**
  - depth-2 trees and leaves of at least one full horizon (252 days for 12 months)
  - early stopping on a purged validation block
  - monotone constraints with an economic sign, for example more volatility can never raise the core
  - a reduced feature set
- **Tail calibration:** one-sided conformal adjustment from out-of-sample errors. It can only make the core more conservative, never less.
- **Validation:** annual walk-forward with purging, 2019–2025, against historical simulation.

Results at 30 Sep 2026:

| | Core 99% | Core 95% |
|---|---|---|
| Savings at sight | 57.6% | 64.0% |
| Savings other | 33.4% | 55.9% |

## Part 2: forecasting

For each horizon *h* = 1 to 12 months, one quantile XGBoost predicts the 5%, 50% and 95% quantiles of `log(B[t+h] / B[t])`. It is backtested year by year from 2019 to 2025 against a "no change" forecast, a historical drift and historical quantile bands.

Main findings:
- **Median forecast:** beats "no change" in only 4 of 24 horizon tests. Daily deposit balances behave close to a random walk.
- **Savings other:** the 5–95% band is 2–32% better than the historical band (pinball loss), with 94–99% coverage.
- **Savings at sight:** the band is too narrow, with 80–93% coverage against a 90% target.

## How to run

```bash
pip install -r requirements.txt

# Part 1: Excel with summary, liquidity buckets, daily core series, backtest and charts
python core_noncore_model.py --input data/savings_deposits_indexed.csv --out results

# Part 2: backtest tables and 12-month forecasts
python forecast_part2.py --input data/savings_deposits_indexed.csv
```

Part 1 takes about one minute. `forecast_part2.py` imports `core_noncore_model.py`, so keep both in the same folder.

## Limitations

- About ten years of overlapping daily data contain only about nine independent 12-month windows, so the 1% tail rests on few stress episodes.
- Savings at sight still shows more 12-month exceptions at 99% than the 1% target in the backtest.
- Basel IRRBB caps on core deposits are not applied, because they depend on how each product is classified.
