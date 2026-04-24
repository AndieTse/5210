# Reflection and Optimization Notes

## Why Backtest Is Required
- IC/ICIR evaluates cross-sectional predictability, but does not represent executable portfolio performance.
- Portfolio backtest captures compounding, volatility clustering and drawdown path.
- Therefore both factor metrics and portfolio metrics are required for a complete evaluation.

## What Was Optimized
- Kept strict correlation threshold (max abs corr <= 0.5) to avoid redundant factors.
- Added dynamic weighting with lagged selection to reduce look-ahead bias.
- Ran parameter sensitivity on TopN, rolling window, and ridge regularization.

## Remaining Limitations
- Transaction cost and slippage are not included.
- Universe is a compact sample set; robustness should be rechecked on broader universes.
- Walk-forward yearly retraining can be added for stronger out-of-sample validation.