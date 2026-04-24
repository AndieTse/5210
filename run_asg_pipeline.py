from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from src.evaluation import (
    add_forward_return,
    evaluate_factors,
    factor_correlation,
    run_dynamic_mega_alpha,
    sharpe_ratio,
    select_low_corr_factors,
)
from src.factor_library import add_base_factors


LOOKBACK_YEARS = 3
AUTO_TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "JPM",
    "BAC",
    "XOM",
    "CVX",
    "WMT",
    "PG",
    "JNJ",
    "KO",
    "PEP",
    "INTC",
    "CSCO",
    "IBM",
    "DIS",
    "MCD",
]


def _get_recent_window(years: int = LOOKBACK_YEARS) -> tuple[pd.Timestamp, pd.Timestamp]:
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.DateOffset(years=years)
    return start_date, end_date


def _slice_recent_years(df: pd.DataFrame, years: int = LOOKBACK_YEARS) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    max_date = out["date"].max()
    if pd.isna(max_date):
        return out
    start_date = max_date - pd.DateOffset(years=years)
    return out[out["date"] >= start_date].copy()


def download_sample_data(data_path: Path) -> pd.DataFrame:
    start_date, end_date = _get_recent_window(LOOKBACK_YEARS)
    raw = yf.download(
        tickers=AUTO_TICKERS,
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("Failed to download data from yfinance.")

    records = []
    for tk in AUTO_TICKERS:
        if tk not in raw.columns.get_level_values(0):
            continue
        tdf = raw[tk].copy()
        if "Close" not in tdf.columns or "Volume" not in tdf.columns:
            continue
        tdf = tdf.reset_index().rename(columns={"Date": "date", "Close": "close", "Volume": "volume"})
        tdf["asset"] = tk
        tdf = tdf[["date", "asset", "close", "volume"]]
        records.append(tdf)

    if not records:
        raise RuntimeError("No valid ticker data found in yfinance output.")
    out = pd.concat(records, ignore_index=True).sort_values(["asset", "date"]).reset_index(drop=True)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(data_path, index=False)
    return out


def build_static_returns(factors_df: pd.DataFrame, selected_factors: list[str]) -> pd.Series:
    if not selected_factors:
        return pd.Series(dtype=float)
    static_daily = factors_df[["date", "fwd_ret_1d"] + selected_factors].dropna().copy()
    if static_daily.empty:
        return pd.Series(dtype=float)
    static_daily["signal"] = static_daily[selected_factors].mean(axis=1)
    static_daily["signal"] = static_daily.groupby("date")["signal"].transform(
        lambda x: x / x.abs().sum() if x.abs().sum() > 0 else x * 0.0
    )
    out = (
        static_daily.assign(weighted_ret=static_daily["signal"] * static_daily["fwd_ret_1d"])
        .groupby("date")["weighted_ret"]
        .sum()
    )
    out.name = "static_ret"
    return out


def calc_backtest_metrics(returns: pd.Series, strategy_name: str) -> dict:
    r = returns.dropna()
    if r.empty:
        return {
            "strategy": strategy_name,
            "days": 0,
            "avg_daily_ret": np.nan,
            "annual_return": np.nan,
            "annual_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "win_rate": np.nan,
        }
    equity = (1.0 + r).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    annual_return = float(equity.iloc[-1] ** (252 / len(r)) - 1.0)
    annual_vol = float(r.std() * np.sqrt(252))
    return {
        "strategy": strategy_name,
        "days": int(len(r)),
        "avg_daily_ret": float(r.mean()),
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe_ratio(r),
        "max_drawdown": float(drawdown.min()),
        "win_rate": float((r > 0).mean()),
    }


def run_dynamic_sensitivity(factors_df: pd.DataFrame) -> pd.DataFrame:
    grid = [
        {"max_factors": 5, "metric_window": 40, "ridge_lambda": 0.5},
        {"max_factors": 5, "metric_window": 60, "ridge_lambda": 1.0},
        {"max_factors": 8, "metric_window": 40, "ridge_lambda": 1.0},
        {"max_factors": 8, "metric_window": 60, "ridge_lambda": 1.0},
        {"max_factors": 10, "metric_window": 60, "ridge_lambda": 1.0},
        {"max_factors": 12, "metric_window": 80, "ridge_lambda": 2.0},
    ]
    records = []
    for cfg in grid:
        ret_df, _, _ = run_dynamic_mega_alpha(
            factors_df,
            max_factors=cfg["max_factors"],
            metric_window=cfg["metric_window"],
            min_ic=0.0,
            min_icir=0.0,
            ridge_lambda=cfg["ridge_lambda"],
        )
        r = ret_df["dynamic_ret"].dropna()
        metrics = calc_backtest_metrics(r, strategy_name="dynamic")
        records.append(
            {
                "max_factors": cfg["max_factors"],
                "metric_window": cfg["metric_window"],
                "ridge_lambda": cfg["ridge_lambda"],
                "sharpe": metrics["sharpe"],
                "annual_return": metrics["annual_return"],
                "max_drawdown": metrics["max_drawdown"],
                "valid_days": metrics["days"],
                "avg_factor_count": float(ret_df["factor_count"].mean()),
            }
        )
    out = pd.DataFrame(records).sort_values("sharpe", ascending=False).reset_index(drop=True)
    return out


def main() -> None:
    root = Path(__file__).resolve().parent
    data_path = root / "data" / "market_data.csv"
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    downloaded = False
    if data_path.exists():
        raw = pd.read_csv(data_path)
    else:
        raw = download_sample_data(data_path)
        downloaded = True
    raw = _slice_recent_years(raw, years=LOOKBACK_YEARS)

    factors_df = add_base_factors(raw)
    factors_df = add_forward_return(factors_df, horizon=1)

    sharpe_df = evaluate_factors(factors_df)
    corr_df = factor_correlation(factors_df)
    selected, realized_max_corr = select_low_corr_factors(sharpe_df, corr_df, max_abs_corr=0.5)

    avg_sharpe_all = float(sharpe_df["sharpe"].mean()) if not sharpe_df.empty else float("nan")

    dynamic_ret_df, dynamic_weights_df, dynamic_panel_df = run_dynamic_mega_alpha(
        factors_df,
        max_factors=10,
        metric_window=60,
        min_ic=0.0,
        min_icir=0.0,
        ridge_lambda=1.0,
    )
    dynamic_sharpe = sharpe_ratio(dynamic_ret_df["dynamic_ret"])
    static_daily = build_static_returns(factors_df, selected_factors=selected)
    static_sharpe = sharpe_ratio(static_daily)
    dynamic_daily = dynamic_ret_df["dynamic_ret"].dropna()

    backtest_metrics = pd.DataFrame(
        [
            calc_backtest_metrics(static_daily, strategy_name="static_equal_weight"),
            calc_backtest_metrics(dynamic_daily, strategy_name="dynamic_alphaforge_lite"),
        ]
    )
    sensitivity_df = run_dynamic_sensitivity(factors_df)
    best_cfg = sensitivity_df.iloc[0].to_dict() if not sensitivity_df.empty else {}

    factors_df.to_csv(reports_dir / "factor_values.csv", index=False)
    sharpe_df.to_csv(reports_dir / "factor_sharpe.csv", index=False)
    corr_df.to_csv(reports_dir / "factor_corr.csv", index=True)
    dynamic_ret_df.to_csv(reports_dir / "dynamic_returns.csv", index=False)
    dynamic_weights_df.to_csv(reports_dir / "dynamic_weights.csv", index=False)
    dynamic_panel_df.to_csv(reports_dir / "dynamic_ic_icir_panel.csv", index=False)
    backtest_metrics.to_csv(reports_dir / "backtest_metrics.csv", index=False)
    sensitivity_df.to_csv(reports_dir / "dynamic_sensitivity.csv", index=False)
    pd.Series(selected, name="selected_factors").to_csv(
        reports_dir / "selected_factors.txt", index=False, header=True
    )

    summary = [
        "# Assignment Summary",
        "",
        f"- Number of factors evaluated: {len(sharpe_df)}",
        f"- Average Sharpe (all factors, no cost): {avg_sharpe_all:.6f}",
        f"- Number of selected factors under corr<=0.5: {len(selected)}",
        f"- Realized max abs corr (selected set): {realized_max_corr}",
        f"- Static Mega-Alpha Sharpe (equal-weight selected factors): {static_sharpe:.6f}",
        f"- Dynamic Mega-Alpha Sharpe (rolling IC/ICIR + ridge weights): {dynamic_sharpe:.6f}",
        "",
        "## Data",
        f"- Source file: {data_path}",
        f"- Data window used: recent {LOOKBACK_YEARS} years",
        f"- Data source: {'Yahoo Finance via yfinance (auto download)' if downloaded else 'Local CSV provided by user'}",
        f"- Access date: {pd.Timestamp.today().strftime('%Y-%m-%d')}",
        "- Auto downloader fields: adjusted close and volume (daily).",
        "- Auto downloader universe: " + ", ".join(AUTO_TICKERS),
        "",
        "## Backtest Necessity",
        "- Factor-level IC/Sharpe validates signal quality, but portfolio-level backtest validates tradeability.",
        "- Added metrics include annual return, annual volatility, max drawdown and win rate.",
        "",
        "## Thinking and Optimization",
        "- Added dynamic parameter sensitivity over TopN/window/ridge_lambda.",
        (
            f"- Best dynamic config by Sharpe: TopN={best_cfg.get('max_factors')}, "
            f"window={best_cfg.get('metric_window')}, lambda={best_cfg.get('ridge_lambda')}, "
            f"Sharpe={best_cfg.get('sharpe')}"
            if best_cfg
            else "- Best dynamic config by Sharpe: not available"
        ),
        "",
        "## AlphaForge-Lite Design",
        "- Stage 1: create a factor zoo and enforce low pairwise correlation.",
        "- Stage 2: daily dynamic selection by rolling IC/ICIR and linear weighting.",
    ]
    (reports_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")

    reflection = [
        "# Reflection and Optimization Notes",
        "",
        "## Why Backtest Is Required",
        "- IC/ICIR evaluates cross-sectional predictability, but does not represent executable portfolio performance.",
        "- Portfolio backtest captures compounding, volatility clustering and drawdown path.",
        "- Therefore both factor metrics and portfolio metrics are required for a complete evaluation.",
        "",
        "## What Was Optimized",
        "- Kept strict correlation threshold (max abs corr <= 0.5) to avoid redundant factors.",
        "- Added dynamic weighting with lagged selection to reduce look-ahead bias.",
        "- Ran parameter sensitivity on TopN, rolling window, and ridge regularization.",
        "",
        "## Remaining Limitations",
        "- Transaction cost and slippage are not included.",
        "- Universe is a compact sample set; robustness should be rechecked on broader universes.",
        "- Walk-forward yearly retraining can be added for stronger out-of-sample validation.",
    ]
    (reports_dir / "project_reflection.md").write_text("\n".join(reflection), encoding="utf-8")


if __name__ == "__main__":
    main()



