from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def get_factor_columns(df: pd.DataFrame) -> List[str]:
    base = {"date", "asset", "close", "volume", "ret_1d", "fwd_ret_1d"}
    return [c for c in df.columns if c not in base]


def add_forward_return(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    out = df.copy()
    out["fwd_ret_1d"] = (
        out.groupby("asset", group_keys=False)["close"].shift(-horizon) / out["close"] - 1.0
    )
    return out


def factor_daily_pnl(df: pd.DataFrame, factor_col: str) -> pd.Series:
    tmp = df[["date", factor_col, "fwd_ret_1d"]].dropna()
    if tmp.empty:
        return pd.Series(dtype=float)

    tmp["signal"] = tmp.groupby("date")[factor_col].transform(
        lambda x: x / np.abs(x).sum() if np.abs(x).sum() > 0 else x * 0.0
    )
    pnl = (tmp["signal"] * tmp["fwd_ret_1d"]).groupby(tmp["date"]).sum()
    pnl.name = factor_col
    return pnl


def sharpe_ratio(returns: pd.Series, annualization: int = 252) -> float:
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    std = returns.std()
    if std == 0 or pd.isna(std):
        return np.nan
    return float((returns.mean() / std) * np.sqrt(annualization))


def evaluate_factors(df: pd.DataFrame) -> pd.DataFrame:
    factor_cols = get_factor_columns(df)
    records = []
    for col in factor_cols:
        pnl = factor_daily_pnl(df, col)
        daily_ic = calc_daily_ic(df, col, method="pearson")
        daily_rank_ic = calc_daily_ic(df, col, method="spearman")
        records.append(
            {
                "factor": col,
                "sharpe": sharpe_ratio(pnl),
                "ic_mean": daily_ic.mean(),
                "icir": rolling_mean_div_std(daily_ic),
                "rank_ic_mean": daily_rank_ic.mean(),
                "rank_icir": rolling_mean_div_std(daily_rank_ic),
            }
        )
    out = pd.DataFrame(records).sort_values("sharpe", ascending=False).reset_index(drop=True)
    return out


def factor_correlation(df: pd.DataFrame) -> pd.DataFrame:
    factor_cols = get_factor_columns(df)
    matrix_input = df[factor_cols].copy()
    return matrix_input.corr(method="pearson")


def select_low_corr_factors(
    sharpe_df: pd.DataFrame, corr_df: pd.DataFrame, max_abs_corr: float = 0.5
) -> Tuple[List[str], float]:
    selected: List[str] = []
    for factor in sharpe_df["factor"].tolist():
        if factor not in corr_df.columns:
            continue
        if not selected:
            selected.append(factor)
            continue
        pair_corr = corr_df.loc[factor, selected].abs().replace([np.inf, -np.inf], np.nan)
        max_corr = pair_corr.max(skipna=True)
        if pd.isna(max_corr) or max_corr <= max_abs_corr:
            selected.append(factor)

    realized = np.nan
    if len(selected) > 1:
        sub = corr_df.loc[selected, selected]
        mask = ~np.eye(len(selected), dtype=bool)
        vals = sub.where(mask).abs().stack().dropna().values
        if len(vals) > 0:
            realized = float(np.max(vals))
        else:
            realized = 0.0
    elif len(selected) == 1:
        realized = 0.0
    return selected, realized


def rolling_mean_div_std(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return np.nan
    std = clean.std()
    if std == 0 or pd.isna(std):
        return np.nan
    return float(clean.mean() / std)


def calc_daily_ic(df: pd.DataFrame, factor_col: str, method: str = "pearson") -> pd.Series:
    tmp = df[["date", factor_col, "fwd_ret_1d"]].dropna()
    if tmp.empty:
        return pd.Series(dtype=float, name=factor_col)
    if method == "spearman":
        out = tmp.groupby("date").apply(
            lambda x: x[factor_col].rank().corr(x["fwd_ret_1d"].rank(), method="pearson")
        )
        out.name = factor_col
        return out
    out = tmp.groupby("date").apply(
        lambda x: x[factor_col].corr(x["fwd_ret_1d"], method=method)
    )
    out.name = factor_col
    return out


def compute_rolling_factor_metrics(
    df: pd.DataFrame, factor_cols: List[str], window: int = 60
) -> Dict[str, pd.DataFrame]:
    ic_data = {}
    rank_ic_data = {}
    for factor in factor_cols:
        ic_data[factor] = calc_daily_ic(df, factor, method="pearson")
        rank_ic_data[factor] = calc_daily_ic(df, factor, method="spearman")
    ic_df = pd.DataFrame(ic_data).sort_index()
    rank_ic_df = pd.DataFrame(rank_ic_data).sort_index()

    rolling_ic = ic_df.rolling(window=window, min_periods=max(20, window // 3)).mean()
    rolling_icir = rolling_ic / ic_df.rolling(window=window, min_periods=max(20, window // 3)).std()
    rolling_rank_ic = rank_ic_df.rolling(window=window, min_periods=max(20, window // 3)).mean()
    rolling_rank_icir = rolling_rank_ic / rank_ic_df.rolling(
        window=window, min_periods=max(20, window // 3)
    ).std()
    return {
        "ic": ic_df,
        "rank_ic": rank_ic_df,
        "rolling_ic": rolling_ic,
        "rolling_icir": rolling_icir,
        "rolling_rank_ic": rolling_rank_ic,
        "rolling_rank_icir": rolling_rank_icir,
    }


def dynamic_select_factors_for_day(
    date: pd.Timestamp,
    rolling_ic: pd.DataFrame,
    rolling_icir: pd.DataFrame,
    max_factors: int = 10,
    min_ic: float = 0.0,
    min_icir: float = 0.0,
) -> List[str]:
    if date not in rolling_ic.index or date not in rolling_icir.index:
        return []
    ic_today = rolling_ic.loc[date].dropna()
    icir_today = rolling_icir.loc[date].dropna()
    aligned = pd.concat([ic_today.rename("ic"), icir_today.rename("icir")], axis=1).dropna()
    filtered = aligned[(aligned["ic"] > min_ic) & (aligned["icir"] > min_icir)]
    if filtered.empty:
        return []
    return filtered.sort_values("ic", ascending=False).head(max_factors).index.tolist()


def fit_linear_weights(
    train_slice: pd.DataFrame,
    selected_factors: List[str],
    ridge_lambda: float = 1.0,
) -> pd.Series:
    tmp = train_slice[selected_factors + ["fwd_ret_1d"]].dropna()
    if tmp.empty or len(selected_factors) == 0:
        return pd.Series(dtype=float)
    x = tmp[selected_factors].values
    y = tmp["fwd_ret_1d"].values.reshape(-1, 1)
    xtx = x.T @ x
    reg = ridge_lambda * np.eye(xtx.shape[0])
    try:
        w = np.linalg.solve(xtx + reg, x.T @ y).flatten()
    except np.linalg.LinAlgError:
        return pd.Series(dtype=float)
    return pd.Series(w, index=selected_factors)


def run_dynamic_mega_alpha(
    df: pd.DataFrame,
    max_factors: int = 10,
    metric_window: int = 60,
    train_window: int = 120,
    min_ic: float = 0.0,
    min_icir: float = 0.0,
    ridge_lambda: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    factor_cols = get_factor_columns(df)
    metrics = compute_rolling_factor_metrics(df, factor_cols=factor_cols, window=metric_window)
    rolling_ic = metrics["rolling_ic"]
    rolling_icir = metrics["rolling_icir"]

    rets_records = []
    weight_records = []
    all_dates = sorted(df["date"].dropna().unique())

    for i, dt in enumerate(all_dates):
        date = pd.Timestamp(dt)
        if i == 0:
            rets_records.append({"date": date, "dynamic_ret": np.nan, "factor_count": 0})
            continue
        score_date = pd.Timestamp(all_dates[i - 1])
        selected = dynamic_select_factors_for_day(
            date=score_date,
            rolling_ic=rolling_ic,
            rolling_icir=rolling_icir,
            max_factors=max_factors,
            min_ic=min_ic,
            min_icir=min_icir,
        )
        day_slice = df[df["date"] == date]
        train_dates = all_dates[max(0, i - train_window) : i]
        train_slice = df[df["date"].isin(train_dates)]
        if not selected:
            rets_records.append({"date": date, "dynamic_ret": np.nan, "factor_count": 0})
            continue
        weights = fit_linear_weights(
            train_slice=train_slice, selected_factors=selected, ridge_lambda=ridge_lambda
        )
        if weights.empty:
            rets_records.append({"date": date, "dynamic_ret": np.nan, "factor_count": 0})
            continue

        signal = day_slice[selected].mul(weights, axis=1).sum(axis=1)
        denom = np.abs(signal).sum()
        if denom == 0 or pd.isna(denom):
            rets_records.append({"date": date, "dynamic_ret": np.nan, "factor_count": len(selected)})
            continue
        signal = signal / denom
        daily_ret = float((signal * day_slice["fwd_ret_1d"]).sum())
        rets_records.append({"date": date, "dynamic_ret": daily_ret, "factor_count": len(selected)})

        for fac, w in weights.items():
            weight_records.append({"date": date, "factor": fac, "weight": float(w)})

    returns_df = pd.DataFrame(rets_records).sort_values("date").reset_index(drop=True)
    weights_df = pd.DataFrame(weight_records).sort_values(["date", "factor"]).reset_index(drop=True)
    panel_df = pd.concat(
        [
            metrics["rolling_ic"].stack().rename("rolling_ic"),
            metrics["rolling_icir"].stack().rename("rolling_icir"),
            metrics["rolling_rank_ic"].stack().rename("rolling_rank_ic"),
            metrics["rolling_rank_icir"].stack().rename("rolling_rank_icir"),
        ],
        axis=1,
    ).reset_index(names=["date", "factor"])
    return returns_df, weights_df, panel_df

