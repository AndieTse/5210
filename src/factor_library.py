from __future__ import annotations

import pandas as pd


def _validate_columns(df: pd.DataFrame) -> None:
    required = {"date", "asset", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    _validate_columns(df)
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["asset", "date"]).reset_index(drop=True)
    return out


def _safe_cs_zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if pd.isna(std) or std == 0:
        return series * 0.0
    return (series - series.mean()) / std


def add_base_factors(df: pd.DataFrame) -> pd.DataFrame:
    out = prepare_data(df)
    g = out.groupby("asset", group_keys=False)

    out["ret_1d"] = g["close"].pct_change(1)
    factor_cols = []

    for w in [3, 5, 10, 20, 40, 60]:
        mom_col = f"mom_{w}"
        rev_col = f"rev_{w}"
        out[mom_col] = g["close"].pct_change(w)
        out[rev_col] = -out[mom_col]
        factor_cols.extend([mom_col, rev_col])

    for w in [10, 20, 40, 60]:
        vol_col = f"vol_{w}"
        out[vol_col] = g["ret_1d"].rolling(w).std().reset_index(level=0, drop=True)
        factor_cols.append(vol_col)

    for w in [5, 10, 20, 40]:
        vm_col = f"volume_mom_{w}"
        vma_col = f"volume_to_ma{w}"
        pvc_col = f"price_volume_corr_{w}"
        pvv_col = f"price_volume_cov_{w}"

        out[vm_col] = g["volume"].pct_change(w)
        out[vma_col] = out["volume"] / g["volume"].rolling(w).mean().reset_index(level=0, drop=True)
        out[pvc_col] = g.apply(
            lambda x: x["close"].rolling(w).corr(x["volume"])
        ).reset_index(level=0, drop=True)
        out[pvv_col] = g.apply(
            lambda x: x["close"].rolling(w).cov(x["volume"])
        ).reset_index(level=0, drop=True)
        factor_cols.extend([vm_col, vma_col, pvc_col, pvv_col])

    out["price_volume_trend_20"] = out["mom_20"] * out["volume_mom_20"]
    out["price_volume_trend_40"] = out["mom_40"] * out["volume_mom_40"]
    factor_cols.extend(["price_volume_trend_20", "price_volume_trend_40"])

    for col in factor_cols:
        out[col] = out.groupby("date")[col].transform(_safe_cs_zscore)

    return out

