"""Loading the ARMD tables.

Every read goes through :func:`load_table`, so the textual null token, the column
restriction and the logging are applied in one place. A missing optional table returns
``None`` and is reported; a missing required table is caught earlier by
:func:`armd.config.check_required`.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

NA_TOKENS = ["Null", "NULL", "null", "NaN", ""]


def load_table(
    data_dir: str,
    name: str,
    usecols: Optional[Sequence[str]] = None,
    parse_dates: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> Optional[pd.DataFrame]:
    path = os.path.join(data_dir, name)
    if not os.path.exists(path):
        if verbose:
            print(f"  [skip] table not found: {name}")
        return None
    df = pd.read_csv(
        path, usecols=usecols, parse_dates=parse_dates,
        na_values=NA_TOKENS, keep_default_na=True, low_memory=False,
    )
    if verbose:
        print(f"  [load] {name}: {len(df):,} rows x {df.shape[1]} cols")
    return df


def restrict_to_patients(df: Optional[pd.DataFrame], patients) -> Optional[pd.DataFrame]:
    if df is None or patients is None or "anon_id" not in df.columns:
        return df
    return df[df["anon_id"].isin(patients)].copy()


def empty_frame(feature: str) -> pd.DataFrame:
    return pd.DataFrame({"order_proc_id_coded": pd.Series(dtype="int64"),
                         feature: pd.Series(dtype="float64")})


def prior_count(
    data_dir: str,
    name: str,
    usecols: Sequence[str],
    time_col: str,
    feature: str,
    patients=None,
    window_days: Optional[float] = None,
) -> pd.DataFrame:
    """Count prior events per culture order.

    The ``*_time_to_culturetime`` columns are positive for events that happened before the
    culture order. ``window_days`` optionally bounds the look-back; without it the count covers
    the whole recorded history.
    """
    df = load_table(data_dir, name, usecols=list(usecols))
    if df is None:
        return empty_frame(feature)
    df = restrict_to_patients(df, patients)
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df = df[df[time_col] >= 0]
    if window_days is not None:
        df = df[df[time_col] <= window_days]
    if not len(df):
        return empty_frame(feature)
    return df.groupby("order_proc_id_coded").size().rename(feature).reset_index()


def chunked_prior_count(
    data_dir: str,
    name: str,
    anon_col: str,
    order_col: str,
    time_col: str,
    feature: str,
    patients=None,
    chunksize: int = 2_000_000,
) -> pd.DataFrame:
    """Same as :func:`prior_count` but streams the file.

    Used for the comorbidity table, which is about 18 GB.
    """
    path = os.path.join(data_dir, name)
    if not os.path.exists(path):
        print(f"  [skip] table not found: {name}")
        return empty_frame(feature)
    parts = []
    for chunk in pd.read_csv(
        path, usecols=[anon_col, order_col, time_col], na_values=NA_TOKENS,
        chunksize=chunksize, low_memory=False,
    ):
        if patients is not None:
            chunk = chunk[chunk[anon_col].isin(patients)]
        chunk[time_col] = pd.to_numeric(chunk[time_col], errors="coerce")
        chunk = chunk[chunk[time_col] >= 0]
        if len(chunk):
            parts.append(chunk.groupby(order_col).size())
    if not parts:
        return empty_frame(feature)
    out = pd.concat(parts).groupby(level=0).sum().rename(feature).reset_index()
    return out.rename(columns={order_col: "order_proc_id_coded"})


def reduce_medians(
    data_dir: str,
    name: str,
    prefix: str,
    stubs: Iterable[str],
    patients=None,
) -> pd.DataFrame:
    """One row per culture order holding the median of the requested measurements."""
    df = load_table(data_dir, name)
    if df is None:
        return pd.DataFrame({"order_proc_id_coded": pd.Series(dtype="int64")})
    df = restrict_to_patients(df, patients)
    keep = ["order_proc_id_coded"]
    for stub in stubs:
        col = prefix + stub
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            keep.append(col)
    if len(keep) == 1:
        print(f"  [warn] {name}: none of the requested columns present ({prefix}*)")
    return df[keep].groupby("order_proc_id_coded", as_index=False).median()


def binary_from_numeric(series: pd.Series) -> np.ndarray:
    """Turn a 0/1/Null column into a float indicator.

    ``demographics.gender`` is 0/1/Null and pandas reads it as float64, so the comparison is
    numeric; comparing against the string "1" would always be false.
    """
    return (pd.to_numeric(series, errors="coerce") == 1).fillna(False).to_numpy(dtype=np.float32)
