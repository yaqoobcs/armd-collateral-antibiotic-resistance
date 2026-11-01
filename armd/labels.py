"""Treatment and collateral-resistance outcome construction."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def next_culture_resistance(
    frame: pd.DataFrame,
    horizon_days: float,
    patient_col: str = "anon_id",
    time_col: str = "order_time_jittered_utc",
    value_col: str = "resistant",
    missing_as_negative: bool = False,
) -> pd.Series:
    """For each culture, the resistance status of the patient's *next* culture.

    The next culture is the first one at a strictly later timestamp; cultures sharing a
    timestamp are not treated as each other's successor. The value is NaN when there is no
    next culture, or when the next culture falls outside ``horizon_days``.

    ``missing_as_negative=True`` instead codes those cultures as 0. Dropping them, as the
    default does, conditions the analysis on being cultured again; having both codings makes
    that dependence measurable.

    Returns a Series aligned to ``frame.index``.
    """
    out = pd.Series(np.nan, index=frame.index, dtype="float64")
    if not len(frame):
        return out
    horizon = pd.Timedelta(days=horizon_days)

    ordered = frame.sort_values([patient_col, time_col], kind="mergesort")
    for _, grp in ordered.groupby(patient_col, sort=False):
        times = grp[time_col].to_numpy()
        values = grp[value_col].to_numpy(dtype="float64")
        n = len(times)
        nxt = np.searchsorted(times, times, side="right")
        has_next = nxt < n
        res = np.full(n, np.nan)
        if has_next.any():
            idx = nxt[has_next]
            within = (times[idx] - times[has_next]) <= horizon
            res[has_next] = np.where(within, values[idx], np.nan)
        if missing_as_negative:
            res = np.nan_to_num(res, nan=0.0)
        out.loc[grp.index] = res
    return out


def collapse_class_resistance(
    definitive: pd.DataFrame,
    drug_class: str,
    class_col: str = "drug_class",
) -> pd.DataFrame:
    """One resistance value per culture order for a drug class.

    A culture tested against several agents of the same class is resistant to the class
    if it is resistant to any of them, which is the conservative reading.
    """
    sub = definitive[definitive[class_col] == drug_class]
    if not len(sub):
        return pd.DataFrame(
            columns=["anon_id", "order_proc_id_coded", "order_time_jittered_utc", "resistant"]
        )
    return (
        sub.groupby(["anon_id", "order_proc_id_coded", "order_time_jittered_utc"], as_index=False)[
            "resistant"
        ]
        .max()
        .sort_values(["anon_id", "order_time_jittered_utc"], kind="mergesort")
        .reset_index(drop=True)
    )


def order_organisms(definitive: pd.DataFrame) -> pd.Series:
    """order_proc_id_coded -> the set of organisms that culture grew."""
    if not len(definitive) or "organism" not in definitive.columns:
        return pd.Series(dtype="object")
    return definitive.groupby("order_proc_id_coded")["organism"].agg(
        lambda v: frozenset(x for x in v.dropna().unique())
    )


def collateral_outcome_map(
    definitive: pd.DataFrame,
    drug_class: str,
    horizon_days: float,
    missing_as_negative: bool = False,
    require_shared_organism: bool = False,
) -> pd.Series:
    """order_proc_id_coded -> collateral resistance at the next culture (0/1).

    By default the next culture counts however it was grown, so the outcome is "the
    patient's next infection resists this class". That is the decision-relevant quantity,
    but it pools two mechanisms: resistance acquired within an organism, and replacement
    of the flora by an organism that was never susceptible. A course of fluoroquinolone
    that clears susceptible Enterobacterales and leaves an enterococcus behind scores an
    event without any acquired resistance anywhere.

    ``require_shared_organism=True`` restricts the outcome to next cultures that grow at
    least one organism in common with the index culture, which isolates the acquired
    pathway. It is the right sensitivity analysis for the mechanistic claim and the wrong
    primary analysis for the clinical one, so we report both.
    """
    collapsed = collapse_class_resistance(definitive, drug_class)
    if not len(collapsed):
        return pd.Series(dtype="float64")
    collapsed["y"] = next_culture_resistance(
        collapsed, horizon_days, missing_as_negative=missing_as_negative
    )

    if require_shared_organism:
        orgs = order_organisms(definitive)
        collapsed["_next_order"] = _next_order_id(collapsed, horizon_days)
        idx_org = collapsed["order_proc_id_coded"].map(orgs)
        nxt_org = collapsed["_next_order"].map(orgs)
        shared = [
            bool(a) and bool(b) and bool(a & b) if isinstance(a, frozenset)
            and isinstance(b, frozenset) else False
            for a, b in zip(idx_org, nxt_org)
        ]
        collapsed.loc[~np.array(shared), "y"] = np.nan

    keep = collapsed.dropna(subset=["y"]).drop_duplicates("order_proc_id_coded")
    return keep.set_index("order_proc_id_coded")["y"].astype(int)


def _next_order_id(frame: pd.DataFrame, horizon_days: float,
                   patient_col: str = "anon_id",
                   time_col: str = "order_time_jittered_utc") -> pd.Series:
    """The order id of each culture's successor, by the same rule as the outcome."""
    out = pd.Series(np.nan, index=frame.index, dtype="object")
    if not len(frame):
        return out
    horizon = pd.Timedelta(days=horizon_days)
    ordered = frame.sort_values([patient_col, time_col], kind="mergesort")
    for _, grp in ordered.groupby(patient_col, sort=False):
        times = grp[time_col].to_numpy()
        ids = grp["order_proc_id_coded"].to_numpy()
        n = len(times)
        nxt = np.searchsorted(times, times, side="right")
        has_next = nxt < n
        res = np.full(n, None, dtype=object)
        if has_next.any():
            j = nxt[has_next]
            within = (times[j] - times[has_next]) <= horizon
            res[has_next] = np.where(within, ids[j], None)
        out.loc[grp.index] = res
    return out


def exposure_indicators(
    exposure: Optional[pd.DataFrame],
    window_days: float,
    class_col: str = "drug_class",
    time_col: str = "time_to_culturetime",
) -> pd.DataFrame:
    """Wide 0/1 indicators of prior exposure to each drug class within the window.

    The window looks backward: ``time_to_culturetime`` is positive for medications given
    before the culture order. This is prior exposure, not therapy started at the time of the
    culture.
    """
    if exposure is None or not len(exposure):
        return pd.DataFrame({"order_proc_id_coded": pd.Series(dtype="int64")})
    t = pd.to_numeric(exposure[time_col], errors="coerce")
    sub = exposure[(t >= 0) & (t <= window_days) & exposure[class_col].notna()]
    if not len(sub):
        return pd.DataFrame({"order_proc_id_coded": pd.Series(dtype="int64")})
    wide = (
        sub.assign(_v=1)
        .pivot_table(index="order_proc_id_coded", columns=class_col, values="_v", aggfunc="max")
        .add_prefix("prior_expo_")
        .reset_index()
    )
    wide.columns.name = None
    return wide


def exposed_orders(exposure_window: pd.DataFrame, drug_class: str) -> set:
    """Culture orders preceded by exposure to ``drug_class`` inside the window."""
    if exposure_window is None or not len(exposure_window):
        return set()
    sel = exposure_window[exposure_window["drug_class"] == drug_class]
    return set(sel["order_proc_id_coded"].unique())
