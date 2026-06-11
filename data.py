"""
Master xlsx loader + data-quality checks + sheet → config overrides.

The sheet has one row per event with at minimum:
    date, type, location, total_participants, ecocitizens_participants,
    collaborator_or_sponsor, expense_LBP, expense_USD, bags, weather, notes.

Column names in the wild vary slightly; `load_sheet` normalises a small
allowlist of aliases. Anything unrecognised is preserved but ignored.

When `derive_overrides(sheet)` runs:
  - It computes ratios that ground real parameters (bags/volunteer by type,
    expense/volunteer by type, turnout/active_base by type).
  - It returns a dict of {param_name: (low, base, high)} pulled from the
    data, with 10th/50th/90th percentiles of the per-event ratio. Anything
    with N<3 events in a category falls back to ASSUMPTION (we return None
    for it; the caller keeps the config default).
  - It logs which overrides were applied so the user sees them at run time.

We deliberately do NOT fit a regression or model on N<30. Ratios only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

LBH = Tuple[float, float, float]

CANONICAL_COLUMNS = {
    "date": ["date", "event_date", "when"],
    "type": ["type", "event_type", "activity", "activity_type"],
    "location": ["location", "place", "site"],
    "location_category": ["location_category", "category", "loc_category"],
    "total_participants": ["total_participants", "participants_total", "attendees", "turnout"],
    "ecocitizens_participants": ["ecocitizens_participants", "ec_participants", "members_attending"],
    "sponsor": ["collaborator_or_sponsor", "sponsor", "collaborator", "partner"],
    "expense_lbp": ["expense_lbp", "cost_lbp", "lbp", "expenses_lbp"],
    "expense_usd": ["expense_usd", "cost_usd", "usd", "expenses_usd"],
    "bags": ["bags", "garbage_bags", "bags_collected", "garbage"],
    "weather": ["weather", "conditions"],
    "notes": ["notes", "comments", "remarks"],
}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [re.sub(r"\s+", "_", c.strip().lower()) for c in out.columns]
    rename = {}
    for canonical, aliases in CANONICAL_COLUMNS.items():
        for alias in aliases:
            if alias in out.columns and alias != canonical:
                rename[alias] = canonical
                break
    return out.rename(columns=rename)


@dataclass
class SheetSummary:
    n_events: int
    date_min: Optional[pd.Timestamp]
    date_max: Optional[pd.Timestamp]
    types_present: List[str]
    locations_present: List[str]
    total_participants: int
    total_lbp: float
    total_usd: float
    total_bags: float
    dq_issues: List[str]


def load_sheet(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Master sheet not found: {path}")
    df = pd.read_excel(path)
    df = _normalise_columns(df)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ("total_participants", "ecocitizens_participants",
              "expense_lbp", "expense_usd", "bags"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def data_quality_issues(df: pd.DataFrame) -> List[str]:
    """Return a list of human-readable DQ issues for the user to triage."""
    issues: List[str] = []

    if "date" in df.columns:
        if df["date"].isna().any():
            issues.append(f"{df['date'].isna().sum()} rows have unparseable dates.")
        if df["date"].duplicated().any():
            dups = df.loc[df["date"].duplicated(keep=False), "date"].dt.date.unique()
            issues.append(f"Duplicate event dates: {list(dups)}.")
    else:
        issues.append("No `date` column found.")

    if "type" in df.columns:
        # Detect obvious typos: rare lowercase variants
        types = df["type"].dropna().astype(str).str.strip().str.lower().value_counts()
        rare = types[types == 1].index.tolist()
        if rare:
            issues.append(f"Event types appearing only once (typos?): {rare}.")
    else:
        issues.append("No `type` column found.")

    for c in ("total_participants", "expense_usd", "expense_lbp", "bags"):
        if c in df.columns:
            n_missing = df[c].isna().sum()
            if n_missing:
                issues.append(f"`{c}`: {n_missing} missing values.")
            if (df[c] < 0).any():
                issues.append(f"`{c}`: negative values present.")

    # Outlier flag: > 3x the 90th percentile is suspicious for small N.
    for c in ("total_participants", "bags"):
        if c in df.columns and df[c].notna().sum() >= 4:
            p90 = df[c].quantile(0.9)
            outliers = df[df[c] > 3 * p90]
            if len(outliers):
                issues.append(
                    f"`{c}`: {len(outliers)} rows > 3× 90th percentile "
                    f"(>{3*p90:.0f}) — check for unit errors."
                )

    return issues


def summarise(df: pd.DataFrame) -> SheetSummary:
    n = len(df)
    dmin = df["date"].min() if "date" in df.columns else None
    dmax = df["date"].max() if "date" in df.columns else None
    types = (
        sorted(df["type"].dropna().astype(str).str.strip().str.lower().unique())
        if "type" in df.columns else []
    )
    locs = (
        sorted(df["location"].dropna().astype(str).str.strip().unique())
        if "location" in df.columns else []
    )
    return SheetSummary(
        n_events=n,
        date_min=dmin, date_max=dmax,
        types_present=types,
        locations_present=locs,
        total_participants=int(df.get("total_participants", pd.Series(dtype=float)).sum()),
        total_lbp=float(df.get("expense_lbp", pd.Series(dtype=float)).sum()),
        total_usd=float(df.get("expense_usd", pd.Series(dtype=float)).sum()),
        total_bags=float(df.get("bags", pd.Series(dtype=float)).sum()),
        dq_issues=data_quality_issues(df),
    )


def _percentile_triplet(values: pd.Series) -> Optional[LBH]:
    """Return (10th, 50th, 90th) percentiles, or None if too few data points."""
    v = values.dropna()
    if len(v) < 3:
        return None
    lo, base, hi = np.percentile(v, [10, 50, 90])
    if lo == hi:  # degenerate, all the same
        spread = max(abs(base) * 0.2, 1.0)
        lo, hi = base - spread, base + spread
    return float(lo), float(base), float(hi)


def derive_overrides(df: pd.DataFrame) -> Dict[str, Dict[str, LBH]]:
    """
    Compute ratio-based parameter overrides from the sheet.

    Returns a nested dict {param_group: {event_type: (low, base, high)}}
    Only includes keys where N>=3 events for that event type. Caller merges
    these into the config and prints what was overridden.
    """
    overrides: Dict[str, Dict[str, LBH]] = {
        "bags_per_volunteer": {},
        "expense_per_volunteer_USD": {},
        "expense_per_volunteer_LBP": {},
    }
    if "type" not in df.columns or "total_participants" not in df.columns:
        return overrides

    t = df["type"].astype(str).str.strip().str.lower()
    for event_type, grp in df.groupby(t):
        if "bags" in grp and grp["bags"].notna().any():
            ratio = grp["bags"] / grp["total_participants"].replace(0, np.nan)
            triplet = _percentile_triplet(ratio)
            if triplet:
                overrides["bags_per_volunteer"][event_type] = triplet
        if "expense_usd" in grp and grp["expense_usd"].notna().any():
            ratio = grp["expense_usd"] / grp["total_participants"].replace(0, np.nan)
            triplet = _percentile_triplet(ratio)
            if triplet:
                overrides["expense_per_volunteer_USD"][event_type] = triplet
        if "expense_lbp" in grp and grp["expense_lbp"].notna().any():
            ratio = grp["expense_lbp"] / grp["total_participants"].replace(0, np.nan)
            triplet = _percentile_triplet(ratio)
            if triplet:
                overrides["expense_per_volunteer_LBP"][event_type] = triplet
    return overrides


def print_summary(summary: SheetSummary) -> None:
    print("=" * 60)
    print("MASTER SHEET SUMMARY")
    print("=" * 60)
    print(f"  Events             : {summary.n_events}")
    if summary.date_min is not None and pd.notna(summary.date_min):
        print(f"  Date range         : {summary.date_min.date()} -> {summary.date_max.date()}")
    print(f"  Event types        : {summary.types_present}")
    print(f"  Locations          : {summary.locations_present}")
    print(f"  Total participants : {summary.total_participants}")
    print(f"  Total expenses LBP : {summary.total_lbp:,.0f}")
    print(f"  Total expenses USD : {summary.total_usd:,.2f}")
    print(f"  Total bags         : {summary.total_bags:,.0f}")
    print("-" * 60)
    if summary.dq_issues:
        print("DATA-QUALITY ISSUES (triage before simulating):")
        for issue in summary.dq_issues:
            print(f"  - {issue}")
    else:
        print("No data-quality issues flagged.")
    print("=" * 60)
