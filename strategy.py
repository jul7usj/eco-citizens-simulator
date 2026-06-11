"""
Strategy simulator + combo ranker.

Two entry points:

1. `simulate_plan(plan)` — given a 12-month list of events (each with month,
   type, location, sponsor, partner_copromo), return per-run totals for
   the year: attendees, new community members, new active base, bags,
   expense_USD, expense_LBP, events_held (post-weather), and a per-event
   breakdown. Used to compare full annual plans.

2. `compare_combos()` — for the user's "where + what should we do?" question.
   Simulates a single event of every (event_type × location) combination
   under identical conditions and returns a ranked DataFrame on three lenses:
     - recruitment   : new community members per event
     - retention     : new active-base members per event
     - efficiency    : new community members per USD spent
   The three rarely agree at N≈15. That disagreement is the answer.

Correlations: turnout and bags are positively correlated empirically
(a bigger crew picks up more bags). We model this by sampling turnout
first and then bags_per_volunteer independently — the *product* preserves
the positive relationship without needing a copula. If the sheet shows
this correlation is stronger than that implies, we'll revisit.

Sanity check: after simulation we compare median per-event turnout and
bags to the historical sheet. If the median is >3× or <1/3× the historical
median, we flag it loudly. That's the brief's "don't report nonsense" rule.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import EVENT_TYPES, LOCATIONS, RunConfig, StrategyConfig
from distributions import sample_pert


@dataclass
class Event:
    month: int                       # 1..12 from now
    event_type: str                  # cleanup / hike / awareness / school_campaign
    location: str                    # urban / coastal / mountain / school
    sponsored: bool = False
    partner_copromo: bool = False


@dataclass
class PlanResult:
    plan_name: str
    n_runs: int
    # Each ndarray is shape (n_runs,) — one number per simulated year.
    attendees_total: np.ndarray
    new_community_members: np.ndarray
    new_active_base: np.ndarray
    bags_total: np.ndarray
    expense_usd_total: np.ndarray
    expense_lbp_total: np.ndarray
    events_held: np.ndarray        # post-weather count
    # Per-event detail, optional, for debugging
    per_event_attendees: Optional[np.ndarray] = None


def _draw(scfg: StrategyConfig, attr: str, key: str, n: int,
          rng: np.random.Generator) -> np.ndarray:
    """Convenience: sample PERT for a dict-keyed parameter."""
    return sample_pert(*getattr(scfg, attr)[key], n=n, rng=rng)


def simulate_event(
    event: Event,
    active_base_at_month: np.ndarray,    # shape (n_runs,) — fed by growth sim
    scfg: StrategyConfig,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Vectorised single-event simulation. Returns a dict of per-run outcomes."""
    n = active_base_at_month.shape[0]

    # Weather: Bernoulli per run on disruption probability for this month.
    wp = sample_pert(*scfg.weather_disruption_probability_monthly[event.month],
                     n=n, rng=rng)
    happened = (rng.random(n) > wp).astype(float)   # 1.0 if event ran, 0.0 if cancelled

    turnout_rate = _draw(scfg, "turnout_per_active_volunteer", event.event_type, n, rng)
    loc_mult = _draw(scfg, "location_turnout_multiplier", event.location, n, rng)
    partner_mult = (sample_pert(*scfg.partner_promotion_turnout_boost, n=n, rng=rng)
                     if event.partner_copromo else np.ones(n))

    attendees = active_base_at_month * turnout_rate * loc_mult * partner_mult * happened

    bags_rate = _draw(scfg, "bags_per_volunteer", event.event_type, n, rng)
    bags = attendees * bags_rate

    usd_rate = _draw(scfg, "expense_per_volunteer_USD", event.event_type, n, rng)
    lbp_rate = _draw(scfg, "expense_per_volunteer_LBP", event.event_type, n, rng)
    sponsor_offset = (sample_pert(*scfg.sponsor_expense_offset, n=n, rng=rng)
                       if event.sponsored else np.zeros(n))
    expense_usd = attendees * usd_rate * (1.0 - sponsor_offset)
    expense_lbp = attendees * lbp_rate * (1.0 - sponsor_offset)

    recruit_rate = _draw(scfg, "recruitment_per_attendee", event.event_type, n, rng)
    loc_recruit_mult = _draw(scfg, "location_recruitment_multiplier", event.location, n, rng)
    new_community = attendees * recruit_rate * loc_recruit_mult

    return {
        "attendees": attendees,
        "bags": bags,
        "expense_usd": expense_usd,
        "expense_lbp": expense_lbp,
        "new_community": new_community,
        "happened": happened,
    }


def simulate_plan(
    plan: List[Event],
    plan_name: str,
    active_base_by_month: np.ndarray,  # shape (n_runs, 13) — from growth sim, month 0..12
    scfg: StrategyConfig,
    run: RunConfig,
) -> PlanResult:
    """
    Simulate one annual plan. `active_base_by_month[:, m]` gives the active
    volunteer base across runs at the start of calendar month m (1..12).
    """
    rng = np.random.default_rng(run.random_seed + hash(plan_name) % 10_000)
    n = run.n_runs

    attendees = np.zeros(n)
    new_community = np.zeros(n)
    bags = np.zeros(n)
    usd = np.zeros(n)
    lbp = np.zeros(n)
    events_held = np.zeros(n)

    for ev in plan:
        m = max(1, min(12, ev.month))
        base = active_base_by_month[:, m]
        out = simulate_event(ev, base, scfg, rng)
        attendees += out["attendees"]
        bags += out["bags"]
        usd += out["expense_usd"]
        lbp += out["expense_lbp"]
        new_community += out["new_community"]
        events_held += out["happened"]

    # Active-base growth proxy: new active members ≈ new community * whatsapp_to_active.
    # We use a single PERT draw per run (consistent with growth.py's interpretation
    # of `whatsapp_to_active_conversion` as a per-run rate).
    from config import GrowthConfig
    gcfg = GrowthConfig()
    active_conv = sample_pert(*gcfg.whatsapp_to_active_conversion, n=n, rng=rng)
    new_active = new_community * active_conv

    return PlanResult(
        plan_name=plan_name,
        n_runs=n,
        attendees_total=attendees,
        new_community_members=new_community,
        new_active_base=new_active,
        bags_total=bags,
        expense_usd_total=usd,
        expense_lbp_total=lbp,
        events_held=events_held,
    )


def compare_combos(
    active_base_at_month: np.ndarray,  # one vector — typically the month-6 active base
    scfg: StrategyConfig,
    run: RunConfig,
    month: int = 6,                    # which calendar month the hypothetical event sits in
    sponsored: bool = False,
    partner_copromo: bool = False,
) -> pd.DataFrame:
    """
    Simulate a single hypothetical event for each (event_type, location)
    combination and rank them on three lenses.

    Returns a DataFrame with one row per combo (16 rows total) containing
    median + 10/90 percentiles for: attendees, new community members,
    new active base, expense_USD, recruitment_per_USD.
    """
    rng = np.random.default_rng(run.random_seed)
    rows = []
    for et in EVENT_TYPES:
        for loc in LOCATIONS:
            ev = Event(month=month, event_type=et, location=loc,
                        sponsored=sponsored, partner_copromo=partner_copromo)
            out = simulate_event(ev, active_base_at_month, scfg, rng)
            # New active proxy = new community * conversion rate
            from config import GrowthConfig
            active_conv = sample_pert(*GrowthConfig().whatsapp_to_active_conversion,
                                       n=run.n_runs, rng=rng)
            new_active = out["new_community"] * active_conv
            # Recruitment per USD spent (guard divide-by-zero)
            usd = np.where(out["expense_usd"] > 0, out["expense_usd"], np.nan)
            recruit_per_usd = out["new_community"] / usd

            rows.append({
                "event_type": et,
                "location": loc,
                "attendees_p50": float(np.median(out["attendees"])),
                "attendees_p10": float(np.percentile(out["attendees"], 10)),
                "attendees_p90": float(np.percentile(out["attendees"], 90)),
                "new_community_p50": float(np.median(out["new_community"])),
                "new_community_p10": float(np.percentile(out["new_community"], 10)),
                "new_community_p90": float(np.percentile(out["new_community"], 90)),
                "new_active_p50": float(np.median(new_active)),
                "expense_usd_p50": float(np.median(out["expense_usd"])),
                "recruit_per_usd_p50": float(np.nanmedian(recruit_per_usd)),
            })
    df = pd.DataFrame(rows)

    df["rank_recruitment"] = df["new_community_p50"].rank(ascending=False).astype(int)
    df["rank_retention"] = df["new_active_p50"].rank(ascending=False).astype(int)
    df["rank_efficiency"] = df["recruit_per_usd_p50"].rank(ascending=False).astype(int)
    df["rank_avg"] = (df["rank_recruitment"] + df["rank_retention"] + df["rank_efficiency"]) / 3.0

    return df.sort_values("rank_avg").reset_index(drop=True)


def sanity_check(
    plan_result: PlanResult, sheet: Optional[pd.DataFrame]
) -> List[str]:
    """Compare median simulated outputs to historical sheet medians."""
    issues: List[str] = []
    if sheet is None or len(sheet) == 0:
        return ["No sheet loaded — cannot sanity-check against historical data."]
    if plan_result.events_held.mean() == 0:
        return ["Sanity check: simulated plan held 0 events on average. Aborting checks."]

    sim_per_event_attendees = (plan_result.attendees_total / plan_result.events_held.clip(min=1)).mean()
    sim_per_event_bags = (plan_result.bags_total / plan_result.events_held.clip(min=1)).mean()

    if "total_participants" in sheet.columns and sheet["total_participants"].notna().any():
        hist = sheet["total_participants"].median()
        ratio = sim_per_event_attendees / max(hist, 1e-6)
        if ratio > 3 or ratio < 1 / 3:
            issues.append(
                f"Simulated attendees/event ({sim_per_event_attendees:.1f}) is "
                f"{ratio:.1f}× historical median ({hist:.1f}). Check params."
            )

    if "bags" in sheet.columns and sheet["bags"].notna().any():
        hist = sheet["bags"].median()
        ratio = sim_per_event_bags / max(hist, 1e-6)
        if ratio > 3 or ratio < 1 / 3:
            issues.append(
                f"Simulated bags/event ({sim_per_event_bags:.1f}) is "
                f"{ratio:.1f}× historical median ({hist:.1f}). Check params."
            )
    return issues
