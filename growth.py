"""
Growth simulator — Monte Carlo projection of community size over H months.

State per month (vectorised across runs):
  whatsapp[t], instagram[t], active_base[t]

Update rules (per month, all parameters resampled each month from PERT
across `n_runs` independent runs):

  organic_new       = whatsapp[t-1] * organic_growth_rate_monthly
  event_new         = sum_of_event_signup_boosts * seasonal_factor[month]
  ig_to_wa          = instagram[t-1] * ig_to_wa_conversion
  churn             = whatsapp[t-1] * monthly_churn_rate
  whatsapp[t]       = whatsapp[t-1] + organic_new + event_new + ig_to_wa - churn

  ig_organic_growth = instagram[t-1] * ig_organic_by_posting_cadence
  ig_sponsor_lift   = (sponsor_multiplier - 1) if a sponsored event ran
  instagram[t]      = instagram[t-1] + ig_organic_growth + ig_sponsor_lift

  active_base[t]    = whatsapp[t] * whatsapp_to_active_conversion

We compute trajectories for each Strategy the user defines and return
percentile bands plus a sensitivity-analysis table.

Sensitivity is computed by correlating each input parameter sample with
the month-H WhatsApp outcome across runs (Spearman rank), so we get a
nonparametric "which inputs explain the variance" ranking without
fitting a regression.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import GrowthConfig, RunConfig
from distributions import sample_pert


@dataclass
class Strategy:
    name: str
    events_per_month: int = 1
    # "none" | "occasional" (~30%) | "regular" (~70%) probability a given event is sponsored
    sponsor_presence: str = "occasional"
    # "low" | "medium" | "high"
    posting_frequency: str = "medium"
    # Probability a given event has partner-NGO co-promotion
    partner_copromo_rate: float = 0.0


SPONSOR_RATE = {"none": 0.0, "occasional": 0.3, "regular": 0.7}


def _ig_posting_triplet(cfg: GrowthConfig, freq: str):
    return {
        "low": cfg.ig_organic_growth_low,
        "medium": cfg.ig_organic_growth_medium,
        "high": cfg.ig_organic_growth_high,
    }[freq]


def simulate_growth(
    cfg: GrowthConfig, run: RunConfig, strategy: Strategy
) -> Dict[str, np.ndarray]:
    """
    Run the growth MC simulation for one Strategy.

    Returns:
      {
        "whatsapp":   ndarray shape (n_runs, H+1),
        "instagram":  ndarray shape (n_runs, H+1),
        "active":     ndarray shape (n_runs, H+1),
        "param_draws": dict of name -> ndarray (n_runs,) of per-run mean values,
                       used by sensitivity analysis.
      }
    """
    rng = np.random.default_rng(run.random_seed)
    N = run.n_runs
    H = run.horizon_months_growth

    whatsapp = np.zeros((N, H + 1))
    instagram = np.zeros((N, H + 1))
    whatsapp[:, 0] = cfg.initial_whatsapp
    instagram[:, 0] = cfg.initial_instagram

    # Track per-run aggregate parameter values for sensitivity analysis.
    # We sum monthly draws and divide at the end → mean across the horizon.
    param_sums: Dict[str, np.ndarray] = {
        k: np.zeros(N) for k in [
            "organic_growth_rate_monthly",
            "event_signup_boost",
            "instagram_to_whatsapp_conversion",
            "whatsapp_to_active_conversion",
            "monthly_churn_rate",
            "sponsor_visibility_multiplier",
            "ig_organic_growth",
        ]
    }

    sponsor_p = SPONSOR_RATE[strategy.sponsor_presence]

    for t in range(1, H + 1):
        # Calendar month for seasonality (project starts at "month 1" → cycle thru)
        cal_month = ((t - 1) % 12) + 1
        season = cfg.seasonal_factor.get(cal_month, 1.0)

        organic_rate = sample_pert(*cfg.organic_growth_rate_monthly, n=N, rng=rng)
        ig_to_wa = sample_pert(*cfg.instagram_to_whatsapp_conversion, n=N, rng=rng)
        churn = sample_pert(*cfg.monthly_churn_rate, n=N, rng=rng)
        ig_organic = sample_pert(*_ig_posting_triplet(cfg, strategy.posting_frequency),
                                  n=N, rng=rng)
        sponsor_mult = sample_pert(*cfg.sponsor_visibility_multiplier, n=N, rng=rng)

        # Event-driven signups: one draw per event, summed.
        event_new = np.zeros(N)
        any_sponsored = np.zeros(N, dtype=bool)
        for _ in range(strategy.events_per_month):
            boost = sample_pert(*cfg.event_signup_boost, n=N, rng=rng)
            event_new += boost * season
            sponsored = rng.random(N) < sponsor_p
            any_sponsored |= sponsored

        organic_new = whatsapp[:, t - 1] * organic_rate
        ig_crossed = instagram[:, t - 1] * ig_to_wa
        churned = whatsapp[:, t - 1] * churn

        whatsapp[:, t] = np.maximum(
            0.0, whatsapp[:, t - 1] + organic_new + event_new + ig_crossed - churned
        )

        ig_org_growth = instagram[:, t - 1] * ig_organic
        ig_sponsor_lift = np.where(any_sponsored,
                                    instagram[:, t - 1] * ig_organic * (sponsor_mult - 1.0),
                                    0.0)
        instagram[:, t] = instagram[:, t - 1] + ig_org_growth + ig_sponsor_lift

        # Accumulate for sensitivity (mean across months)
        param_sums["organic_growth_rate_monthly"] += organic_rate
        param_sums["instagram_to_whatsapp_conversion"] += ig_to_wa
        param_sums["monthly_churn_rate"] += churn
        param_sums["ig_organic_growth"] += ig_organic
        param_sums["sponsor_visibility_multiplier"] += sponsor_mult
        if strategy.events_per_month > 0:
            param_sums["event_signup_boost"] += event_new / max(1, strategy.events_per_month)

    # Active base derived from final WA size with a single conversion draw
    # (this is the "in the past 6 months" interpretation — a per-run rate).
    active_rate = sample_pert(*cfg.whatsapp_to_active_conversion, n=N, rng=rng)
    param_sums["whatsapp_to_active_conversion"] = active_rate
    active = whatsapp * active_rate[:, None]

    return {
        "whatsapp": whatsapp,
        "instagram": instagram,
        "active": active,
        "param_draws": {k: v / H for k, v in param_sums.items()
                         if k != "whatsapp_to_active_conversion"} | {
            "whatsapp_to_active_conversion": param_sums["whatsapp_to_active_conversion"]
        },
    }


def summarise_trajectory(traj: np.ndarray) -> pd.DataFrame:
    """Per-month median and 10–90 percentile band across runs."""
    p10 = np.percentile(traj, 10, axis=0)
    p50 = np.percentile(traj, 50, axis=0)
    p90 = np.percentile(traj, 90, axis=0)
    return pd.DataFrame({"month": np.arange(traj.shape[1]),
                          "p10": p10, "p50": p50, "p90": p90})


def sensitivity_ranking(
    param_draws: Dict[str, np.ndarray], outcome: np.ndarray
) -> pd.DataFrame:
    """Spearman correlation of each input with the outcome at month H."""
    rows = []
    for name, draws in param_draws.items():
        rho, _ = spearmanr(draws, outcome)
        rows.append({"parameter": name, "spearman_rho": float(rho),
                      "abs_rho": float(abs(rho))})
    df = pd.DataFrame(rows).sort_values("abs_rho", ascending=False).reset_index(drop=True)
    df["share_of_signal"] = df["abs_rho"] / df["abs_rho"].sum()
    return df.drop(columns=["abs_rho"])
