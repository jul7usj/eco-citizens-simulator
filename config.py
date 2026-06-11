"""
EcoCitizens Simulator — parameter configuration.

Every parameter is declared as a (low, base, high) triplet representing the
10th / 50th / 90th percentile of a Beta-PERT distribution. PERT is the
standard distribution for elicited "plausible range" triplets: it respects
the bounds the user gave (won't sample outside [low, high]) and concentrates
mass around the base. For rates already in [0,1] it's used directly; for
non-negative quantities (counts, money) it is also bounded but that is
honest — we are claiming `high` is the 90th percentile, not infinity.

Provenance markers in inline comments:
  FROM_SHEET   — derived from the master xlsx, N events noted.
  ASSUMPTION   — not in data; defensible default. Will print at run time
                 so the user knows what they are accepting.

Until the master .xlsx is loaded, every value is ASSUMPTION. Once
data.py loads the sheet, it overrides what it can and prints a diff.
"""
from dataclasses import dataclass, field
from typing import Dict, Tuple

LBH = Tuple[float, float, float]  # (low, base, high)


@dataclass
class GrowthConfig:
    initial_whatsapp: int = 560          # FROM_SHEET — confirmed community size at J-MED start
    initial_instagram: int = 631         # FROM_SHEET — confirmed community size at J-MED start
    initial_active_base: int = 60        # FROM_SHEET — median EC participants per cleanup event
    # was 41 (N=7 cleanups), rounded up for school/awareness regulars not counted in cleanup turnout

    organic_growth_rate_monthly: LBH = (0.005, 0.015, 0.04)
    # ASSUMPTION — 0.5–4%/mo organic drift for an active local NGO community.

    event_signup_boost: LBH = (3.0, 10.0, 25.0)
    # ASSUMPTION — new WA joins per event. Replace with sheet ratio when N≥10.

    instagram_to_whatsapp_conversion: LBH = (0.002, 0.008, 0.02)
    # ASSUMPTION — fraction of IG followers crossing to WA per month.

    whatsapp_to_active_conversion: LBH = (0.05, 0.15, 0.30)
    # ASSUMPTION — fraction of WA members attending ≥1 event in any 6-month window.

    monthly_churn_rate: LBH = (0.01, 0.025, 0.05)
    # ASSUMPTION — WA groups bleed 1–5%/mo without engagement.

    sponsor_visibility_multiplier: LBH = (1.1, 1.5, 2.5)
    # ASSUMPTION — IG growth lift in months with a visibly sponsored event.

    ig_organic_growth_low: LBH = (0.005, 0.01, 0.02)
    ig_organic_growth_medium: LBH = (0.01, 0.02, 0.04)
    ig_organic_growth_high: LBH = (0.02, 0.04, 0.08)
    # ASSUMPTION — IG monthly organic growth by posting cadence.

    # Lebanon: spring + early autumn best for outdoor events; deep winter
    # (rain) and August (heatwave) suppress turnout and signup velocity.
    seasonal_factor: Dict[int, float] = field(default_factory=lambda: {
        1: 0.7, 2: 0.8, 3: 1.1, 4: 1.3, 5: 1.3, 6: 1.1,
        7: 0.9, 8: 0.7, 9: 1.1, 10: 1.2, 11: 1.0, 12: 0.7,
    })
    # ASSUMPTION — replace with Lebanon climatology when available.


@dataclass
class StrategyConfig:
    # Per event-type ratios. Keys: cleanup / hike / awareness / school_campaign.

    turnout_per_active_volunteer: Dict[str, LBH] = field(default_factory=lambda: {
        "cleanup":          (0.10, 0.25, 0.50),
        "hike":             (0.05, 0.15, 0.30),
        "awareness":        (0.03, 0.10, 0.25),
        "school_campaign":  (0.02, 0.08, 0.20),
    })
    # ASSUMPTION — cleanups draw the most committed volunteers.

    bags_per_volunteer: Dict[str, LBH] = field(default_factory=lambda: {
        "cleanup":          (0.5, 1.5, 3.0),
        "hike":             (0.2, 0.6, 1.5),
        "awareness":        (0.0, 0.1, 0.3),
        "school_campaign":  (0.0, 0.2, 0.8),
    })
    # ASSUMPTION — replace with sheet ratios.

    expense_per_volunteer_USD: Dict[str, LBH] = field(default_factory=lambda: {
        "cleanup":          (1.0, 3.0, 8.0),
        "hike":             (0.5, 2.0, 5.0),
        "awareness":        (0.5, 1.5, 4.0),
        "school_campaign":  (0.5, 2.0, 6.0),
    })
    expense_per_volunteer_LBP: Dict[str, LBH] = field(default_factory=lambda: {
        "cleanup":          (50_000, 200_000, 600_000),
        "hike":             (30_000, 100_000, 300_000),
        "awareness":        (20_000, 80_000, 250_000),
        "school_campaign":  (30_000, 120_000, 400_000),
    })
    # ASSUMPTION — LBP and USD never merged.

    # Location categories: urban / coastal / mountain / school
    location_turnout_multiplier: Dict[str, LBH] = field(default_factory=lambda: {
        "urban":    (0.9, 1.1, 1.4),
        "coastal":  (0.9, 1.2, 1.5),
        "mountain": (0.5, 0.8, 1.1),
        "school":   (0.7, 1.0, 1.3),
    })
    # ASSUMPTION — accessibility + visibility per location.

    # Each attending volunteer brings on average X new community members
    # (via word-of-mouth, social posts, friends-of-friends in the same event).
    recruitment_per_attendee: Dict[str, LBH] = field(default_factory=lambda: {
        "cleanup":          (0.05, 0.15, 0.35),
        "hike":             (0.05, 0.20, 0.45),
        "awareness":        (0.02, 0.08, 0.20),
        "school_campaign":  (0.03, 0.10, 0.25),
    })
    # ASSUMPTION — recruitment multiplier per event type.

    location_recruitment_multiplier: Dict[str, LBH] = field(default_factory=lambda: {
        "urban":    (0.8, 1.0, 1.3),
        "coastal":  (1.0, 1.3, 1.8),     # visually striking → IG-worthy
        "mountain": (0.9, 1.2, 1.6),
        "school":   (0.7, 1.0, 1.3),
    })
    # ASSUMPTION — coastal + mountain do best on social.

    sponsor_expense_offset: LBH = (0.3, 0.6, 0.9)
    # ASSUMPTION — fraction of event expense absorbed by sponsor.

    partner_promotion_turnout_boost: LBH = (1.1, 1.3, 1.7)
    # ASSUMPTION — partner co-promo multiplier.

    weather_disruption_probability_monthly: Dict[int, LBH] = field(default_factory=lambda: {
        1: (0.20, 0.35, 0.50), 2: (0.15, 0.30, 0.45),
        3: (0.10, 0.20, 0.30), 4: (0.05, 0.10, 0.20),
        5: (0.02, 0.05, 0.12), 6: (0.01, 0.03, 0.08),
        7: (0.01, 0.02, 0.05), 8: (0.05, 0.10, 0.20),
        9: (0.03, 0.08, 0.18), 10: (0.10, 0.20, 0.30),
        11: (0.15, 0.30, 0.45), 12: (0.20, 0.35, 0.50),
    })
    # ASSUMPTION — rain in winter, heat in Aug. Replace with climatology.


@dataclass
class RunConfig:
    n_runs: int = 10_000
    horizon_months_growth: int = 12     # J-MED grant period: Jul 2026 → Jun 2027
    horizon_months_strategy: int = 12
    random_seed: int = 42
    results_dir: str = "results"


EVENT_TYPES = ("cleanup", "hike", "awareness", "school_campaign")
LOCATIONS = ("urban", "coastal", "mountain", "school")
