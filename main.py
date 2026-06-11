"""
Orchestrator. Run this to do an end-to-end pass:

    python main.py --sheet path/to/master.xlsx

Without --sheet, every parameter stays at the config default (all ASSUMPTION)
and the run will print a loud warning. That mode is for inspecting the
simulator shape; it's not a real recommendation.

Sequence:
  1. Load + summarise sheet, print DQ issues.
  2. Derive ratio overrides from sheet, merge into StrategyConfig.
  3. Run Growth simulator for 2–3 named strategies, save trajectory plots
     and CSVs.
  4. Feed the median active-base trajectory into the Strategy simulator's
     combo ranker → answers "which location × event_type boost us most".
  5. Sanity-check vs. historical sheet. Flag if simulated medians are >3×
     or <1/3× the historical medians.
  6. Write run_log.txt with every parameter actually used.

Outputs land in `results/` next to this file.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from textwrap import dedent
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")  # No display server on this machine; render to PNG.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import EVENT_TYPES, LOCATIONS, GrowthConfig, RunConfig, StrategyConfig
from data import derive_overrides, load_sheet, print_summary, summarise
from growth import Strategy, sensitivity_ranking, simulate_growth, summarise_trajectory
from strategy import compare_combos, sanity_check, simulate_plan


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def _apply_overrides(scfg: StrategyConfig, overrides: dict) -> List[str]:
    """Merge sheet-derived overrides into the StrategyConfig. Returns log lines."""
    applied: List[str] = []
    for group, per_type in overrides.items():
        target = getattr(scfg, group)
        for event_type, triplet in per_type.items():
            if event_type in target:
                applied.append(
                    f"  FROM_SHEET -> {group}[{event_type}] = "
                    f"({triplet[0]:.4g}, {triplet[1]:.4g}, {triplet[2]:.4g}) "
                    f"(was {target[event_type]})"
                )
                target[event_type] = triplet
    return applied


def _plot_trajectory(traj_df: pd.DataFrame, label: str, ax) -> None:
    ax.fill_between(traj_df["month"], traj_df["p10"], traj_df["p90"], alpha=0.2)
    ax.plot(traj_df["month"], traj_df["p50"], linewidth=2, label=label)


def run_growth_strategies(
    gcfg: GrowthConfig, scfg: StrategyConfig, run: RunConfig,
    strategies: List[Strategy],
) -> dict:
    """Run growth sim for each strategy, save trajectory plots + comparison table."""
    print("\n" + "=" * 60)
    print("GROWTH SIMULATOR")
    print("=" * 60)

    results = {}
    for state_var in ("whatsapp", "instagram", "active"):
        fig, ax = plt.subplots(figsize=(9, 5))
        for strat in strategies:
            r = results.setdefault(strat.name, simulate_growth(gcfg, run, strat))
            traj_df = summarise_trajectory(r[state_var])
            _plot_trajectory(traj_df, strat.name, ax)
        ax.set_title(f"Growth — {state_var} (median + 10–90% band)")
        ax.set_xlabel("Month from now")
        ax.set_ylabel(state_var)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(RESULTS / f"growth_{state_var}.png", dpi=120)
        plt.close(fig)

    # Comparison table at horizon
    rows = []
    for strat in strategies:
        r = results[strat.name]
        for state_var in ("whatsapp", "instagram", "active"):
            final = r[state_var][:, -1]
            rows.append({
                "strategy": strat.name,
                "state": state_var,
                "p10": float(np.percentile(final, 10)),
                "p50": float(np.percentile(final, 50)),
                "p90": float(np.percentile(final, 90)),
            })
    comp = pd.DataFrame(rows)
    comp.to_csv(RESULTS / "growth_comparison.csv", index=False)
    print(comp.to_string(index=False))

    # Sensitivity (using the first strategy as the baseline)
    base = results[strategies[0].name]
    sens = sensitivity_ranking(base["param_draws"], base["whatsapp"][:, -1])
    sens.to_csv(RESULTS / "growth_sensitivity.csv", index=False)
    print("\nSensitivity (driver ranking on month-H WhatsApp size):")
    print(sens.to_string(index=False))

    return results


def run_combo_ranker(
    scfg: StrategyConfig, run: RunConfig, active_base_vec: np.ndarray,
    month: int = 6, sponsored: bool = False, partner_copromo: bool = False,
) -> pd.DataFrame:
    """Run the (event_type × location) combo ranker."""
    print("\n" + "=" * 60)
    print(f"STRATEGY SIMULATOR — combo ranker "
          f"(month={month}, sponsored={sponsored}, partner_copromo={partner_copromo})")
    print("=" * 60)
    df = compare_combos(active_base_vec, scfg, run,
                         month=month, sponsored=sponsored,
                         partner_copromo=partner_copromo)
    df.to_csv(RESULTS / "combo_ranking.csv", index=False)

    print("\nTop 5 combos by AVERAGE rank across recruitment / retention / efficiency:")
    cols = ["event_type", "location",
             "new_community_p50", "new_active_p50",
             "recruit_per_usd_p50",
             "rank_recruitment", "rank_retention", "rank_efficiency", "rank_avg"]
    print(df[cols].head(5).to_string(index=False))

    # Plot: heatmap of new_community by (event_type, location)
    pivot = df.pivot(index="event_type", columns="location",
                      values="new_community_p50")
    pivot = pivot.reindex(index=list(EVENT_TYPES), columns=list(LOCATIONS))
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlGn")
    ax.set_xticks(range(len(LOCATIONS)), LOCATIONS)
    ax.set_yticks(range(len(EVENT_TYPES)), EVENT_TYPES)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.1f}",
                     ha="center", va="center", color="black", fontsize=9)
    ax.set_title("Median new community members per event (combo grid)")
    fig.colorbar(im, ax=ax, label="new community members")
    fig.tight_layout()
    fig.savefig(RESULTS / "combo_heatmap_recruitment.png", dpi=120)
    plt.close(fig)

    return df


def write_run_log(
    args, gcfg: GrowthConfig, scfg: StrategyConfig, run: RunConfig,
    overrides_log: List[str], sanity_flags: List[str],
) -> None:
    """Write run_log.txt with every parameter actually used."""
    lines = ["# EcoCitizens Simulator — run log",
              f"# args: {vars(args)}",
              ""]
    lines.append("[RunConfig]")
    lines.append(json.dumps(asdict(run), indent=2))
    lines.append("")
    lines.append("[GrowthConfig]")
    lines.append(json.dumps({k: v for k, v in asdict(gcfg).items()
                              if not k.startswith("_")},
                              indent=2, default=str))
    lines.append("")
    lines.append("[StrategyConfig]")
    lines.append(json.dumps(asdict(scfg), indent=2, default=str))
    lines.append("")
    lines.append("[Overrides applied from sheet]")
    if overrides_log:
        lines.extend(overrides_log)
    else:
        lines.append("  (none — every parameter is ASSUMPTION)")
    lines.append("")
    lines.append("[Sanity flags]")
    lines.extend(sanity_flags if sanity_flags else ["  (none)"])
    (RESULTS / "run_log.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sheet", type=str, default=None,
                         help="Path to master xlsx. If omitted, all params stay ASSUMPTION.")
    parser.add_argument("--runs", type=int, default=None,
                         help="Override Monte Carlo run count.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--horizon-growth", type=int, default=None)
    parser.add_argument("--horizon-strategy", type=int, default=None)
    parser.add_argument("--combo-month", type=int, default=6,
                         help="Calendar month (1-12) the combo ranker simulates the event in.")
    parser.add_argument("--combo-sponsored", action="store_true")
    parser.add_argument("--combo-partner", action="store_true")
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)

    gcfg = GrowthConfig()
    scfg = StrategyConfig()
    run = RunConfig()
    if args.runs is not None: run.n_runs = args.runs
    if args.seed is not None: run.random_seed = args.seed
    if args.horizon_growth is not None: run.horizon_months_growth = args.horizon_growth
    if args.horizon_strategy is not None: run.horizon_months_strategy = args.horizon_strategy

    # 1. Sheet
    sheet = None
    overrides_log: List[str] = []
    if args.sheet:
        sheet = load_sheet(args.sheet)
        print_summary(summarise(sheet))
        overrides = derive_overrides(sheet)
        overrides_log = _apply_overrides(scfg, overrides)
        if overrides_log:
            print("\nApplied sheet-derived overrides:")
            for line in overrides_log:
                print(line)
        else:
            print("\nNo overrides could be derived (need ≥3 events per type).")
    else:
        print("WARNING: no --sheet provided. Every parameter is ASSUMPTION. "
               "This run is for inspecting simulator shape only.")

    # 2. Default strategy panel for growth
    strategies = [
        Strategy(name="status_quo",     events_per_month=1, sponsor_presence="occasional",
                  posting_frequency="medium"),
        Strategy(name="growth_push",    events_per_month=2, sponsor_presence="occasional",
                  posting_frequency="high"),
        Strategy(name="sponsor_heavy",  events_per_month=2, sponsor_presence="regular",
                  posting_frequency="high"),
    ]
    growth_results = run_growth_strategies(gcfg, scfg, run, strategies)

    # 3. Combo ranker — feed the median active base at month `combo-month`
    # from the baseline (status_quo) growth strategy.
    base_active = growth_results["status_quo"]["active"][:, args.combo_month]
    combo_df = run_combo_ranker(scfg, run, base_active,
                                  month=args.combo_month,
                                  sponsored=args.combo_sponsored,
                                  partner_copromo=args.combo_partner)

    # 4. Optional: simulate a concrete 12-event plan built from the top combo
    # as a worked example.
    top = combo_df.iloc[0]
    from strategy import Event
    plan = [Event(month=((i % 12) + 1), event_type=top["event_type"],
                   location=top["location"],
                   sponsored=args.combo_sponsored,
                   partner_copromo=args.combo_partner) for i in range(12)]
    # Build active_base_by_month from the status_quo growth trajectory
    n = run.n_runs
    active_by_month = np.zeros((n, 13))
    active_by_month[:, 1:] = growth_results["status_quo"]["active"][:, 1:13]
    active_by_month[:, 0] = growth_results["status_quo"]["active"][:, 0]
    plan_result = simulate_plan(plan, "top_combo_repeated_12x",
                                  active_by_month, scfg, run)
    sanity_flags = sanity_check(plan_result, sheet)

    print("\n" + "=" * 60)
    print("TOP-COMBO 12-MONTH PLAN (top combo repeated each month)")
    print("=" * 60)
    print(f"  Combo                 : {top['event_type']} × {top['location']}")
    print(f"  Events held (p50)     : {np.percentile(plan_result.events_held, 50):.0f}")
    print(f"  Total attendees (p50) : {np.percentile(plan_result.attendees_total, 50):.0f}")
    print(f"  New community  (p50)  : {np.percentile(plan_result.new_community_members, 50):.0f}")
    print(f"  New active     (p50)  : {np.percentile(plan_result.new_active_base, 50):.0f}")
    print(f"  Bags           (p50)  : {np.percentile(plan_result.bags_total, 50):.0f}")
    print(f"  Expenses USD   (p50)  : {np.percentile(plan_result.expense_usd_total, 50):,.0f}")
    print(f"  Expenses LBP   (p50)  : {np.percentile(plan_result.expense_lbp_total, 50):,.0f}")
    if sanity_flags:
        print("\nSANITY-CHECK FLAGS:")
        for f in sanity_flags:
            print(f"  ! {f}")

    write_run_log(args, gcfg, scfg, run, overrides_log, sanity_flags)
    print(f"\nResults written to: {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
