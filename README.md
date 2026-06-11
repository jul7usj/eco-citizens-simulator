# EcoCitizens Event Strategy Simulator

Monte Carlo simulator for evaluating event strategies at a small environmental NGO.
Built to answer: where should we run events, and what kind, to maximize community 
growth and environmental impact?

## What it does

- Runs 10,000 simulated scenarios across 16 event-type × location combinations
- Uses Beta-PERT distributions for elicited parameter ranges (honest about uncertainty)
- Ranks strategies across three lenses: recruitment, retention, cost-efficiency
- Derives parameter overrides from historical event data when available (N≥3 per type)
- Flags calibration gaps via Spearman sensitivity analysis and sanity checks

## Structure

| File | Role |
|------|------|
| `config.py` | All parameters as (low, base, high) PERT triplets. Provenance marked ASSUMPTION vs FROM_SHEET |
| `distributions.py` | Beta-PERT sampler and Bernoulli helper |
| `growth.py` | WhatsApp / Instagram / active-base growth simulation + sensitivity ranking |
| `strategy.py` | Per-event simulation, combo ranker, annual plan simulator, sanity check |
| `data.py` | Master sheet loader, data-quality checks, ratio-based override derivation |
| `main.py` | Orchestrator — runs the full pipeline, writes results and run_log.txt |
| `build_sheet.py` | Helper to build the master events xlsx from raw records |

## How to run

```bash
pip install -r requirements.txt

# Assumption-only run (inspect simulator shape)
python main.py

# With real event data
python main.py --sheet your_events.xlsx --combo-month 9
```

## Key design choices

**Why Beta-PERT?** With N≈12 events, there is not enough data to fit a distribution.
PERT takes three defensible numbers (minimum, most-likely, maximum) and concentrates 
mass around the mode. Standard in risk analysis for small-N expert elicitation.

**Why Spearman rank correlation for sensitivity?** Nonparametric — does not assume 
linearity between inputs and outputs, which the model does not have.

**Honest calibration status:** 2 of ~30 parameters are currently grounded in 
historical data (bags per volunteer for cleanups and hikes, derived from 12 real 
events). The sensitivity analysis identifies which remaining assumptions drive the 
most variance — those are next.

## What the data showed

One event type scheduled purely for community building turned out to produce nearly 
the same environmental yield per volunteer as dedicated cleanup events. The model 
caught this only after real bag-count data replaced the prior assumptions.

One data point was excluded from ratio calculations after producing a physically 
impossible output (3x below the observed minimum). The attendance count was preserved 
for growth modeling. Excluding the corrupted ratio shifted the 12-month projection 
by approximately 80%.
