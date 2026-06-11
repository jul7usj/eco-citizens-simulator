import pandas as pd
import numpy as np

events = [
    # === FOUNDING ERA (2023-2024) — no social media, no alpha team ===
    {"era": "founding", "date": "2023-01-01", "type": "school_campaign", "location": "Zahle",  "location_category": "school",   "total_participants": 96,  "ecocitizens_participants": None, "sponsor": None, "expense_lbp": None, "expense_usd": None, "bags": None, "notes": "Sainte Famille school; date approximate"},
    {"era": "founding", "date": "2023-01-01", "type": "school_campaign", "location": "Ksara",  "location_category": "school",   "total_participants": 114, "ecocitizens_participants": None, "sponsor": None, "expense_lbp": None, "expense_usd": None, "bags": None, "notes": "Antonines school; date approximate"},
    {"era": "founding", "date": "2023-01-01", "type": "school_campaign", "location": "Zahle",  "location_category": "school",   "total_participants": 107, "ecocitizens_participants": None, "sponsor": None, "expense_lbp": None, "expense_usd": None, "bags": None, "notes": "Saint Coeur school; date approximate"},
    {"era": "founding", "date": "2023-01-01", "type": "school_campaign", "location": "Rayak",  "location_category": "school",   "total_participants": 84,  "ecocitizens_participants": None, "sponsor": None, "expense_lbp": None, "expense_usd": None, "bags": None, "notes": "Sainte Famille Rayak; date approximate"},
    {"era": "founding", "date": "2023-02-25", "type": "school_campaign", "location": "Byblos", "location_category": "school",   "total_participants": 17,  "ecocitizens_participants": 17,  "sponsor": None, "expense_lbp": None, "expense_usd": None, "bags": None, "notes": "Caritas group"},
    {"era": "founding", "date": "2023-02-25", "type": "school_campaign", "location": "Byblos", "location_category": "school",   "total_participants": 18,  "ecocitizens_participants": 18,  "sponsor": None, "expense_lbp": None, "expense_usd": None, "bags": None, "notes": "Parroisse group"},
    {"era": "founding", "date": "2023-02-25", "type": "school_campaign", "location": "Byblos", "location_category": "school",   "total_participants": 26,  "ecocitizens_participants": 26,  "sponsor": None, "expense_lbp": None, "expense_usd": None, "bags": None, "notes": "Scouts group"},
    {"era": "founding", "date": "2023-03-04", "type": "cleanup",         "location": "Tayouneh, Beirut",  "location_category": "urban",    "total_participants": 13,  "ecocitizens_participants": 7,   "sponsor": None, "expense_lbp": None, "expense_usd": None, "bags": 15,  "notes": "6 international"},
    {"era": "founding", "date": "2023-03-11", "type": "cleanup",         "location": "Byblos",            "location_category": "coastal",  "total_participants": 70,  "ecocitizens_participants": 61,  "sponsor": None, "expense_lbp": None, "expense_usd": None, "bags": 100, "notes": "9 international"},
    {"era": "founding", "date": "2023-04-14", "type": "cleanup",         "location": "Jounieh Beach",     "location_category": "coastal",  "total_participants": 33,  "ecocitizens_participants": 27,  "sponsor": None, "expense_lbp": None, "expense_usd": None, "bags": 50,  "notes": "6 Italian + 27 local"},
    {"era": "founding", "date": "2023-05-20", "type": "cleanup",         "location": "Zahleh",            "location_category": "urban",    "total_participants": 118, "ecocitizens_participants": 112, "sponsor": "Michel Daher Social Foundation", "expense_lbp": None, "expense_usd": None, "bags": 150, "notes": "Largest founding-era event"},
    {"era": "founding", "date": "2023-06-07", "type": "cleanup",         "location": "Adlieh, Beirut",    "location_category": "urban",    "total_participants": 20,  "ecocitizens_participants": 11,  "sponsor": None, "expense_lbp": None, "expense_usd": None, "bags": 25,  "notes": "9 Spanish group"},
    {"era": "founding", "date": "2023-06-10", "type": "cleanup",         "location": "Ain Kfar Zabad",    "location_category": "mountain", "total_participants": 41,  "ecocitizens_participants": 32,  "sponsor": "Ain Kfar Zabad Municipality", "expense_lbp": None, "expense_usd": None, "bags": 60,  "notes": "9 intl + 32 local"},
    {"era": "founding", "date": "2023-11-18", "type": "cleanup",         "location": "Jbeil - Citadelle", "location_category": "coastal",  "total_participants": 14,  "ecocitizens_participants": 14,  "sponsor": None, "expense_lbp": None, "expense_usd": None, "bags": 16,  "notes": "All local"},
    {"era": "founding", "date": "2024-03-23", "type": "cleanup",         "location": "Biel, Beirut",      "location_category": "urban",    "total_participants": 16,  "ecocitizens_participants": 16,  "sponsor": None, "expense_lbp": None, "expense_usd": None, "bags": 25,  "notes": ""},
    {"era": "founding", "date": "2024-06-22", "type": "hike",            "location": "Yahchouch",         "location_category": "mountain", "total_participants": 54,  "ecocitizens_participants": 47,  "sponsor": None, "expense_lbp": None, "expense_usd": None, "bags": 50,  "notes": "First hike"},

    # === 2025 ERA — alpha team formed, social media launched ===
    {"era": "2025", "date": "2025-02-14", "type": "hike", "location": "Wadi el Salib, Kfardebian", "location_category": "mountain", "total_participants": 26,  "ecocitizens_participants": 26,  "sponsor": None,             "expense_lbp": None, "expense_usd": None, "bags": 30,  "notes": "Mountain hike"},
    {"era": "2025", "date": "2025-06-14", "type": "hike", "location": "Rechmaya",                  "location_category": "mountain", "total_participants": 47,  "ecocitizens_participants": 47,  "sponsor": "Interact",       "expense_lbp": None, "expense_usd": None, "bags": 40,  "notes": "Partner: Interact"},
    # DQ FLAG: 100 participants / 30 bags = 0.30/person. Measured floor is 1.1/person. IMPOSSIBLE.
    # Bags excluded (set None). Participant count kept for growth modelling but ratio not derived.
    {"era": "2025", "date": "2025-07-11", "type": "hike", "location": "Baskinta",                  "location_category": "mountain", "total_participants": 100, "ecocitizens_participants": 100, "sponsor": "Interact; Ministry of Environment", "expense_lbp": None, "expense_usd": None, "bags": None, "notes": "DQ FLAG bags: 30 bags/100 people=0.30/person impossible; excluded from ratio calc"},
    {"era": "2025", "date": "2025-08-23", "type": "hike", "location": "Akoura",                    "location_category": "mountain", "total_participants": 48,  "ecocitizens_participants": 48,  "sponsor": "Live Love Lebanon","expense_lbp": None, "expense_usd": None, "bags": 125, "notes": "Partner: Live Love Lebanon"},

    # === 2026 — large multi-sponsor beach cleanups (EC as partial partner) ===
    {"era": "2026", "date": "2026-05-16", "type": "cleanup", "location": "Byblos Bahsa",       "location_category": "coastal", "total_participants": 100, "ecocitizens_participants": 8,  "sponsor": "SWIM; Byblos Municipality; Sanita; Lebtivity; Virgin; Aquafina; Taqa; Red Cross; Caritas; Scouts; US Embassy", "expense_lbp": None, "expense_usd": 0.0, "bags": 150, "notes": "Multi-org; EC as partner org"},
    {"era": "2026", "date": "2026-05-30", "type": "cleanup", "location": "Amchit Boulevard",   "location_category": "coastal", "total_participants": 150, "ecocitizens_participants": 15, "sponsor": "SWIM; Interact; Rotaract; Rotary; Sanita; Aquafina; Virgin; Lebtivity; Taqa; Scouts Marins", "expense_lbp": None, "expense_usd": 0.0, "bags": 200, "notes": "Multi-org; EC as partner org"},
]

df = pd.DataFrame(events)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)
df.to_excel("ecocitizens_master_events_clean.xlsx", index=False)

print("=== Events by era ===")
for era, grp in df.groupby("era"):
    types = grp["type"].value_counts().to_dict()
    total_p = int(grp["total_participants"].sum())
    total_b = grp["bags"].sum()
    print(f"  {era}: {len(grp)} events, {total_p} participants, {int(total_b) if not pd.isna(total_b) else '?'} bags | {types}")

print()
print("=== Bag ratio per event (valid bags only) ===")
valid = df[df["bags"].notna() & df["total_participants"].notna()].copy()
valid["ratio"] = valid["bags"] / valid["total_participants"]
for _, row in valid.iterrows():
    print(f"  {str(row['date'].date())}  {row['type']:15s}  {row['location'][:28]:28s}  {int(row['total_participants']):4d}p  {int(row['bags']):5d}bags  {row['ratio']:.2f}/person")

print()
print("=== Hike bag ratios (for simulation override) ===")
hikes = valid[valid["type"] == "hike"]
print(f"  N={len(hikes)} valid hike events")
ratios = hikes["ratio"].values
if len(ratios) >= 3:
    lo, base, hi = [float(x) for x in [ratios.min(), round(float(hikes['ratio'].median()),2), ratios.max()]]
    import numpy as np
    p10, p50, p90 = np.percentile(ratios, [10, 50, 90])
    print(f"  p10/p50/p90 = ({p10:.2f}, {p50:.2f}, {p90:.2f})  [replaces assumption (0.2, 0.6, 1.5)]")
