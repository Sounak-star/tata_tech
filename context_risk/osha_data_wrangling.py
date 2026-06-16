"""
==========================================================================
 OSHA Data Wrangling for Context Risk Model
==========================================================================
 PURPOSE
 -------
 1. Load the raw OSHA Severe Injury Reports CSV.
 2. Filter to construction-vehicle / heavy-equipment incidents.
 3. Extract the context variables needed for the XGBoost model:
       • machine_type   – standardised equipment category
       • hazard_type    – what kind of incident (struck-by, caught-in, …)
       • task_type      – operating vs. maintenance vs. loading …
       • time_of_day    – Morning / Afternoon / Evening / Night / Unknown
       • weather_cond   – Clear / Rain / Snow-Ice / Wind / Heat / Unknown
       • severity       – severity proxy (hospitalized, amputation, eye loss)
 4. Compute hazard-frequency percentages.
 5. Save cleaned CSV → ready for XGBoost training.
==========================================================================
"""

import pandas as pd
import numpy as np
import re
import os
from pathlib import Path

# ─── Config ────────────────────────────────────────────────────────────
RAW_CSV = "January2015toSeptember2025.csv"
OUTPUT_DIR = "cleaned_data"
CLEANED_CSV = os.path.join(OUTPUT_DIR, "osha_construction_vehicles_cleaned.csv")
HAZARD_FREQ_CSV = os.path.join(OUTPUT_DIR, "hazard_frequency.csv")
SUMMARY_TXT = os.path.join(OUTPUT_DIR, "wrangling_summary.txt")

# NAICS codes starting with 23 = Construction sector
CONSTRUCTION_NAICS_PREFIX = "23"

# ─── 1. Load Data ──────────────────────────────────────────────────────
print("Loading raw OSHA data …")
df = pd.read_csv(RAW_CSV, low_memory=False)
print(f"   Total records: {len(df):,}")

# ─── 2. Filter to Construction Sector ──────────────────────────────────
df["NAICS_str"] = df["Primary NAICS"].astype(str).str.strip()
construction = df[df["NAICS_str"].str.startswith(CONSTRUCTION_NAICS_PREFIX)].copy()
print(f"   Construction sector (NAICS 23*): {len(construction):,}")

# ─── 3. Identify Vehicle / Heavy-Equipment Incidents ───────────────────
# We look at SourceTitle, Secondary Source Title, EventTitle, AND Final Narrative
# to maximise recall — many vehicle incidents are only mentioned in the narrative.

VEHICLE_PATTERNS = {
    "Excavator":    r"\bexcavator\b",
    "Backhoe":      r"\bbackhoe\b",
    "Bulldozer":    r"\b(?:bulldozer|dozer)\b",
    "Loader":       r"\b(?:front[- ]?end[- ]?loader|wheel[- ]?loader|skid[- ]?steer|bobcat|loader)\b",
    "Crane":        r"\bcrane\b",
    "Forklift":     r"\b(?:forklift|fork[- ]?lift|telehandler|order picker)\b",
    "Dump Truck":   r"\bdump\s*truck\b",
    "Truck":        r"\b(?:truck|semi|tractor[- ]?trailer|tanker)\b",
    "Aerial Lift":  r"\b(?:aerial[- ]?lift|boom[- ]?lift|scissor[- ]?lift|cherry[- ]?picker|manlift|man[- ]?lift|jlg|genie)\b",
    "Roller":       r"\b(?:roller|compactor)\b",
    "Grader":       r"\bgrader\b",
    "Trencher":     r"\btrencher\b",
    "Concrete Mixer": r"\b(?:concrete[- ]?mixer|concrete[- ]?truck|cement[- ]?mixer)\b",
    "Concrete Pump":  r"\b(?:concrete[- ]?pump|boom[- ]?pump)\b",
    "Paver":        r"\bpaver\b",
    "Pile Driver":  r"\bpile[- ]?driver\b",
}

def classify_machine(row):
    """Return the first matching machine type from multiple text columns."""
    texts = " ".join([
        str(row.get("SourceTitle", "")),
        str(row.get("Secondary Source Title", "")),
        str(row.get("EventTitle", "")),
        str(row.get("Final Narrative", "")),
    ]).lower()

    for machine, pattern in VEHICLE_PATTERNS.items():
        if re.search(pattern, texts):
            return machine
    return None

print("Classifying machine types …")
construction["machine_type"] = construction.apply(classify_machine, axis=1)

# Keep only rows where a vehicle/machine was identified
vehicles = construction[construction["machine_type"].notna()].copy()
print(f"   Vehicle/equipment incidents found: {len(vehicles):,}")
print(f"   Machine type distribution:")
print(vehicles["machine_type"].value_counts().to_string(header=False))
print()

# ─── 4. Extract Hazard Type (from EventTitle) ──────────────────────────
HAZARD_MAP = {
    "Struck-by":        r"struck\s*by",
    "Caught-in":        r"caught\s*in|compressed|pinched|crushed",
    "Back-over":        r"back.?over|backed?\s*(?:over|into|up)",
    "Tip-over":         r"tip.?over|tipped?\s*over|overturn|rollover|roll.?over",
    "Fall":             r"fall|fell",
    "Electrocution":    r"electr|shock|power\s*line",
    "Collision":        r"collis|crash|impact|rear[- ]?end",
    "Run-over":         r"run.?over|ran\s*over",
    "Pinch-Point":      r"pinch|caught\s*between",
    "Contact-with":     r"contact\s*with|struck\s*against",
    "Burns":            r"burn|fire|ignit|explos",
}

def classify_hazard(row):
    """Map EventTitle + Narrative to a hazard category."""
    texts = " ".join([
        str(row.get("EventTitle", "")),
        str(row.get("Final Narrative", "")),
    ]).lower()

    for hazard, pattern in HAZARD_MAP.items():
        if re.search(pattern, texts):
            return hazard
    return "Other"

print("Classifying hazard types …")
vehicles["hazard_type"] = vehicles.apply(classify_hazard, axis=1)

# ─── 5. Extract Task Type (from EventTitle + Narrative) ────────────────
TASK_MAP = {
    "Operating":        r"\boperat\b|\bdriv\b|\bmoving\b|\btravel\b|\bback\s?up\b|\brevers\b",
    "Maintenance":      r"\bmaintenan\b|\brepair\b|\bclean\b|\bservic\b|\binspect\b|\boil\b|\bgrease\b",
    "Loading/Unloading": r"\bload\b|\bunload\b|\blifting\b|\blower\b|\brais\b|\bhoisting\b|\brigg\b",
    "Excavation":       r"\bexcavat\b|\bdig\b|\btrench\b|\bgrade\b|\bgrade\b",
    "Demolition":       r"\bdemol\b|\bwreck\b|\btear\s*down\b",
    "Mounting/Dismounting": r"\benter\b|\bexit\b|\bclimb\b|\bstep\b|\bdescend\b|\bmount\b|\bdismount\b|\bgetting\s*(in|out|on|off)\b",
    "Setup/Positioning": r"\bsetup\b|\bset\s*up\b|\bposition\b|\bstabil\b|\boutrigger\b|\bassembl\b",
    "Idling/Parked":    r"\bidle\b|\bpark\b|\bstation\b",
}

def classify_task(row):
    texts = " ".join([
        str(row.get("EventTitle", "")),
        str(row.get("Final Narrative", "")),
    ]).lower()

    for task, pattern in TASK_MAP.items():
        if re.search(pattern, texts):
            return task
    return "General Construction"

print("Classifying task types …")
vehicles["task_type"] = vehicles.apply(classify_task, axis=1)

# ─── 6. Extract Time of Day ───────────────────────────────────────────
# EventDate has date only (no time). We mine the narrative for time cues.
TIME_PATTERNS = {
    "Night":     r"\bnight\b|\b(?:1[0-1]|12)\s*(?:p\.?m\.?|pm)\b|\bmidnight\b|\b[2-5]\s*(?:a\.?m\.?|am)\b|\bafter\s*dark\b|\bpredawn\b",
    "Morning":   r"\bmorning\b|\b(?:[5-9]|10|11)\s*(?:a\.?m\.?|am)\b|\bdaybreak\b|\bdawn\b|\bsunrise\b",
    "Afternoon": r"\bafternoon\b|\b(?:12|[1-4])\s*(?:p\.?m\.?|pm)\b|\bmidday\b|\bnoon\b|\blunch\b",
    "Evening":   r"\bevening\b|\b(?:[5-9])\s*(?:p\.?m\.?|pm)\b|\bdusk\b|\bsunset\b|\btwilight\b|\bend\s*of\s*(?:the\s*)?day\b",
}

def extract_time_of_day(narrative):
    text = str(narrative).lower()
    for period, pattern in TIME_PATTERNS.items():
        if re.search(pattern, text):
            return period
    return "Unknown"

print("Extracting time of day …")
vehicles["time_of_day"] = vehicles["Final Narrative"].apply(extract_time_of_day)

# Also use EventDate to derive month (seasonality proxy) and day-of-week
vehicles["EventDate_parsed"] = pd.to_datetime(vehicles["EventDate"], format="mixed", errors="coerce")
vehicles["month"] = vehicles["EventDate_parsed"].dt.month
vehicles["day_of_week"] = vehicles["EventDate_parsed"].dt.day_name()
vehicles["year"] = vehicles["EventDate_parsed"].dt.year

# ─── 7. Extract Weather / Conditions ──────────────────────────────────
# Sources: Secondary Source Title (structured) + narrative (NLP)
WEATHER_STRUCTURED = {
    "Snow/Ice":  r"ice|sleet|snow|hail|freez",
    "Wind":      r"wind|gust|turbulence",
    "Rain":      r"\brain\b",
    "Heat":      r"\bheat\b.*environmental|heat\s*stroke|heat\s*exhaust",
}

WEATHER_NARRATIVE = {
    "Rain":      r"\brain\b|\brainy\b|\bwet\s*(?:weather|conditions?)\b|\bstorm\b|\bdownpour\b",
    "Snow/Ice":  r"\bsnow\b|\bicy\b|\bice\b|\bfrost\b|\bfreez\b|\bsleet\b|\bblack\s*ice\b",
    "Wind":      r"\bwind\b|\bwindy\b|\bgust\b|\bblow\b.*\bover\b",
    "Heat":      r"\bheat\b|\bhot\s*(?:weather|day|conditions?|temperature)\b|\bheat\s*(?:stroke|exhaust|stress)\b",
    "Fog":       r"\bfog\b|\bfoggy\b|\blow\s*visib\b|\bmist\b",
    "Muddy":     r"\bmud\b|\bmuddy\b|\bsoft\s*ground\b",
    "Wet Surface": r"\bwet\b|\bslippery\b|\bpuddle\b",
}

def extract_weather(row):
    # First check structured fields
    sec_src = str(row.get("Secondary Source Title", "")).lower()
    for condition, pattern in WEATHER_STRUCTURED.items():
        if re.search(pattern, sec_src):
            return condition

    # Then check narrative
    narrative = str(row.get("Final Narrative", "")).lower()
    for condition, pattern in WEATHER_NARRATIVE.items():
        if re.search(pattern, narrative):
            return condition

    return "Clear/Unknown"

print("Extracting weather conditions …")
vehicles["weather_condition"] = vehicles.apply(extract_weather, axis=1)

# ─── 8. Compute Severity Score ─────────────────────────────────────────
# Simple severity proxy: Hospitalized=1, Amputation=3, Eye Loss=2
vehicles["Hospitalized"] = pd.to_numeric(vehicles["Hospitalized"], errors="coerce").fillna(0)
vehicles["Amputation"]   = pd.to_numeric(vehicles["Amputation"], errors="coerce").fillna(0)
vehicles["Loss of Eye"]  = pd.to_numeric(vehicles["Loss of Eye"], errors="coerce").fillna(0)

vehicles["severity_score"] = (
    vehicles["Hospitalized"] * 1 +
    vehicles["Amputation"]   * 3 +
    vehicles["Loss of Eye"]  * 2
).clip(upper=5)

# ─── 9. Select Final Columns ──────────────────────────────────────────
KEEP_COLS = [
    "ID",
    "EventDate",
    "year",
    "month",
    "day_of_week",
    "State",
    "Primary NAICS",
    "machine_type",
    "hazard_type",
    "task_type",
    "time_of_day",
    "weather_condition",
    "severity_score",
    "Hospitalized",
    "Amputation",
    "Loss of Eye",
    "EventTitle",
    "SourceTitle",
    "Secondary Source Title",
    "NatureTitle",
    "Part of Body Title",
    "Final Narrative",
]

cleaned = vehicles[KEEP_COLS].copy()

# ─── 10. Hazard Frequency Table (for Dev 1) ───────────────────────────
print()
print("📊 Computing hazard frequencies …")

# Overall hazard distribution
hazard_freq = (
    cleaned["hazard_type"]
    .value_counts(normalize=True)
    .reset_index()
)
hazard_freq.columns = ["hazard_type", "frequency"]
hazard_freq["percentage"] = (hazard_freq["frequency"] * 100).round(2)

# Per-machine hazard distribution
machine_hazard = (
    cleaned.groupby(["machine_type", "hazard_type"])
    .size()
    .reset_index(name="count")
)
machine_totals = machine_hazard.groupby("machine_type")["count"].transform("sum")
machine_hazard["percentage"] = (machine_hazard["count"] / machine_totals * 100).round(2)

print("\n── Overall Hazard Frequencies ──")
print(hazard_freq.to_string(index=False))

print("\n── Top hazard per machine type ──")
top_per_machine = machine_hazard.sort_values("percentage", ascending=False).groupby("machine_type").head(3)
for machine in sorted(top_per_machine["machine_type"].unique()):
    subset = top_per_machine[top_per_machine["machine_type"] == machine]
    print(f"\n  {machine}:")
    for _, r in subset.iterrows():
        print(f"    {r['hazard_type']:20s} {r['percentage']:6.1f}%  (n={r['count']})")

# ─── 11. Save Outputs ─────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

cleaned.to_csv(CLEANED_CSV, index=False)
print(f"\nCleaned dataset saved → {CLEANED_CSV}  ({len(cleaned):,} rows)")

hazard_freq.to_csv(HAZARD_FREQ_CSV, index=False)
machine_hazard.to_csv(
    os.path.join(OUTPUT_DIR, "hazard_frequency_by_machine.csv"), index=False
)
print(f"Hazard frequencies saved → {HAZARD_FREQ_CSV}")

# ─── 12. Summary Report ───────────────────────────────────────────────
summary_lines = [
    "=" * 60,
    " OSHA Data Wrangling Summary",
    "=" * 60,
    f"Raw records:              {len(df):>8,}",
    f"Construction (NAICS 23*): {len(construction):>8,}",
    f"Vehicle/equipment incidents: {len(vehicles):>7,}",
    "",
    "── Machine Type Distribution ──",
    cleaned["machine_type"].value_counts().to_string(),
    "",
    "── Hazard Type Distribution ──",
    cleaned["hazard_type"].value_counts().to_string(),
    "",
    "── Task Type Distribution ──",
    cleaned["task_type"].value_counts().to_string(),
    "",
    "── Time of Day Distribution ──",
    cleaned["time_of_day"].value_counts().to_string(),
    "",
    "── Weather Condition Distribution ──",
    cleaned["weather_condition"].value_counts().to_string(),
    "",
    "── Severity Score Stats ──",
    cleaned["severity_score"].describe().to_string(),
    "",
    f"Date range: {cleaned['EventDate'].min()} to {cleaned['EventDate'].max()}",
    f"States represented: {cleaned['State'].nunique()}",
    "",
    "Files produced:",
    f"  • {CLEANED_CSV}",
    f"  • {HAZARD_FREQ_CSV}",
    f"  • {os.path.join(OUTPUT_DIR, 'hazard_frequency_by_machine.csv')}",
]

summary = "\n".join(summary_lines)
with open(SUMMARY_TXT, "w") as f:
    f.write(summary)
print(f"Summary report → {SUMMARY_TXT}")
print()
print(summary)
