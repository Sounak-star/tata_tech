"""
==========================================================================
 Context Risk Score API (CLI + Importable Function)
==========================================================================
 PURPOSE
 -------
 Wrap the trained XGBoost Context Risk Model

 CLI USAGE
 ---------
   python context_risk_api.py                          # interactive mode
   python context_risk_api.py --machine Excavator --task Excavation
   python context_risk_api.py --machine Crane --task Loading/Unloading --weather Wind --time Night

 IMPORT USAGE
 ------------
   from step3_context_risk_api import get_context_risk

   result = get_context_risk(
       machine_type="Excavator",
       task_type="Excavation",
       time_of_day="Morning",
       weather_condition="Rain",
   )
   print(result)
   # {'risk_score': 34.2, 'risk_bucket': 'Medium', 'risk_color': '#FF9800'}
==========================================================================
"""

import os
import sys
import json
import argparse
import numpy as np
import joblib
from datetime import datetime
from xgboost import XGBRegressor

# ─── Paths (relative to this script) ──────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR    = os.path.join(BASE_DIR, "model")
MODEL_PATH   = os.path.join(MODEL_DIR, "context_risk_xgb.json")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoders.joblib")
CONFIG_PATH  = os.path.join(MODEL_DIR, "model_config.json")

# ─── Load Model Artefacts ─────────────────────────────────────────────
with open(CONFIG_PATH, "r") as f:
    MODEL_CONFIG = json.load(f)

FEATURE_COLS  = MODEL_CONFIG["feature_columns"]
LABEL_CLASSES = MODEL_CONFIG["label_classes"]

_model = XGBRegressor()
_model.load_model(MODEL_PATH)

_label_encoders = joblib.load(ENCODER_PATH)

# ─── Risk Bucket Definitions ──────────────────────────────────────────
RISK_BUCKETS = {
    "Low":      (0,  25, "#4CAF50"),   # green
    "Medium":   (25, 50, "#FF9800"),   # orange
    "High":     (50, 75, "#F44336"),   # red
    "Critical": (75, 100, "#9C27B0"),  # purple
}

def _get_bucket(score):
    if score < 25:  return "Low"
    elif score < 50: return "Medium"
    elif score < 75: return "High"
    else: return "Critical"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PUBLIC API — importable by Dev 1 / Integration team
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_context_risk(
    machine_type: str,
    task_type: str,
    time_of_day: str = "Unknown",
    weather_condition: str = "Clear/Unknown",
    month: int = None,
    day_of_week: str = None,
) -> dict:
    """
    Get the Context Risk Score for a given operational scenario.

    Parameters
    ----------
    machine_type : str
        Equipment type. Valid values:
        Aerial Lift, Backhoe, Bulldozer, Concrete Mixer, Concrete Pump,
        Crane, Dump Truck, Excavator, Forklift, Grader, Loader, Paver,
        Pile Driver, Roller, Trencher, Truck

    task_type : str
        Current task. Valid values:
        Demolition, Excavation, General Construction, Idling/Parked,
        Loading/Unloading, Maintenance, Mounting/Dismounting,
        Operating, Setup/Positioning

    time_of_day : str, optional
        Morning, Afternoon, Evening, Night, Unknown (default: Unknown)

    weather_condition : str, optional
        Clear/Unknown, Heat, Muddy, Rain, Snow/Ice, Wet Surface, Wind
        (default: Clear/Unknown)

    month : int, optional
        Month 1-12. Auto-detected from current date if omitted.

    day_of_week : str, optional
        Monday-Sunday. Auto-detected from current date if omitted.

    Returns
    -------
    dict with keys:
        risk_score  : float  (0-100)
        risk_bucket : str    (Low / Medium / High / Critical)
        risk_color  : str    (hex color code for UI badge)
        inputs      : dict   (echo of resolved inputs)
    """
    # Auto-fill date-based defaults
    now = datetime.now()
    if month is None:
        month = now.month
    if day_of_week is None:
        day_of_week = now.strftime("%A")

    inputs = {
        "machine_type": machine_type,
        "task_type": task_type,
        "time_of_day": time_of_day,
        "weather_condition": weather_condition,
        "month": month,
        "day_of_week": day_of_week,
    }

    # Encode features
    row = []
    for col in FEATURE_COLS:
        val = inputs[col]
        if col in _label_encoders:
            le = _label_encoders[col]
            if val in le.classes_:
                val = le.transform([val])[0]
            else:
                val = 0  # fallback for unseen categories
        row.append(val)

    # Predict
    pred = _model.predict(np.array([row]))[0]
    score = float(np.clip(round(pred, 1), 0, 100))
    bucket = _get_bucket(score)
    _, _, color = RISK_BUCKETS[bucket]

    return {
        "risk_score": score,
        "risk_bucket": bucket,
        "risk_color": color,
        "inputs": inputs,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLI Interface
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _print_options(label, options):
    """Pretty-print a numbered list of options."""
    print(f"\n  {label}:")
    for i, opt in enumerate(options, 1):
        print(f"    {i:>2}. {opt}")


def _pick(label, options, default=None):
    """Let user pick from a numbered list, with optional default."""
    _print_options(label, options)
    default_idx = None
    if default and default in options:
        default_idx = options.index(default) + 1

    prompt = f"  Enter number"
    if default_idx:
        prompt += f" [default: {default_idx}={default}]"
    prompt += ": "

    while True:
        raw = input(prompt).strip()
        if raw == "" and default_idx:
            return options[default_idx - 1]
        try:
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        except ValueError:
            # Also accept the exact string
            if raw in options:
                return raw
        print(f"    ⚠  Please enter a number 1-{len(options)}")


def _print_result(result):
    """Display the prediction result with visual formatting."""
    score  = result["risk_score"]
    bucket = result["risk_bucket"]
    inputs = result["inputs"]

    # Colored badge using ANSI codes
    badge_colors = {
        "Low":      "\033[92m",   # green
        "Medium":   "\033[93m",   # yellow
        "High":     "\033[91m",   # red
        "Critical": "\033[95m",   # magenta
    }
    reset = "\033[0m"
    color = badge_colors.get(bucket, "")

    print()
    print("  ┌──────────────────────────────────────────────────┐")
    print("  │           CONTEXT RISK ASSESSMENT                │")
    print("  ├──────────────────────────────────────────────────┤")
    print(f"  │  Machine:   {inputs['machine_type']:<36s}  │")
    print(f"  │  Task:      {inputs['task_type']:<36s}  │")
    print(f"  │  Time:      {inputs['time_of_day']:<36s}  │")
    print(f"  │  Weather:   {inputs['weather_condition']:<36s}  │")
    print(f"  │  Month:     {inputs['month']:<36}  │")
    print(f"  │  Day:       {inputs['day_of_week']:<36s}  │")
    print("  ├──────────────────────────────────────────────────┤")

    # Score bar
    bar_len = 30
    filled = int(score / 100 * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"  │  Score:  {color}{score:>5.1f}/100{reset}  [{bar}]  │")
    print(f"  │  Bucket: {color}{bucket:>8s}{reset}                                │")
    print("  └──────────────────────────────────────────────────┘")


def interactive_mode():
    """Run interactive prompt loop."""
    print()
    print("=" * 56)
    print("  Context Risk Model — Interactive Mode")
    print("=" * 56)

    machines = LABEL_CLASSES["machine_type"]
    tasks    = LABEL_CLASSES["task_type"]
    times    = LABEL_CLASSES["time_of_day"]
    weathers = LABEL_CLASSES["weather_condition"]
    days     = LABEL_CLASSES["day_of_week"]

    while True:
        machine = _pick("Machine Type", machines)
        task    = _pick("Task Type", tasks)
        time    = _pick("Time of Day", times, default="Unknown")
        weather = _pick("Weather Condition", weathers, default="Clear/Unknown")

        result = get_context_risk(
            machine_type=machine,
            task_type=task,
            time_of_day=time,
            weather_condition=weather,
        )
        _print_result(result)

        print()
        again = input("  Run another prediction? [Y/n]: ").strip().lower()
        if again in ("n", "no", "q", "quit", "exit"):
            break


def cli_mode():
    """Parse command-line args and run prediction."""
    parser = argparse.ArgumentParser(
        description="Context Risk Model — Get baseline risk score for construction scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python context_risk_api.py --machine Excavator --task Excavation
  python context_risk_api.py --machine Crane --task "Loading/Unloading" --weather Wind --time Night
  python context_risk_api.py --interactive
  python context_risk_api.py --demo
        """,
    )

    parser.add_argument("--machine", type=str, help="Machine type (e.g., Excavator, Crane, Truck)")
    parser.add_argument("--task", type=str, help="Task type (e.g., Operating, Excavation)")
    parser.add_argument("--time", type=str, default="Unknown", help="Time of day (default: Unknown)")
    parser.add_argument("--weather", type=str, default="Clear/Unknown", help="Weather condition (default: Clear/Unknown)")
    parser.add_argument("--month", type=int, default=None, help="Month 1-12 (default: current)")
    parser.add_argument("--day", type=str, default=None, help="Day of week (default: current)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive mode")
    parser.add_argument("--demo", action="store_true", help="Run demo with predefined scenarios")
    parser.add_argument("--json", action="store_true", help="Output as JSON instead of formatted display")

    args = parser.parse_args()

    # Interactive mode
    if args.interactive or (args.machine is None and not args.demo):
        interactive_mode()
        return

    # Demo mode
    if args.demo:
        run_demo()
        return

    # Single prediction from CLI args
    result = get_context_risk(
        machine_type=args.machine,
        task_type=args.task,
        time_of_day=args.time,
        weather_condition=args.weather,
        month=args.month,
        day_of_week=args.day,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_result(result)


def run_demo():
    """Run predefined demo scenarios to showcase the model."""
    print()
    print("=" * 56)
    print("  🧪  Context Risk Model — Demo Scenarios")
    print("=" * 56)

    demos = [
        ("Excavator", "Excavation",          "Morning",   "Clear/Unknown"),
        ("Crane",     "Loading/Unloading",   "Afternoon", "Wind"),
        ("Bulldozer", "Operating",           "Night",     "Rain"),
        ("Forklift",  "Loading/Unloading",   "Morning",   "Clear/Unknown"),
        ("Aerial Lift", "Maintenance",       "Afternoon", "Snow/Ice"),
        ("Truck",     "General Construction", "Evening",  "Muddy"),
        ("Loader",    "Operating",           "Night",     "Snow/Ice"),
        ("Roller",    "Operating",           "Afternoon", "Heat"),
    ]

    for machine, task, time, weather in demos:
        result = get_context_risk(
            machine_type=machine,
            task_type=task,
            time_of_day=time,
            weather_condition=weather,
        )
        _print_result(result)
    print()


# ─── Entry Point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    cli_mode()
