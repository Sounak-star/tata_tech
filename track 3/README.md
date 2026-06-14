# Pipeline 2 - Track 3: Context Risk Model

This module handles **Track 3: Context Risk Model** for the Edge AI Operator Safety & Human-Machine Interaction system. It generates a **Context Risk Score (0-100)** to set the baseline danger level based on the operational context (machine type, task, time, weather).

> **Note:** This model does *not* predict live crashes. It provides a baseline risk assessment using historical OSHA incident data to enhance situational awareness.

---

Tasks Completed

### 1: OSHA Data Wrangling (`osha_data_wrangling.py`)
- **Downloaded & Cleaned:** Processed 104,548 raw OSHA Severe Injury Reports (2015-2025).
- **Filtered:** Isolated 4,287 construction-vehicle and heavy-equipment incidents.
- **Feature Extraction:** Extracted critical context variables using Regex/NLP on narratives and structured fields:
  - `machine_type` (e.g., Excavator, Crane, Forklift)
  - `hazard_type` (e.g., Struck-by, Fall, Caught-in)
  - `task_type` (e.g., Operating, Maintenance, Loading/Unloading)
  - `time_of_day` (Morning, Afternoon, Evening, Night)
  - `weather_condition` (Clear, Rain, Snow/Ice, Wind, Heat, Muddy)
- **Hazard Frequencies:** Computed specific hazard frequencies (e.g., "Excavators have a 60.7% Struck-by frequency").

### 2: The XGBoost Context Model (`xgboost_context_risk.py`)
- **Target Engineering:** Since OSHA data doesn't have a direct "risk score", an engineered composite target (0-100) was built combining:
  - Incident Severity (Hospitalizations, Amputations, Loss of Eye)
  - Hazard Lethality weights
  - Context Frequency
  - Weather & Time multipliers
- **Model Training:** Trained a lightweight **XGBoost Regressor** on the 4,287 incidents.
- **Evaluation & Visuals:** Generated performance metrics and plots for feature importance, residuals, and risk distributions.

### 3: API / CLI (`context_risk_api.py`)
- **Integration Function:** Built a simple, importable Python function `get_context_risk()` that can be called to get the score, bucket, and UI color badge.
- **CLI Interface:** Added an interactive and command-line interface to easily test scenarios without writing code.

---

## Model Performance Analysis

The model effectively learns to rank context risk based on historical severity and frequency.

- **R² Score:** 0.31 (Test Set)
- **Bucket Accuracy:** 60.0% (Correctly classifies the scenario into Low, Medium, High, Critical)
- **Tolerance Bands:** Predicts within ±10 points of the engineered score 60% of the time, and within ±15 points 74.6% of the time.

### Why R² = 0.31 is successful for this use-case:
The model is not a live sensor crash predictor. It is a **static context baseline**. Identical contexts in history (e.g., Excavator operating in the morning) resulted in varying severities (from minor bruises to amputations). The XGBoost model successfully averages this variance to provide a generalized, robust **baseline risk lookup** that handles unseen combinations gracefully.

### Feature Importance
1. **Machine Type (46%)** - *What you are operating* is the biggest risk factor.
2. **Task Type (19%)** - *What you are doing with it* is the second biggest.
3. **Month / Weather / Time** - Environmental and seasonal factors modulate the baseline.

*(See the `plots/` directory for detailed visualizations of actual vs predicted scores, feature importance, and average risk by machine type).*

---

## How to Use the Model (Step 3)

The script `context_risk_api.py` relies on the trained XGBoost model artifacts in the `model/` folder.


import the `get_context_risk` function into your own Python scripts:

```python
from context_risk_api import get_context_risk

result = get_context_risk(
    machine_type="Excavator",
    task_type="Excavation",
    time_of_day="Morning",
    weather_condition="Rain"
)

print(result)
```

**Output:**
```json
{
  "risk_score": 34.2,
  "risk_bucket": "Medium",
  "risk_color": "#FF9800",
  "inputs": {
    "machine_type": "Excavator",
    "task_type": "Excavation",
    "time_of_day": "Morning",
    "weather_condition": "Rain",
    "month": 6,
    "day_of_week": "Sunday"
  }
}
```
*(The `risk_color` hex code maps to the UI dashboard badge color).*

### 2. Command Line Interface (Testing & Demo)

**Interactive Mode:**
```bash
python context_risk_api.py --interactive
```

**Direct CLI Arguments:**
```bash
python context_risk_api.py --machine Crane --task "Loading/Unloading" --weather Wind --time Night
```

**JSON Output for CLI Pipeline:**
```bash
python context_risk_api.py --machine Forklift --task "Loading/Unloading" --json
```

**Demo Mode (Pre-defined scenarios):**
```bash
python context_risk_api.py --demo
```

---

## Directory Structure

```
.
├── cleaned_data/
│   ├── osha_construction_vehicles_cleaned.csv  # The cleaned dataset
│   ├── hazard_frequency_for_dev1.csv           # Stats for Simulator (Dev 1)
│   └── wrangling_summary.txt                   # Wrangling report
├── model/
│   ├── context_risk_xgb.json                   # Trained XGBoost weights
│   ├── label_encoders.joblib                   # Feature encoders
│   └── model_config.json                       # Valid inputs and mappings
├── plots/                                      # Performance visualisations
├── step1_osha_data_wrangling.py                # Wrangling Script
├── step2_xgboost_context_risk.py               # Model Training Script
└── step3_context_risk_api.py                   # API / CLI Wrapper
```