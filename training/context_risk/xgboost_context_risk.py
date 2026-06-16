"""
==========================================================================
 XGBoost Context Risk Model
==========================================================================
 PURPOSE
 -------
 Trains a lightweight XGBoost model on the cleaned OSHA data.
   Input:  machine_type, task_type, time_of_day, weather_condition,
           month, day_of_week
   Output: Context Risk Score (0–100)

 This does NOT predict live crashes. It sets a baseline danger level

 TARGET ENGINEERING
 ------------------
 The raw dataset doesn't have a "risk score". We engineer one by
 combining three historical signals:
   1. Severity factor     – how bad injuries tend to be for this combo
   2. Hazard lethality    – weight by hazard type dangerousness
   3. Incident frequency  – how common this scenario is historically
 The composite is then min-max scaled to 0–100.
==========================================================================
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
)
from xgboost import XGBRegressor

# ─── Config ────────────────────────────────────────────────────────────
CLEANED_CSV = "cleaned_data/osha_construction_vehicles_cleaned.csv"
MODEL_DIR   = "model"
PLOTS_DIR   = "plots"

FEATURE_COLS = [
    "machine_type",
    "task_type",
    "time_of_day",
    "weather_condition",
    "month",
    "day_of_week",
]

RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1. Load Cleaned Data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("Loading cleaned data …")
df = pd.read_csv(CLEANED_CSV)
print(f"   Records: {len(df):,}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  2. Engineer the Target → Context Risk Score (0–100)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("Engineering Context Risk Score target …")

# --- Component A: Severity factor (per-row, already computed) ---------
#     severity_score ranges 1-5. Normalize to 0-1.
df["sev_norm"] = (df["severity_score"] - 1) / 4

# --- Component B: Hazard lethality weights ----------------------------
#     Expert-informed weights reflecting how dangerous each hazard type is
#     on construction sites (loosely inspired by OSHA "Fatal Four" + data).
HAZARD_LETHALITY = {
    "Struck-by":     0.85,
    "Tip-over":      0.90,
    "Back-over":     0.92,
    "Run-over":      0.95,
    "Electrocution": 0.88,
    "Caught-in":     0.75,
    "Fall":          0.70,
    "Collision":     0.80,
    "Burns":         0.72,
    "Pinch-Point":   0.55,
    "Contact-with":  0.50,
    "Other":         0.45,
}
df["hazard_lethality"] = df["hazard_type"].map(HAZARD_LETHALITY).fillna(0.45)

# --- Component C: Incident frequency per context combination ----------
#     How often does this specific (machine, task, hazard) combo appear?
#     More frequent = more probable risk scenario.
combo_key = df.groupby(["machine_type", "task_type", "hazard_type"]).size()
combo_key = combo_key / combo_key.max()  # normalize to 0-1
combo_key.name = "freq_norm"
df = df.merge(
    combo_key.reset_index(),
    on=["machine_type", "task_type", "hazard_type"],
    how="left",
)
df["freq_norm"] = df["freq_norm"].fillna(0.01)

# --- Component D: Machine-type base risk ------------------------------
#     Historical severity-weighted incident rate per machine type.
machine_risk = (
    df.groupby("machine_type")["severity_score"]
    .agg(["mean", "count"])
)
machine_risk["base_risk"] = (
    machine_risk["mean"] / machine_risk["mean"].max() * 0.5 +
    machine_risk["count"] / machine_risk["count"].max() * 0.5
)
machine_risk_map = machine_risk["base_risk"].to_dict()
df["machine_base_risk"] = df["machine_type"].map(machine_risk_map).fillna(0.3)

# --- Component E: Weather risk modifier --------------------------------
WEATHER_RISK = {
    "Clear/Unknown": 0.0,
    "Rain":          0.50,
    "Snow/Ice":      1.20,
    "Wind":          0.80,
    "Heat":          0.40,
    "Fog":           1.00,
    "Muddy":         0.50,
    "Wet Surface":   0.40,
}
df["weather_risk"] = df["weather_condition"].map(WEATHER_RISK).fillna(0.0)

# --- Component F: Time-of-day risk modifier ----------------------------
TIME_RISK = {
    "Morning":   0.00,
    "Afternoon": 0.10,
    "Evening":   0.60,
    "Night":     1.20,
    "Unknown":   0.10,
}
df["time_risk"] = df["time_of_day"].map(TIME_RISK).fillna(0.10)

# --- Composite Risk Score (0-100) -------------------------------------
#     Base operational score mapped to roughly 0-60 points for a typical scenario
base_op_score = (
    40.0 * df["sev_norm"] +
    35.0 * df["hazard_lethality"] +
    15.0 * df["machine_base_risk"] +
    10.0 * df["freq_norm"]
)

#     Environment acts as a direct multiplier (1.0 = normal, >1.0 = bad conditions)
env_multiplier = 1.0 + df["weather_risk"] + df["time_risk"]

#     Final score clipped to 100. This absolute mapping prevents rare 
#     extreme outliers from squashing the rest of the distribution.
df["context_risk_score"] = (base_op_score * env_multiplier).clip(upper=100.0).round(2)

print(f"   Risk Score stats:")
print(f"     Mean:   {df['context_risk_score'].mean():.1f}")
print(f"     Median: {df['context_risk_score'].median():.1f}")
print(f"     Std:    {df['context_risk_score'].std():.1f}")
print(f"     Min:    {df['context_risk_score'].min():.1f}")
print(f"     Max:    {df['context_risk_score'].max():.1f}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3. Encode Categorical Features
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("Encoding categorical features …")

label_encoders = {}
df_model = df.copy()

for col in FEATURE_COLS:
    if df_model[col].dtype == object or df_model[col].dtype.name == "str":
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        label_encoders[col] = le
        print(f"   {col}: {len(le.classes_)} classes → {list(le.classes_)}")

X = df_model[FEATURE_COLS].values
y = df_model["context_risk_score"].values

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  4. Train / Test Split
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("Splitting data (80/20) …")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)
print(f"   Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  5. Train XGBoost Regressor
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("Training XGBoost model …")

# GPU auto-detection (dev brief: training MUST use CUDA when available).
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
try:
    from gpu_utils import xgb_device, describe
    _DEVICE = xgb_device()
    print(f"   compute device: {describe()}")
except Exception:
    _DEVICE = "cpu"

model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.5,
    min_child_weight=3,
    random_state=RANDOM_STATE,
    verbosity=0,
    n_jobs=-1,
    tree_method="hist",
    device=_DEVICE,          # 'cuda' on an NVIDIA GPU, else 'cpu'
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  6. Evaluate Model Performance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 60)
print(" MODEL PERFORMANCE EVALUATION")
print("=" * 60)

y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)

# --- Regression Metrics -----------------------------------------------
metrics = {
    "Train": {
        "MAE":  mean_absolute_error(y_train, y_pred_train),
        "RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "R²":   r2_score(y_train, y_pred_train),
        "Explained Variance": explained_variance_score(y_train, y_pred_train),
    },
    "Test": {
        "MAE":  mean_absolute_error(y_test, y_pred_test),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "R²":   r2_score(y_test, y_pred_test),
        "Explained Variance": explained_variance_score(y_test, y_pred_test),
    },
}

for split, m in metrics.items():
    print(f"\n── {split} Set ──")
    for name, val in m.items():
        print(f"   {name:25s}: {val:.4f}")

# --- Cross-Validation ------------------------------------------------
print("\n── 5-Fold Cross-Validation (on full data) ──")
cv_r2   = cross_val_score(model, X, y, cv=5, scoring="r2")
cv_mae  = -cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
cv_rmse = np.sqrt(-cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error"))

print(f"   R²   : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
print(f"   MAE  : {cv_mae.mean():.4f} ± {cv_mae.std():.4f}")
print(f"   RMSE : {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")

# --- Accuracy-like metric: % of predictions within ±N points ----------
print("\n── Prediction Accuracy (tolerance bands) ──")
errors = np.abs(y_test - y_pred_test)
for tol in [2, 5, 10, 15]:
    pct = (errors <= tol).mean() * 100
    print(f"   Within ±{tol:>2d} points: {pct:6.1f}%")

# --- Risk Bucket Classification Accuracy ------------------------------
#     Bin into Low/Medium/High/Critical and check classification accuracy
def risk_bucket(score):
    if score < 25:
        return "Low"
    elif score < 50:
        return "Medium"
    elif score < 75:
        return "High"
    else:
        return "Critical"

buckets_true = np.array([risk_bucket(s) for s in y_test])
buckets_pred = np.array([risk_bucket(s) for s in y_pred_test])
bucket_acc = (buckets_true == buckets_pred).mean() * 100

print(f"\n── Risk Bucket Classification ──")
print(f"   Buckets: Low (0-25) | Medium (25-50) | High (50-75) | Critical (75-100)")
print(f"   Bucket Accuracy: {bucket_acc:.1f}%")

from sklearn.metrics import classification_report, confusion_matrix
bucket_labels = ["Low", "Medium", "High", "Critical"]
# Only report labels that actually appear in the data
present_labels = sorted(set(buckets_true) | set(buckets_pred),
                        key=lambda x: bucket_labels.index(x))
print(f"\n   Classification Report (by bucket):")
print(classification_report(buckets_true, buckets_pred, labels=present_labels, zero_division=0))

print("=" * 60)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  7. Generate Performance Plots
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n Generating performance plots …")

sns.set_theme(style="whitegrid", font_scale=1.1)

# --- Plot 1: Actual vs Predicted scatter ------------------------------
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test, y_pred_test, alpha=0.4, s=15, c="#2196F3", edgecolors="none")
ax.plot([0, 100], [0, 100], "--", color="#F44336", linewidth=2, label="Perfect Prediction")
ax.set_xlabel("Actual Risk Score")
ax.set_ylabel("Predicted Risk Score")
ax.set_title("Context Risk Score: Actual vs Predicted")
ax.legend()
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "actual_vs_predicted.png"), dpi=150)
plt.close()

# --- Plot 2: Residual distribution ------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
residuals = y_test - y_pred_test
ax.hist(residuals, bins=50, color="#4CAF50", edgecolor="white", alpha=0.8)
ax.axvline(0, color="#F44336", linestyle="--", linewidth=2)
ax.set_xlabel("Residual (Actual − Predicted)")
ax.set_ylabel("Count")
ax.set_title(f"Residual Distribution (Mean={residuals.mean():.2f}, Std={residuals.std():.2f})")
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "residual_distribution.png"), dpi=150)
plt.close()

# --- Plot 3: Feature Importance (top features) ------------------------
fig, ax = plt.subplots(figsize=(8, 5))
importances = model.feature_importances_
feature_imp = pd.DataFrame({
    "Feature": FEATURE_COLS,
    "Importance": importances,
}).sort_values("Importance", ascending=True)

colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_imp)))
ax.barh(feature_imp["Feature"], feature_imp["Importance"], color=colors)
ax.set_xlabel("Feature Importance (Gain)")
ax.set_title("XGBoost Feature Importance")
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=150)
plt.close()

# --- Plot 4: Risk Score Distribution -----------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df["context_risk_score"], bins=50, color="#FF9800", edgecolor="white", alpha=0.8)
ax.axvline(25, color="green", linestyle="--", linewidth=1.5, label="Low / Medium")
ax.axvline(50, color="orange", linestyle="--", linewidth=1.5, label="Medium / High")
ax.axvline(75, color="red", linestyle="--", linewidth=1.5, label="High / Critical")
ax.set_xlabel("Context Risk Score")
ax.set_ylabel("Count")
ax.set_title("Distribution of Engineered Context Risk Scores")
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "risk_score_distribution.png"), dpi=150)
plt.close()

# --- Plot 5: Confusion Matrix for Risk Buckets ------------------------
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(buckets_true, buckets_pred, labels=present_labels)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=present_labels, yticklabels=present_labels, ax=ax)
ax.set_xlabel("Predicted Bucket")
ax.set_ylabel("Actual Bucket")
ax.set_title("Risk Bucket Confusion Matrix")
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "confusion_matrix_buckets.png"), dpi=150)
plt.close()

# --- Plot 6: Mean Risk by Machine Type --------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
machine_risk_plot = (
    df.groupby("machine_type")["context_risk_score"]
    .agg(["mean", "std", "count"])
    .sort_values("mean", ascending=True)
)
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(machine_risk_plot)))
bars = ax.barh(machine_risk_plot.index, machine_risk_plot["mean"], color=colors,
               xerr=machine_risk_plot["std"], capsize=3)
ax.set_xlabel("Mean Context Risk Score")
ax.set_title("Average Risk Score by Machine Type")
# Add count annotations
for i, (idx, row) in enumerate(machine_risk_plot.iterrows()):
    ax.text(row["mean"] + row["std"] + 1, i, f'n={int(row["count"])}',
            va="center", fontsize=9, color="gray")
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "risk_by_machine_type.png"), dpi=150)
plt.close()

print(f"   Plots saved to {PLOTS_DIR}/")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  8. Save Model + Artefacts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n Saving model artefacts …")

# Save XGBoost model
model.save_model(os.path.join(MODEL_DIR, "context_risk_xgb.json"))

# Save label encoders
joblib.dump(label_encoders, os.path.join(MODEL_DIR, "label_encoders.joblib"))

# Save feature config for the API (Step 3)
model_config = {
    "feature_columns": FEATURE_COLS,
    "label_classes": {col: list(le.classes_) for col, le in label_encoders.items()},
    "risk_buckets": {
        "Low": [0, 25],
        "Medium": [25, 50],
        "High": [50, 75],
        "Critical": [75, 100],
    },
    "hazard_lethality_weights": HAZARD_LETHALITY,
    "weather_risk_weights": WEATHER_RISK,
    "time_risk_weights": TIME_RISK,
    "model_metrics": {
        "test_r2": float(metrics["Test"]["R²"]),
        "test_mae": float(metrics["Test"]["MAE"]),
        "test_rmse": float(metrics["Test"]["RMSE"]),
        "cv_r2_mean": float(cv_r2.mean()),
        "bucket_accuracy": float(bucket_acc),
    },
}
with open(os.path.join(MODEL_DIR, "model_config.json"), "w") as f:
    json.dump(model_config, f, indent=2)

# Save training data with risk scores (for analysis)
df[FEATURE_COLS + ["context_risk_score", "hazard_type", "severity_score"]].to_csv(
    os.path.join(MODEL_DIR, "training_data_with_scores.csv"), index=False
)

print(f"   ✅ Model            → {MODEL_DIR}/context_risk_xgb.json")
print(f"   ✅ Label encoders   → {MODEL_DIR}/label_encoders.joblib")
print(f"   ✅ Model config     → {MODEL_DIR}/model_config.json")
print(f"   ✅ Training data    → {MODEL_DIR}/training_data_with_scores.csv")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  9. Quick Demo: Sample Predictions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 60)
print(" SAMPLE PREDICTIONS")
print("=" * 60)

demo_scenarios = [
    {"machine_type": "Excavator",  "task_type": "Excavation",       "time_of_day": "Morning",   "weather_condition": "Clear/Unknown", "month": 6,  "day_of_week": "Monday"},
    {"machine_type": "Crane",      "task_type": "Loading/Unloading", "time_of_day": "Afternoon", "weather_condition": "Wind",          "month": 3,  "day_of_week": "Wednesday"},
    {"machine_type": "Bulldozer",  "task_type": "Operating",        "time_of_day": "Night",     "weather_condition": "Rain",          "month": 11, "day_of_week": "Friday"},
    {"machine_type": "Forklift",   "task_type": "Loading/Unloading", "time_of_day": "Morning",   "weather_condition": "Clear/Unknown", "month": 7,  "day_of_week": "Tuesday"},
    {"machine_type": "Aerial Lift", "task_type": "Maintenance",     "time_of_day": "Afternoon", "weather_condition": "Snow/Ice",      "month": 1,  "day_of_week": "Thursday"},
    {"machine_type": "Truck",      "task_type": "General Construction", "time_of_day": "Evening", "weather_condition": "Muddy",        "month": 10, "day_of_week": "Saturday"},
]

for scenario in demo_scenarios:
    # Encode
    row = []
    for col in FEATURE_COLS:
        val = scenario[col]
        if col in label_encoders:
            le = label_encoders[col]
            if val in le.classes_:
                val = le.transform([val])[0]
            else:
                val = 0  # fallback for unseen categories
        row.append(val)

    pred = model.predict(np.array([row]))[0]
    pred = np.clip(pred, 0, 100)
    bucket = risk_bucket(pred)

    print(f"\n  {scenario['machine_type']:15s} | {scenario['task_type']:22s} | "
          f"{scenario['time_of_day']:10s} | {scenario['weather_condition']:13s}")
    print(f"  → Risk Score: {pred:.1f}/100  [{bucket}]")

print("\n" + "=" * 60)
print("=" * 60)
