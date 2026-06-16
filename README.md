# SAARTHI — SmartCabin AI Copilot 🛠

**Edge-AI copilot for operator safety on heavy machinery.** Fully offline, in the
cab. It recognises the operator, watches for fatigue and blind-spot danger in
real time, **explains every alert in plain words**, learns *when* and *how
strongly* to warn, and **adapts the delivery to each person** — buzz + flash for
a hearing-impaired operator, colour-safe cues for a colour-blind one, simpler
screens for a trainee, alerts in their own language.

> **Design rule we never break:** *Safety decides **WHAT** to warn about. The
> person decides **HOW** it's delivered. Learning decides **WHEN** — but can never
> silence a real emergency.*

Built for **Tata Technologies InnoVent-27**, category 3.2.2.4 (Edge AI for
Operator Safety & Human–Machine Interaction). Full plan: [`docs/saarthi-official-plan-v3.md`](docs/saarthi-official-plan-v3.md).

---

## ⚡ TL;DR — run the demo

```bash
# 1. install (CPU only, no GPU, no internet needed after install)
pip install -r requirements.txt

# 2a. run the live system (server + dashboard + phone page)
python run_demo.py
#  → open http://localhost:8000          (the SmartCabin dashboard)
#  → open http://localhost:8000/phone    (the phone buzz page)

# 2b. or run it in the terminal, no browser, no server:
python run_demo.py --headless 200
```

The demo runs with **zero hardware** — webcam, second camera and phone are all
optional. The trained fatigue model is real; only its *input frames* are
simulated when no webcam is present.

---

## 🧠 What you're looking at

The whole project is **three pipelines**. Two are trained offline (on our
laptops, before the demo). The third runs live — it's the product.

| Pipeline | What it is | Where it lives |
|---|---|---|
| **A — Fatigue brain** | MediaPipe features → XGBoost drowsiness classifier + SHAP reasons + 30-s personal calibration | `training/pipeline_a_fatigue/` |
| **B — Intervention brain** | Data-driven Gymnasium worksite simulator + hard-rule engine + adaptive/PPO intervention policy | `training/pipeline_b_simulator/` |
| **Context Risk** | XGBoost trained on real OSHA injury reports → "how risky is this *situation*?" | `training/context_risk/` |
| **C — Live system** | The in-cab copilot that fuses all of the above in real time | `brain/`, `dashboard/`, `phone/` |

### Live data flow (Pipeline C)

```
[cabin cam]  [outside cam]  [telemetry replay]      <- inputs (all optional/simulated)
     |             |              |
     v             v              v
  Q2 fatigue    Q3 zones      Q4 tilt   + Q1 face-ID -> loads profile
 (Pipeline A)  (YOLO/sim)    (telemetry)   (DeepFace/manual)
     +--------------+---------------+
                    v
        HYBRID DECISION ENGINE  (brain/engine.py)
          Tier 1 - hard rules (Level 3, untouchable)
          Tier 2 - adaptive policy (Levels 0-2, reuses Pipeline B)
                    v
     PERSONALISER + REASON CARD  (brain/personalize.py, brain/reason.py)
                    v  FastAPI + WebSocket (brain/server.py)
        [dashboard]  [phone buzz]  [event log (SQLite)]
```

About 6-10x/second: signals in -> the brain answers four questions -> the engine
picks a level -> the personaliser picks the delivery -> outputs fire with a Reason
Card -> the event is logged locally. **Video is processed and thrown away — it
never exists on disk.**

---

## 📁 Repository layout

```
tata_tech/
├── brain/                      # Pipeline C — the live in-cab system
│   ├── server.py               #   FastAPI + WebSocket; serves dashboard & phone
│   ├── pipeline.py             #   Brain: one full tick (the orchestrator)
│   ├── engine.py               #   Hybrid Decision Engine (Tier-1 rules + Tier-2 policy)
│   ├── fatigue.py              #   Q2 - wraps Pipeline A's XGBoost model
│   ├── persons.py              #   Q3 - blind-spot zones (YOLO optional)
│   ├── tilt.py                 #   Q4 - tilt rule
│   ├── profiles.py             #   Q1 - face-ID + operator profiles
│   ├── personalize.py          #   delivery adaptation (buzz/flash/colour/language)
│   ├── reason.py               #   Reason Cards + SQLite event log
│   ├── risk.py                 #   transparent live risk score (0-100)
│   ├── demo_source.py          #   scripted offline signal source (the 5-min beats)
│   └── paths.py                #   wires the offline packages onto sys.path
├── dashboard/index.html        # machine-panel UI (vanilla JS over WebSocket)
├── phone/index.html            # phone buzz page (Vibration API)
├── data/
│   ├── profiles.json           # operator cards (hearing/colour/language/experience)
│   ├── telemetry.csv           # excavator replay (FlywheelAI-derived structure)
│   ├── hazard_frequency.csv    # Dev3 -> Dev1 handoff (real OSHA hazard mix)
│   └── events.db               # local alert log (git-ignored, created at runtime)
├── training/
│   ├── pipeline_a_fatigue/     # fatigue model, feature extraction, calibration
│   ├── pipeline_b_simulator/   # Gymnasium env, hazard injector, reward, policies
│   └── context_risk/           # OSHA wrangling + XGBoost context-risk model
├── docs/                       # the official plan + demo script
├── run_demo.py                 # one-command launcher (server or headless)
└── requirements.txt            # consolidated dependencies
```

---

## 🚀 Installation

CPU-only; no GPU required. Python 3.10+.

```bash
pip install -r requirements.txt
```

That installs the **core** (numpy, pandas, xgboost, shap, scikit-learn,
fastapi, uvicorn, websockets, gymnasium) — everything needed to run the live
demo and all three pipelines.

**Optional** dependencies (commented in `requirements.txt`) enable the real
hardware path and are *not* needed for the demo:
`opencv-python`, `mediapipe` (webcam → fatigue features), `ultralytics`
(YOLO11 person detection), `deepface` (face-ID), `stable-baselines3` (PPO training).

The system degrades gracefully: each module reports its backend at
`GET /api/health`, e.g. `fatigue_backend: xgboost`, `person_backend: simulated`.

---

## 🎬 Running the demo

### Live (recommended)

```bash
python run_demo.py
```

- **http://localhost:8000** — the SmartCabin dashboard (operator card, fatigue
  gauge, blind-spot zones, live risk score, alert banner, Reason Card, incident
  timeline, context-risk badge).
- **http://localhost:8000/phone** — open on an Android phone on the same
  WiFi/hotspot (use the laptop's LAN IP, e.g. `http://192.168.x.x:8000/phone`),
  tap **"enable buzz"**, and it vibrates on alerts. *(iPhones block the
  Vibration API — the page falls back to an on-screen buzz animation.)*

The demo loops through the scripted beats automatically. You can switch operators
live from the dashboard buttons (Ravi / Priya / Arjun) and watch the delivery and
UI adapt.

### Headless (no browser/server)

```bash
python run_demo.py --headless 200
```

Prints one line per tick: phase, operator, fatigue %, zone, tilt, risk score,
alert level + tier, and the Reason Card title. Great for a quick sanity check.

### What the demo shows (mapped to the plan's 5 minutes)

| Beat | What happens | Feature proven |
|---|---|---|
| **Recognition** | Ravi's profile card loads (hearing-impaired · हिंदी) | Q1 face-ID + profiles |
| **Fatigue** | eyes droop → phone **buzzes** + blue/white flash + Hindi text + Reason Card "PERCLOS …" | Pipeline A model + SHAP + personaliser |
| **Fatigue L3** | sustained drowsiness while moving → **Level 3** | Tier-1 hard rule |
| **Blind spot** | worker in RED zone while reversing → **Level 3**, "machine slowing" | Tier-1 hard rule (no AI can silence it) |
| **Tilt** | slope climb → tilt-critical → **Level 3** | telemetry rule on real-structure data |
| **Adaptation** | switch to Priya (trainee) → simpler UI, **earlier** warnings | personaliser + engine |
| **Edge proof** | kill the internet — everything keeps running | fully offline |

See [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md) for the full beat-by-beat script.

---

## 🔬 Using each pipeline on its own

### Pipeline A — Fatigue (`training/pipeline_a_fatigue/`)

The trained XGBoost classifier with streaming inference and SHAP reasons:

```python
from fatigue_monitor import calibrate, FatigueMonitor

baseline = calibrate(alert_feature_rows)      # ~30 s of alert windows
monitor  = FatigueMonitor(baseline)
result   = monitor.update(feature_window)     # dict of 12 features
# -> {"decision","p_at_risk","severity","reasons":[(feature,value,direction)...]}
```

Webcam feature extraction lives in `extract_features.py` (needs MediaPipe).
Full training flow: `extract_features.py` → `balance_data.py` (SMOTE) →
`train_xgboost.py`. XGBoost training auto-uses **CUDA** (`device='cuda'`) when an
NVIDIA GPU is present (see `training/gpu_utils.py`), else CPU.

### Pipeline B — Simulator (`training/pipeline_b_simulator/`)

A custom Gymnasium worksite simulator with deterministic Tier-1 hard rules and a
reward function, ready for PPO training. **Integrity-verified (12/12 checks):**

```bash
cd training/pipeline_b_simulator
python verify_pipeline2.py     # 12/12 checks: gym contract, hard rules, reward hook...
python demo.py                 # adaptive-threshold policy across scenarios
python cabin_env.py            # random-policy smoke run
```

The env (`cabin_env.py`) exposes the standard Gymnasium API
(`Discrete(3)` actions, 5-dim `Box` observation). Hazard frequencies come from
the real OSHA analysis in `data/hazard_frequency.csv`; the reward function is
`rewards/reward_shaping.py`; the shipped Tier-2 fallback is the adaptive policy
in `policies/`.

**PPO training (CUDA, PyTorch):**

```bash
cd training/pipeline_b_simulator
pip install -r requirements_dev2.txt          # torch (CUDA build) + stable-baselines3
python train_ppo.py --steps 200000            # trains on the GPU (device auto-detected)
tensorboard --logdir runs/ppo_tb              # watch reward convergence
python enjoy_ppo.py                           # run the saved policy; it never emits Level 3
```

The PPO agent only chooses soft Levels 0–2; Tier-1 hard rules stay in the env and
can't be overridden. The trained policy is saved to
`models/ppo_intervention_policy.zip`. Per the plan's checkpoint rule, if PPO
doesn't beat the fallback it stays on the roadmap and the adaptive-threshold
policy ships — both are honest and demo-able.

### Context Risk (`training/context_risk/`)

```python
from context_risk_api import get_context_risk
get_context_risk(machine_type="Excavator", task_type="Excavation",
                 time_of_day="Morning", weather_condition="Rain")
# -> {"risk_score","risk_bucket","risk_color", ...}
```

Re-run the analysis: unzip `cleaned_data.zip`, then `osha_data_wrangling.py` →
`xgboost_context_risk.py`. Plots land in `context_risk/plots/`.

---

## 🌐 API reference (live server)

| Method | Route | Purpose |
|---|---|---|
| `WS` | `/ws` | live frame stream (JSON, ~6-10 Hz) |
| `GET` | `/` | the dashboard |
| `GET` | `/phone` | the phone buzz page |
| `GET` | `/api/health` | backends + tick rate |
| `GET` | `/api/profiles` | all operator profiles |
| `POST` | `/api/operator/{id}` | switch active operator (`ravi`/`priya`/`arjun`) |
| `GET` | `/api/timeline` | recent Level >= 2 incidents from the event log |

Each WebSocket frame contains: `operator`, `fatigue`, `zone`, `tilt`, `machine`,
`risk_score`, `alert` (level/label/tier/delivery), `reason_card`, `context_risk`.

---

## ✅ What's live vs. roadmap

**Live & demo-able today:** real XGBoost fatigue model + SHAP reasons, personal
calibration, Hybrid Decision Engine with all three Tier-1 hard rules, adaptive
Tier-2 policy, personaliser (hearing/colour/language/trainee), Reason Cards,
transparent risk score, blind-spot zones, tilt rule, dashboard, phone buzz,
SQLite incident log, context-risk badge — **all offline on a CPU.**

**Roadmap (in the plan, not claimed as built):** PPO agent replacing the adaptive
fallback, YOLO fine-tune on construction datasets, full sensor fusion / real IMU,
Site Pulse cloud sync, OEM integration. See plan §13.

---

## 🧩 How it was built (branch provenance)

This `pipe-2` integration fuses work from the team branches:
`pipe-1`/`sounak` (Pipeline A), `sharu` (Pipeline B reward/policy/env),
`pipe-2` (Pipeline B simulator + context risk), plus the `brain/` live system
that ties them together.

---

*The idea is the easy 10%. Skeleton first, muscle after, roadmap on a slide.*
