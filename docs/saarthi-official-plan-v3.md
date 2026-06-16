# Project SAARTHI — SmartCabin AI Copilot
## Official Plan v3 — the single source of truth (plain-language edition)

**Competition:** Tata Technologies InnoVent-27
**Category:** 3.2.2.4 — Edge AI for Operator Safety & Human–Machine Interaction
**Team:** 5 members · **Build:** 100% software · everything built fresh by us.

> One rule decided everything in this plan: **if we can't show it running live in the 5-minute demo, it goes on the roadmap slide, not in the feature list.** Judges score what they see.

---

## 1. The pitch & the problem

**Problem:** Operators of heavy machinery face safety risks from fatigue, distraction, blind spots and varying skill levels. Today's safety systems use fixed rules and generic alerts — one beep for everyone — so they're ignored, distrusted, and useless for operators who can't hear a beep or read a warning in English. Sites also lack honest insight into near-misses.

**Our answer in one breath:** SAARTHI is an edge-AI copilot that lives **inside the cab**, fully offline. It recognises the operator, watches for fatigue and blind-spot danger in real time, **explains every alert in plain words**, learns the best moment and strength for each warning, and **changes how the machine talks to each person** — buzz + flash for a hearing-impaired operator, colour-safe cues for a colour-blind one, simpler screens for a trainee, alerts in their own language. Supervisors get a privacy-safe summary — never surveillance, never video.

**The design rule we never break:** *Safety decides WHAT to warn about. The person decides HOW it's delivered. Learning decides WHEN — but can never silence a real emergency.*

---

## 2. Mini-dictionary (one line each — read this first)

| Word | Plain meaning |
|---|---|
| **Edge AI** | The AI runs on the machine itself — no internet, no cloud |
| **Model** | A trained "brain file" that answers a question from data |
| **Landmarks** | Dots the AI places on a face so we can measure eyes/mouth/head |
| **EAR / PERCLOS** | How open the eyes are / % of recent time they were nearly shut |
| **Quantization** | Shrinking a model so it's small and fast — like compressing a video |
| **Telemetry** | The machine's own numbers: engine, speed, reversing, tilt |
| **Simulator** | A software "practice world" where an AI can try things safely |
| **PPO agent** | An AI that learns by practising in the simulator and getting scores (rewards) for good decisions |
| **SHAP** | A tool that asks a trained model "which input made you decide that?" |
| **SMOTE** | A one-line trick that balances a dataset when one class is rare |
| **WebSocket** | A pipe that pushes instant messages from our Python brain to the screen |
| **Profile** | A saved card per operator: hearing, colour vision, language, experience |

---

## 3. The architecture — three pipelines

The whole project is three pipelines. Two run **before** the demo (training, on our laptops). One runs **live** (the actual product).

### Pipeline A — train the fatigue brain (offline)

```
UTA-RLDD  +  YawDD   (real drowsiness video datasets)
        ↓
MediaPipe feature extraction
(turn every video frame into numbers:
EAR, PERCLOS, yawn ratio, head angle, blink rate)
        ↓
SMOTE  (balance the rare yawning class)
        ↓
XGBoost training
(learns the drowsiness pattern from ~100,000 rows)
        ↓
Fatigue classifier  (saved model + SHAP explanations)
```

Why this way: we never train on raw video — we train on the *numbers* MediaPipe extracts. That makes the model tiny, fast on a CPU, and explainable. Published research reports roughly 90%+ accuracy for XGBoost on exactly this kind of tabular safety data, so we can quote real benchmarks.

### Pipeline B — train the intervention brain (offline)

```
OSHA accident reports  +  real excavator control data
        ↓
Analyze patterns & risk factors
(how often hazards happen, how severe,
how real machines actually move)
        ↓
Build a data-informed worksite simulator
(a custom practice world in Python/Gymnasium)
        ↓
Generate realistic scenarios
(hazards appear at real-world frequencies;
operator drowsiness replayed from UTA-RLDD;
machine behaviour replayed from real control data)
        ↓
Train the PPO agent — soft zone only
(it learns WHEN to nudge and HOW strongly,
for alert levels 0–2, per operator type)
        ↓
Deploy inside the Hybrid Decision Engine
(hard safety rules own Level 3 —
the agent can NEVER override them)
```

Why this way: the simulator's events aren't invented — their frequency and severity come from real accident statistics, and the machine's behaviour comes from real excavator recordings. The agent only optimises comfort and timing; emergencies stay deterministic.

**Side branch — the Context Risk Model:** a small XGBoost trained on the OSHA reports that answers "how dangerous is this *situation*?" (machine type + task + time of day + conditions → risk level). It has two jobs: it makes dangerous scenarios appear more often during PPO training, and it can show a "today's context risk" badge on the supervisor page. It does **not** predict live crashes — no public dataset exists for that, so the live risk score stays a transparent formula.

### Pipeline C — the live system (this is the product)

```
INPUTS
[Cabin camera]  [Outside camera]  [Telemetry replay]
      |               |            (from real excavator
      |               |             control data + tilt)
      └───────┬───────┴───────┬──────────┘
              ↓               ↓
THE BRAIN (all on-device, fully offline)
  1. Who is it?        → DeepFace → loads profile
  2. Tired/distracted? → MediaPipe → XGBoost fatigue
                          model + personal calibration
  3. Danger outside?   → YOLO11-nano → GREEN/AMBER/RED zones
  4. Machine at risk?  → tilt rule on telemetry
              ↓
HYBRID DECISION ENGINE
  Tier 1 — hard rules (Level 3, untouchable)
  Tier 2 — PPO policy (Levels 0–2 timing & strength)
              ↓
PERSONALISER + REASON CARD
  (adapts delivery to the operator's profile;
   every alert shows WHY it fired)
              ↓  FastAPI + WebSocket
OUTPUTS
[Dashboard]  [Phone buzz]  [Voice clip]  [Event log]
              ↓ (summaries only, when online)
[Site Pulse supervisor page]
```

About 10 times every second: frames and telemetry go in → the brain answers its four questions → the engine picks an alert level → the personaliser picks the delivery → outputs fire with a Reason Card → the event is logged locally. **Video is processed and thrown away — it never exists on disk.**

---

## 4. The live system, block by block

### Block 1 — Inputs

| Real machine | Our demo stand-in |
|---|---|
| Camera facing the operator | Laptop webcam |
| Camera watching behind the machine | Phone / second webcam / real excavator footage from our dataset |
| Machine data incl. tilt | A CSV replayed line-by-line — **derived from real excavator control recordings**, with tilt and environment columns added |

### Block 2 — The brain's four questions

**Q1 — Who is sitting here?** DeepFace (free Python library; one function call compares the webcam face to saved photos). Loads that operator's profile. We enrol with our own team photos — no dataset needed.

**Q2 — Are they tired or distracted?** MediaPipe places ~468 dots on the face. We compute EAR, PERCLOS, yawn ratio, blink rate and head pose, and feed those numbers into the trained XGBoost fatigue classifier. SHAP then tells us *which* signal drove the decision — that becomes the Reason Card.
**Personal calibration:** the first ~30 seconds with a new operator, the system learns *their* normal blink pattern and stores it in the profile. Thresholds tuned to the person beat one-size-fits-all.

**Q3 — Anyone in danger outside?** YOLO11-nano (tiny, free, pre-trained) draws boxes around people in the outside feed. Box size → honest zones: GREEN safe / AMBER caution / RED danger. Optional upgrade: fine-tune on construction-site images so it stays sharp around helmets, vests and machinery.

**Q4 — Is the machine itself at risk?** One clean rule on telemetry: tilt beyond a safe angle while moving → slope warning; beyond a critical angle → Level 3.

### Block 3 — The Hybrid Decision Engine (the heart)

**Tier 1 — hard rules. Deterministic. Nothing can override them:**

| Situation | Level |
|---|---|
| Eyes closed > 2 s while machine moving | 3 — act now |
| Person in RED zone + machine reversing | 3 — overrides all |
| Tilt past critical while moving | 3 |

**Tier 2 — the learned policy.** For everything below an emergency (gentle nudges, early warnings, escalation speed), the PPO agent decides the timing and strength — trained in the simulator, personalised by operator type. If the agent isn't training stably by its checkpoint (see phases), a simple fallback ships instead: **adaptive thresholds** — if an operator keeps dismissing gentle nudges, the next warning comes earlier and stronger. Both versions are honest and demo-able.

A transparent **live risk score (0–100)** for the dashboard combines the three signals with a visible formula — fatigue probability, zone level, tilt — so a judge can interrogate every number.

### Block 4 — Personaliser + Reason Cards

The profile decides delivery — never the danger level:
- Hearing-impaired → strong phone buzz + big screen flash (zero reliance on sound)
- Colour-blind → blue/white high-contrast + icons (never red-vs-green alone)
- Trainee → simpler wording, earlier warnings, fewer gauges
- Language → text + pre-recorded voice clip (Hindi / Tamil / Telugu / English)

**Reason Card:** every alert shows *why* in plain words — "Eyes closed 2.3 s while machine moving (PERCLOS 71%)" — plus a tiny live graph. SHAP supplies the fatigue reasons; the rules explain themselves.

### Block 5 — Outputs & memory

- **Dashboard** — a web app styled like a machine panel: operator card, zone view, risk score, alert banners, Reason Card, and a rolling **60-second Incident Timeline** (replay the *events* of any Level-3 — never video).
- **Phone = the "seat"** — an Android phone on local WiFi buzzes for haptic alerts (browser Vibration API; iPhones block it).
- **Voice clips** — pre-recorded alert phrases per language.
- **Simulated machine actions** — "machine slowing / safe mode" animations triggered by Level 3.
- **Memory (local only):** `profiles.json`, an SQLite event log with timestamps and reasons (audit-ready by design), the rolling timeline.
- **Site Pulse** (stretch): when internet returns, event *summaries* sync to one supervisor page — alerts by level, near-miss count, shift summary, and the context-risk badge. Aggregated and privacy-safe.

---

## 5. All datasets — confirmed links, owners, and when

**Group 1 — Fatigue training (Pipeline A) · AI Lead · download week 1**

| Dataset | What it gives | Link |
|---|---|---|
| UTA-RLDD | 30 hrs video, 60 people, 3 drowsiness levels — our main training set | kaggle.com/datasets/minhngt02/uta-rldd |
| YawDD | ~351 driver videos with yawning — sharpens yawn detection | kaggle.com/datasets/enider/yawdd-dataset |
| NTHU-DDD (optional) | Extra validation set — request by email, takes days | cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/ |

**Group 2 — Real excavator operations (Pipelines B & C) · Integration Lead · download week 1**

| Dataset | What it gives | Link |
|---|---|---|
| FlywheelAI excavator dataset | Real excavator sessions: 4 camera angles + synchronized joystick/control CSVs — powers our telemetry replay AND blind-spot demo footage | huggingface.co/datasets/FlywheelAI/excavator-dataset |
| Excavator-motion (optional) | Full dig-to-load trajectories with joint angles, 3 machine models | huggingface.co/datasets/fuxi-robot/excavator-motion |

**Group 3 — Accident statistics (Pipeline B + Context Risk Model) · RL & Simulation Lead · download week 1**

| Dataset | What it gives | Link |
|---|---|---|
| OSHA Severe Injury Reports (official) | Every reported severe US workplace injury since 2015, with incident descriptions — seeds simulator frequencies + trains the context risk model | osha.gov/severe-injury-reports |
| OSHA Kaggle mirror 2015–17 | Same data, one-click CSV | kaggle.com/datasets/ruqaiyaship/osha-accident-and-injury-data-1517 |
| 1M injury records 2016–2021 | Bigger mirror for the context model | kaggle.com/datasets/robikscube/osha-injury-data-20162021 |
| Building Safer Sites (2025) | New large-scale multi-level construction-safety dataset built for exactly this kind of model | arxiv.org/abs/2508.09203 |

**Group 4 — Construction-site vision (YOLO fine-tune, Phase 2) · Vision Lead**

| Dataset | What it gives | Link |
|---|---|---|
| MOCS | 41,668 images from 174 real sites, 13 classes (worker, excavator, crane…) | anlab340.com/Archives/IndexArctype/index/t_id/17.html |
| SODA | 19,846 images, 15 classes incl. person, helmet, vest | via github.com/YUZ128pitt/OpenConstruction |

**Group 5 — PPE / hard hat (stretch) · Vision Lead**

| Dataset | What it gives | Link |
|---|---|---|
| Hard Hat Workers | Helmet/head/person boxes, demo-ready | public.roboflow.com/object-detection/hard-hat-workers |
| Safety Helmet Detection | ~5k annotated images | kaggle.com/datasets/andrewmvd/hard-hat-detection |

**Group 6 — No dataset needed**
Face ID (DeepFace is pretrained — enrol with team photos) · voice clips (we record our own) · profiles & tilt thresholds (derived from Group 2 data).

---

## 6. Tech stack — one line each on why

| Piece | Tool | Why |
|---|---|---|
| Face dots | MediaPipe Face Landmarker | Free, real-time on CPU |
| Fatigue model | XGBoost (+ SMOTE via imbalanced-learn) | Tiny, fast, ~90%+ benchmarks on tabular safety data |
| Explainability | SHAP | Reads the XGBoost model → powers Reason Cards |
| Person detection | YOLO11-nano → ONNX → INT8 | Newest nano model, real-time on CPU |
| Face ID | DeepFace | One function call, pretrained |
| Simulator + agent | Gymnasium + Stable-Baselines3 (PPO) | Standard, well-documented RL tooling |
| Brain runtime | Python + OpenCV + ONNX Runtime | Industry standard |
| Brain → screen pipe | FastAPI + WebSocket | Instant push of alerts as JSON |
| Dashboard | React + Vite + Tailwind | Light, fast to learn from zero |
| Haptics | Android phone + Vibration API | Zero hardware, great on stage |
| Storage | SQLite + JSON | Simple, local, audit-friendly |
| Edge proof | Measured MB + FPS, internet killed live, "Jetson-class ready" backed by numbers; (stretch) AWS IoT Greengrass emulation | Honest and sponsor-aligned |

**Repo layout:**

```
saarthi/
  brain/        # camera.py, fatigue.py, calibrate.py, persons.py,
                # faceid.py, tilt.py, engine.py, personalize.py, server.py
  training/     # pipeline_a_fatigue/, pipeline_b_simulator/, context_risk/
  dashboard/    # React+Vite app (panel, zones, reason card, timeline, site-pulse)
  phone/        # the buzz page
  data/         # profiles.json, telemetry.csv, demo clips, voice clips
  docs/         # this plan, demo script, stats + sources, roadmap slide
```

---

## 7. What makes ours hard to copy

1. **Personal calibration** — learns *your* normal in 30 seconds
2. **Alert-translation layer** — disability/experience adaptation is the product's core, not a slide
3. **Reason Cards** — explainability you can see, on every alert
4. **Learned intervention timing with a safety guarantee** — a PPO agent tunes the nudges, hard rules own the emergencies
5. **Real-data grounding everywhere** — real drowsiness videos, real excavator recordings, real accident statistics
6. **Incident Timeline** — replayable, audit-ready, with zero video stored
7. **Regional languages + full-offline edge** — built, measured, demonstrated live

Any one layer is copyable. All seven, running live, is ours.

---

## 8. Team of 5 — roles and week-1 wins

| Role | Owns | Week-1 win |
|---|---|---|
| **AI Lead** | Pipeline A: MediaPipe features, SMOTE, XGBoost, SHAP, calibration | Webcam blink counter; UTA-RLDD downloaded |
| **Vision Lead** | YOLO11 zones + DeepFace ID (+ MOCS fine-tune later) | Person boxes on excavator footage; a teammate's face recognised |
| **App Lead** | Dashboard, phone buzz, languages, Reason Card UI, Incident Timeline | Dashboard skeleton showing a WebSocket test message |
| **Integration Lead** | FastAPI/WebSocket glue, engine.py, profiles, telemetry replay from real control data, metrics | "Brain says hi" travels Python → screen; FlywheelAI data downloaded |
| **RL & Simulation Lead** | Pipeline B: simulator, PPO, reward design + the Context Risk Model | OSHA CSV downloaded; hazard-frequency analysis started |

Submission writing, sourced statistics, the roadmap slide and demo direction are **shared duties** — assign each to a named owner in the week-1 meeting. Aim for genuine team diversity; criterion 6.0 scores it.

---

## 9. Build phases

**Phase 0 — now → idea submission.** Register, confirm official stage dates, write submission answers (section 10 maps them). **Scope lock:** from here, new ideas go to the roadmap slide, not the build.

**Phase 1 — walking skeleton (≈2–3 wks).** One thin thread end-to-end, ugly is fine: webcam → "eyes closed too long" (simple rule) → WebSocket → red banner. In parallel: datasets downloaded, simulator skeleton, XGBoost first training run.

**Phase 2 — full body (≈3–4 wks).** Swap the rule for the trained XGBoost model. Add face ID + profiles, person zones, tilt rule, hard-rules engine, personaliser, phone buzz, languages, Reason Cards. Simulator generating scenarios; PPO training begins.

**⚑ PPO checkpoint (end of Phase 2 + 2 weeks):** if the agent trains stably and beats the fallback, it ships. If not, **adaptive thresholds ship instead and PPO moves to the roadmap.** This decision is made once, on the date, by the whole team — no extensions.

**Phase 3 — polish + proof (≈2–3 wks).** Calibration, Incident Timeline, Site Pulse + context-risk badge, colour-safe palettes, measure and print the numbers (model MBs, FPS, latency), record the backup video.

**Phase 4 — demo drill.** Rehearse the 5 minutes until boring. Test on the weakest laptop available.

---

## 10. Hitting the scoring sheet

- **6.0 Diversity:** inclusion is *in the product* (hearing-impaired, colour-blind, trainee, languages) + sourced workforce stats + a genuinely diverse team
- **6.1 Novelty:** everyone *detects* the operator — we **adapt the machine to the operator**, and our alert timing is *learned* in a safety-bounded way. Calibration + Reason Cards + the hybrid engine are the outside-the-box story
- **6.2 Feasibility:** live end-to-end demo; quantized models with measured numbers; published ~90%+ benchmarks for our model class; a clean repo; logic a judge can interrogate line-by-line
- **6.3 Impact & scalability:** fewer accidents and near-misses [insert sourced figures]; opens operator jobs to people currently excluded; retrofit kit for any machine brand; languages scale across India; the roadmap (section 13) shows the business ceiling

---

## 11. The 5-minute demo, beat by beat

> Before stage: laptop + phone on a **local hotspot**. Killing the *internet* later proves edge while the buzz still works.

1. **0:00 — Hook.** "Meet Ravi, an excavator operator who's hard of hearing…" + one hard statistic.
2. **0:40 — Recognition.** Teammate sits → profile card: *"Ravi · hearing-impaired · हिंदी."* The 30-s calibration bar runs (pre-warmed).
3. **1:20 — Fatigue.** Eyes droop → **phone BUZZES** + blue/white flash + Hindi text. Point at the **Reason Card**: "PERCLOS 71%, eyes closed 2.3 s — the system shows its work." One line: "and the *timing* of that nudge was learned by an agent trained on real accident statistics."
4. **2:10 — Blind spot.** A person appears behind on the second feed while telemetry says *reversing* → RED zone, strong buzz, "machine slowing" animation. "This one is a hard rule — no AI is allowed to silence it."
5. **2:50 — Tilt.** Telemetry replays a slope climb → slope warning fires. "Same brain also watches the machine itself — on real excavator data."
6. **3:15 — Adaptation.** Profile switch to *Priya, trainee* → dashboard visibly simplifies, warnings come earlier.
7. **3:45 — Edge proof.** Kill the internet live — everything keeps running. Numbers card: "all models ≈ __ MB, __ FPS on this laptop's CPU — Jetson-class ready." Flash the **Incident Timeline** replay.
8. **4:30 — Close.** Site Pulse glance (context-risk badge) → impact stats → roadmap in one breath → end on Ravi.

**Backups (all three ready):**

| If this breaks | We do this |
|---|---|
| Webcam / lighting misbehaves | Pre-recorded clip fed into the same pipeline |
| Phone vibration blocked | On-screen buzz animation + sound effect |
| Stage laptop too slow | Lower camera resolution; FPS stays smooth |

---

## 12. Risks and where each one is killed

| Risk | Killed by |
|---|---|
| "Which dataset did you train on?" | Named, linked, downloadable datasets for every model (section 5) |
| "Is your simulation realistic?" | Hazard frequencies from real accident reports; machine behaviour from real excavator recordings |
| "What if the agent learns to stay quiet?" | Hard rules own Level 3; the agent physically cannot touch them |
| "How do you explain an alert?" | Reason Cards: SHAP for the model, rule traces for the rest |
| PPO doesn't train in time | The checkpoint rule — adaptive thresholds ship, PPO moves to roadmap |
| Webcam fatigue detection shaky | Trained model + personal calibration + backup clip + reported accuracy |
| "Is this really edge?" | All models local, measured MB/FPS, internet killed on stage |
| Inclusion looks bolted-on | It IS the core (alert-translation layer) + sourced stats + the live no-beep buzz moment |
| Scope creep | The Phase-0 scope lock: new ideas → roadmap, not build |

---

## 13. The Roadmap slide (vision, never claimed as built)

- Fleet-wide workforce intelligence: operator development scores, skill-based task allocation, fatigue-aware shift planning
- Insurance-ready risk profiles and automated compliance reporting
- Full reinforcement learning across the whole intervention range, trained on real fleet data
- True digital twin of specific machines; full sensor fusion (real IMU tip-over prediction, environmental sensing)
- OEM partnerships: factory-fitted SAARTHI; AWS Greengrass fleet sync
- Pilot site → certification path

One sentence for the stage: *"Everything you saw runs today, offline, on this laptop — and here's the platform it grows into."*

---

## 14. First 7 days — start here

1. Confirm official InnoVent dates and rules; register the team.
2. Week-1 meeting: assign the 5 roles + the shared story duties; agree the scope lock; check the 6.0 diversity box honestly.
3. Create the repo with the layout above + a shared task board.
4. **AI Lead:** download UTA-RLDD + YawDD; MediaPipe blink counter running.
5. **Vision Lead:** DeepFace hello-world (recognise a teammate); YOLO11 boxes on excavator footage.
6. **App Lead:** Vite+React dashboard skeleton + WebSocket echo test.
7. **Integration Lead:** download FlywheelAI excavator data; FastAPI server pushing a fake alert to the dashboard; first telemetry.csv derived from real control data.
8. **RL & Simulation Lead:** download the OSHA Kaggle CSV; first hazard-frequency analysis; simulator skeleton (empty Gym environment that runs).
9. Everyone: watch one "day in an excavator cab" video.

When the blink counter, the person boxes, the dashboard echo and the empty simulator all exist — the skeleton walks. Everything after is muscle.

---

*The idea is the easy 10%. Five people who each own a pipeline, a drilled demo, and the discipline of the scope lock — that's the 90% that wins. Skeleton first. Muscle after. Roadmap on a slide.*
