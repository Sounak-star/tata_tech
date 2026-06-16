# SAARTHI — 5-Minute Demo Script

> Before stage: laptop + phone on a **local hotspot**. Start the server
> (`python run_demo.py`), open the dashboard on the laptop and the phone page on
> an Android phone. Killing the *internet* later proves edge while the buzz still
> works. The demo loop auto-runs through the beats below and repeats.

| Time | Beat | What you say | What the screen does |
|---|---|---|---|
| 0:00 | **Hook** | "Meet Ravi, an excavator operator who's hard of hearing…" + one hard statistic. | Dashboard idle, Ravi's card showing (hearing-impaired · हिंदी). |
| 0:40 | **Recognition** | "The cab recognises him and loads his profile — language, hearing, experience." | Operator card; calibration assumed pre-warmed. |
| 1:20 | **Fatigue** | "His eyes start to droop. No beep — he can't hear it. Instead the **phone buzzes**, the screen flashes blue/white, and the alert is in Hindi." Point at the **Reason Card**: "PERCLOS 71%, the system shows its work." Add: "the *timing* of that nudge is decided by a safety-bounded policy trained on real accident statistics." | Fatigue gauge climbs; banner → LEVEL 1/2; phone buzzes; Reason Card lists SHAP factors. |
| 2:10 | **Fatigue → emergency** | "If he keeps going, it escalates to a hard rule no AI can soften." | Banner → **LEVEL 3 EMERGENCY**, "Eyes closed > 2 s while machine moving". |
| 2:30 | **Blind spot** | "A worker steps behind the machine while it's reversing." | Zone → **RED**, reversing = YES, banner → **LEVEL 3**, "Person in RED zone while reversing". "This is a hard rule — no AI is allowed to silence it." |
| 3:00 | **Tilt** | "Same brain also watches the machine itself — on real excavator data." | Tilt climbs to 28°, banner → **LEVEL 3**, "Tilt past critical while moving". |
| 3:15 | **Adaptation** | "Now a trainee, Priya, takes the seat." Click **Priya**. "The dashboard simplifies and her warnings come earlier." | Operator card switches; warnings fire at lower fatigue (earlier nudges). |
| 3:45 | **Edge proof** | "Watch — I'll kill the internet." (disable WiFi/unplug). "Everything keeps running. All models are on this laptop's CPU." | Dashboard keeps streaming; phone keeps buzzing. |
| 4:30 | **Close** | Glance at the **context-risk badge** (from real OSHA data) and the **incident timeline** (replayable, zero video stored) → impact stats → roadmap in one breath → end on Ravi. | Timeline shows logged Level-3 events; context badge shows site risk. |

## Backups (all three ready)

| If this breaks | Do this |
|---|---|
| Webcam / lighting misbehaves | The demo already runs on the simulated source — nothing to do. |
| Phone vibration blocked (iPhone) | The phone page shows an on-screen buzz animation instead. |
| Stage laptop too slow | Lower `TICK_HZ` in `brain/server.py`; the loop stays smooth. |

## Quick verification before going on stage

```bash
python run_demo.py --headless 200      # see all beats fire in the terminal
cd training/pipeline_b_simulator && python verify_pipeline2.py   # 12/12 checks
curl -s http://localhost:8000/api/health                          # backends OK
```
