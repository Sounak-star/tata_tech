"""
run_demo.py — one command to run SAARTHI.

    python run_demo.py                # launch the live server + dashboard
    python run_demo.py --headless 60  # run 60 brain ticks in the terminal (no server)

The live server serves:
    http://localhost:8000        the SmartCabin dashboard
    http://localhost:8000/phone  the phone buzz page (open on an Android phone on
                                 the same WiFi/hotspot using the laptop's LAN IP)
"""

from __future__ import annotations

import argparse


def headless(n: int) -> None:
    from brain.pipeline import Brain
    from brain.demo_source import DemoSource

    brain, source = Brain(), DemoSource()
    print(f"fatigue backend: {brain.fatigue.backend} | "
          f"vision: {brain.persons.backend} | faceid: {brain.faceid.backend}")
    print("-" * 92)
    for _ in range(n):
        sig = source.step()
        if sig.get("_switch_operator"):
            brain.switch_operator(sig["_switch_operator"])
        f = brain.tick(sig)
        a = f["alert"]
        rc = f["reason_card"]["title"] if f["reason_card"] else "-"
        print(f"t{f['tick']:>3} {sig['phase']:<13} | {f['operator']['name'][:12]:<12} "
              f"| fat {f['fatigue']['p']:.2f} | zone {f['zone']:<5} "
              f"| tilt {f['tilt']['tilt_angle']:>4}° | risk {f['risk_score']['score']:>5} "
              f"| L{a['level']} {a['label']:<10} [{a['tier']}] | {rc}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the SAARTHI demo.")
    ap.add_argument("--headless", nargs="?", const=60, type=int, default=None,
                    metavar="TICKS", help="run N ticks in the terminal, no server")
    args = ap.parse_args()

    if args.headless is not None:
        headless(args.headless)
    else:
        from brain.server import main as serve
        print("SAARTHI live → http://localhost:8000  (phone: /phone)")
        serve()


if __name__ == "__main__":
    main()
