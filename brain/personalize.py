"""
brain/personalize.py — the Personaliser (Block 4).

The design rule we never break:
    Safety decides WHAT to warn about. The person decides HOW it's delivered.
    Learning decides WHEN — but can never silence a real emergency.

The profile decides delivery — never the danger level:
  • Hearing-impaired → strong phone buzz + big screen flash (zero reliance on sound)
  • Colour-blind     → blue/white high-contrast + icons (never red-vs-green alone)
  • Trainee          → simpler wording, fewer gauges (earlier warnings handled in engine)
  • Language         → text + pre-recorded voice clip (hi / ta / te / en)
"""

from __future__ import annotations

from typing import Dict

# Short alert phrases per language, keyed by level. Voice clips share these keys.
_PHRASES: Dict[str, Dict[int, str]] = {
    "en": {1: "Heads up — stay alert.", 2: "Warning — take action now.",
           3: "EMERGENCY — machine slowing."},
    "hi": {1: "सावधान रहें।", 2: "चेतावनी — तुरंत कार्रवाई करें।",
           3: "आपातकाल — मशीन धीमी हो रही है।"},
    "ta": {1: "கவனமாக இருங்கள்.", 2: "எச்சரிக்கை — உடனே நடவடிக்கை.",
           3: "அவசரம் — இயந்திரம் நிற்கிறது."},
    "te": {1: "అప్రమత్తంగా ఉండండి.", 2: "హెచ్చరిక — వెంటనే చర్య.",
           3: "అత్యవసరం — యంత్రం ఆగుతోంది."},
}

# Colour-safe vs standard palettes (hex).
_PALETTE_STANDARD = {1: "#FFC107", 2: "#FF9800", 3: "#F44336"}
_PALETTE_COLORSAFE = {1: "#90CAF9", 2: "#1E88E5", 3: "#0D47A1"}


def personalise(level: int, profile: dict) -> dict:
    """Turn an alert level + operator profile into a concrete delivery plan."""
    if level <= 0:
        return {"buzz": False, "flash": False, "level": 0}

    lang = profile.get("language", "en")
    colorsafe = profile.get("color_vision", "normal") != "normal"
    hearing_impaired = profile.get("hearing") == "impaired"
    trainee = profile.get("experience") == "trainee"

    palette = _PALETTE_COLORSAFE if colorsafe else _PALETTE_STANDARD
    phrases = _PHRASES.get(lang, _PHRASES["en"])

    return {
        "level": level,
        # Hearing-impaired: always buzz + flash; others escalate with level.
        "buzz": True if hearing_impaired else level >= 2,
        "buzz_strength": "strong" if (hearing_impaired or level >= 3) else "soft",
        "flash": True if hearing_impaired else level >= 2,
        "color": palette.get(level, palette[max(palette)]),
        "use_icons": colorsafe,
        "text": phrases.get(level, ""),
        "voice_clip": f"{lang}_level{level}.mp3",
        "language": lang,
        "simplified_ui": trainee,
        "sound": not hearing_impaired,
    }
