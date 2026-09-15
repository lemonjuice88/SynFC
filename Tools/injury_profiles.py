"""
injury_profiles.py — research-grounded injury type reference
==================================================================
A STATIC reference table, NOT a trained model -- deliberately, after
concluding a single club's Transfermarkt injury history (maybe 50-100
records) is nowhere near enough data to train anything meaningful
without massive overfitting. Instead, this maps common injury-type
KEYWORDS (as they tend to appear in Transfermarkt's free-text injury
names) to typical duration ranges and recurrence/chronic risk levels,
grounded in real sports-medicine research (see SOURCES below) rather
than invented numbers.

Used to ENRICH tools.get_player_injury_history()'s real, per-player
injury records with this reference context -- e.g. "this specific
injury lasted 22 days" (real data) + "this injury TYPE typically has a
high recurrence risk" (research-grounded context) together, not
conflated.

SOURCES (paraphrased, not directly quoted -- see individual profile
notes for which finding backs which number):
- Ekstrand et al. 2011, UEFA Club Injury Study (Br J Sports Med) --
  24 clubs, 8 seasons, 4,983 injuries.
- Whalan et al., hip/groin injury recurrence in football (Knee Surg
  Sports Traumatol Arthrosc).
- UCHealth summary of professional soccer injury research (ankle
  sprain recurrence/chronic instability).
- Calf muscle strain recurrence study, Soft Tissue Injury Registry,
  Australian Football League, 2014-2023.
- General orthopedic/sports-medicine consensus for fracture/
  dislocation/tendinopathy recovery windows (widely-cited clinical
  ranges, not a single specific study).

Caveat: these are POPULATION-LEVEL averages from research literature,
not predictions for any specific player -- a real player's recovery
can differ significantly. Frame outputs as "typical for this injury
type" context, not a diagnosis or guarantee.
"""

from typing import Optional

# Each profile: (min_days, max_days) typical duration, recurrence_risk
# (chance of the SAME injury happening again within ~1-2 years),
# chronic_risk (chance it becomes a recurring/long-term issue), and a
# short note on where the numbers come from.
INJURY_PROFILES = {
    "hamstring": {
        "keywords": ["hamstring"],
        "duration_days_range": (10, 28),
        "recurrence_risk": "medium-high",
        "chronic_risk": "medium",
        "note": "UEFA Club Injury Study: ~15 days average absence, 16% recurrence rate.",
    },
    "groin_adductor": {
        "keywords": ["groin", "adductor"],
        "duration_days_range": (7, 42),
        "recurrence_risk": "high",
        "chronic_risk": "high",
        "note": "52% of players with a longstanding (28+ day) groin injury had another "
                "groin injury the following preseason; 73% of those were recurrent, 27% chronic.",
    },
    "ankle": {
        "keywords": ["ankle"],
        "duration_days_range": (14, 21),
        "recurrence_risk": "high",
        "chronic_risk": "high",
        "note": "Highest recurrence rate of any lower-limb injury in pro soccer; "
                "about half of players develop chronic ankle instability.",
    },
    "calf": {
        "keywords": ["calf"],
        "duration_days_range": (10, 21),
        "recurrence_risk": "medium-high",
        "chronic_risk": "medium",
        "note": "13-21% recurrence within 2 years; recurrences average ~35.6 days lost.",
    },
    "knee_cruciate": {
        "keywords": ["cruciate", "acl", "meniscus"],
        "duration_days_range": (150, 270),
        "recurrence_risk": "medium",
        "chronic_risk": "medium",
        "note": "Long recovery is well-established clinically; treat this range as "
                "general orthopedic consensus, not a single specific study citation.",
    },
    "knee_other": {
        "keywords": ["knee"],
        "duration_days_range": (14, 60),
        "recurrence_risk": "medium",
        "chronic_risk": "medium",
        "note": "Generic knee injury -- wide range since severity varies a lot; "
                "check for a more specific match (cruciate/meniscus) first.",
    },
    "thigh_muscle": {
        "keywords": ["thigh", "quadriceps", "quad"],
        "duration_days_range": (7, 19),
        "recurrence_risk": "medium",
        "chronic_risk": "low-medium",
        "note": "Non-contact thigh muscle injuries average ~18.5 days, "
                "contact/contusion-type average ~7.5 days.",
    },
    "fracture": {
        "keywords": ["fracture", "broken"],
        "duration_days_range": (42, 56),
        "recurrence_risk": "low",
        "chronic_risk": "low",
        "note": "General clinical consensus for bone fracture healing; not a "
                "recurrence-prone injury type in the same way soft-tissue injuries are.",
    },
    "dislocation": {
        "keywords": ["dislocation", "dislocated"],
        "duration_days_range": (28, 42),
        "recurrence_risk": "medium-high",
        "chronic_risk": "medium",
        "note": "Recurrent dislocations are a documented clinical pattern, "
                "sometimes requiring surgical intervention if they repeat.",
    },
    "achilles_tendon": {
        "keywords": ["achilles"],
        "duration_days_range": (60, 270),
        "recurrence_risk": "high",
        "chronic_risk": "high",
        "note": "Ranges hugely by severity -- tendinopathy (overuse, chronic by "
                "nature) vs. full rupture (months-long recovery) are very different; "
                "this range spans both, treat cautiously.",
    },
    "tendinopathy_overuse": {
        "keywords": ["tendinopathy", "patellar tendon", "shin splint", "overuse"],
        "duration_days_range": (14, 90),
        "recurrence_risk": "high",
        "chronic_risk": "high",
        "note": "Overuse injuries are chronic/recurring by definition -- tied to "
                "training load, tend to resurface without load management changes.",
    },
    "concussion": {
        "keywords": ["concussion", "head injury"],
        "duration_days_range": (7, 21),
        "recurrence_risk": "medium",
        "chronic_risk": "medium",
        "note": "Return-to-play protocols typically require a minimum ~1-2 week "
                "graduated process; exact timelines vary by federation protocol.",
    },
}

_UNKNOWN_PROFILE = {
    "duration_days_range": None,
    "recurrence_risk": "unknown",
    "chronic_risk": "unknown",
    "note": "No matching reference profile for this injury description -- "
            "reason from the real recorded duration alone, don't guess at typical patterns.",
}


def classify_injury(injury_name: str) -> dict:
    """Matches a free-text injury name (as it tends to appear in
    Transfermarkt's injury history, e.g. "Hamstring injury", "Torn
    muscle fibre in groin") against the reference profiles above.

    Returns the matched profile dict (with a "category" key added), or
    an "unknown" fallback if nothing matches -- callers should NOT
    invent a plausible-sounding profile when this happens.
    """
    if not injury_name:
        return {"category": "unknown", **_UNKNOWN_PROFILE}

    name_lower = injury_name.lower()

    # Check more specific categories before generic ones (e.g. "cruciate"
    # before generic "knee") -- dict insertion order above is deliberate.
    for category, profile in INJURY_PROFILES.items():
        if any(keyword in name_lower for keyword in profile["keywords"]):
            return {"category": category, **{k: v for k, v in profile.items() if k != "keywords"}}

    return {"category": "unknown", **_UNKNOWN_PROFILE}


def enrich_injury_history(injuries: list) -> list:
    """Takes the `injuries` list from tools.get_player_injury_history()
    (each item: {"injury": ..., "from": ..., "until": ..., "days_missed": ...})
    and adds a research-grounded profile to each entry, WITHOUT altering
    the real recorded data.
    """
    enriched = []
    for injury in injuries:
        profile = classify_injury(injury.get("injury", ""))
        enriched.append({**injury, "reference_profile": profile})
    return enriched


if __name__ == "__main__":
    # Quick manual test:
    #   python injury_profiles.py "Hamstring injury"
    import sys
    import json

    name = " ".join(sys.argv[1:]) or "Hamstring injury"
    print(json.dumps(classify_injury(name), indent=2, ensure_ascii=False))