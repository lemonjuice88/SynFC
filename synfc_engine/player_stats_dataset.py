"""
player_stats_dataset.py — local top-5-league player stats (2024/25)
========================================================================
A local, offline replacement/supplement for FBref: a Kaggle dataset
covering every player in the top 5 European leagues (Premier League,
La Liga, Ligue 1, Bundesliga, Serie A) for the 2024/25 season, with
per-90 stats, xG/xA/xG-chain/xG-buildup, etc.

Why this exists: FBref's bot protection has blocked every scraping
approach tried so far (cloudscraper, curl_cffi with TLS impersonation,
session warm-up + Referer). This dataset has no such problem -- it's a
plain local CSV, no network request, no bot detection, instant lookup.
The obvious tradeoff: it's a snapshot (as of when it was scraped from
FBref, ~2 months old at time of writing), not live data, and it only
covers the current 2024/25 season in the top 5 leagues -- a player in
a different league/season won't be found here.

IMPORTANT — name matching contract:
This module does NOT do typo correction (e.g. it will NOT figure out
that "UgoChokwou" means "Ugochukwu"). That correction already happens
upstream in core.extract_player_names() (an LLM call that fixes
misspellings before any tool is ever called). By the time a name
reaches find_player_stats() here, it's expected to already be the
corrected, real spelling -- this function only does light normalization
(case-insensitive, accent-insensitive) on top of that, e.g. matching
"Bruno Guimaraes" (no accent) against the dataset's "Bruno Guimarães".

Setup: just needs pandas (already a dependency via tools.py). Place the
CSV at data/Top_5_European_Leagues_2024_25_Complete_Player_Stats.csv
relative to this file, or point DATASET_PATH elsewhere.
"""

import os
import unicodedata
from typing import List, Optional

import pandas as pd

# This file lives in synfc_engine/, but data/ is now a SIBLING folder
# (SynFC/data/), not a child of synfc_engine/ -- so we go up one
# level (to the project root) before looking for it.
_ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))       # .../SynFC/synfc_engine
_PROJECT_ROOT = os.path.dirname(_ENGINE_DIR)                    # .../SynFC

DATASET_PATH = os.environ.get(
    "PLAYER_STATS_CSV_PATH",
    os.path.join(_PROJECT_ROOT, "data",
                 "Top_5_European_Leagues_2024_25_Complete_Player_Stats.csv"),
)

_df = None  # loaded lazily, once, and cached (see _get_dataframe)


def _get_dataframe() -> Optional[pd.DataFrame]:
    global _df
    if _df is None:
        if not os.path.isfile(DATASET_PATH):
            return None
        _df = pd.read_csv(DATASET_PATH)
        _df["_normalized_player"] = _df["player"].apply(_normalize_name)
    return _df


def _normalize_name(name: str) -> str:
    """Case-insensitive, accent-insensitive normalization -- turns
    "Bruno Guimarães" and "bruno guimaraes" into the same comparable
    string. NOT a typo corrector (see module docstring)."""
    if not name:
        return ""
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii")
    return ascii_only.lower().strip()


def find_player_stats(player_name: str) -> dict:
    """Looks up a player's 2024/25 top-5-league season stats by name.
    Expects an already name-corrected `player_name` (see module
    docstring) -- matching here is exact after normalization, not
    fuzzy/typo-tolerant.

    If the player appears in multiple rows (mid-season transfer, or a
    data-quality duplicate in the source dataset), ALL matching rows
    are returned as separate "stints" rather than silently summed --
    summing could double-count genuine duplicate-entry artifacts in the
    source data, so it's left to the caller/persona to interpret.

    Example return (single stint):
        {"found": True, "name": "Bruno Guimarães", "stints": [
            {"team": "Newcastle United", "league": "ENG-Premier League",
             "position": "MF", "age": 24.0, "minutes": "2450",
             "goals": "3", "assists": "5", "xg": "2.8", "xa": "4.1",
             "xg_per90": "0.10", "key_passes": "38", ...}
        ]}

    If nothing matches:
        {"found": False, "reason": "..."}
    """
    df = _get_dataframe()
    if df is None:
        return {
            "found": False,
            "reason": f"Dataset file not found at '{DATASET_PATH}'. Set "
                      f"PLAYER_STATS_CSV_PATH or place the CSV in the expected location.",
        }

    target = _normalize_name(player_name)
    if not target:
        return {"found": False, "reason": "Empty player name given."}

    matches = df[df["_normalized_player"] == target]
    if matches.empty:
        return {
            "found": False,
            "reason": (
                f"'{player_name}' not found in the top-5-league 2024/25 dataset "
                f"(player may be in a different league, a different season, or "
                f"the name doesn't match exactly even after normalization)."
            ),
        }

    stints = []
    for _, row in matches.iterrows():
        stints.append({
            "team": row.get("team", "unknown"),
            "league": row.get("league", "unknown"),
            "position": row.get("pos", "unknown"),
            "age": row.get("age", "unknown"),
            "minutes": row.get("Playing Time_Min", "unknown"),
            "matches_played": row.get("Playing Time_MP", "unknown"),
            "starts": row.get("Playing Time_Starts", "unknown"),
            "goals": row.get("Performance_Gls", "unknown"),
            "assists": row.get("Performance_Ast", "unknown"),
            "goals_plus_assists": row.get("Performance_G+A", "unknown"),
            "yellow_cards": row.get("Performance_CrdY", "unknown"),
            "red_cards": row.get("Performance_CrdR", "unknown"),
            "goals_per90": row.get("Per 90 Minutes_Gls", "unknown"),
            "assists_per90": row.get("Per 90 Minutes_Ast", "unknown"),
            "xg": row.get("xg", "unknown"),
            "xa": row.get("xa", "unknown"),
            "np_xg": row.get("np_xg", "unknown"),
            "key_passes": row.get("key_passes", "unknown"),
            "xg_chain": row.get("xg_chain", "unknown"),
            "xg_buildup": row.get("xg_buildup", "unknown"),
        })

    return {
        "found": True,
        "name": player_name,
        "stints": stints,
    }


if __name__ == "__main__":
    # Quick manual test:
    #   python player_stats_dataset.py "Bruno Guimaraes"
    import sys
    import json

    name = " ".join(sys.argv[1:]) or "Bruno Guimaraes"
    result = find_player_stats(name)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))