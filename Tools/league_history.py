"""
league_history_scraper.py — per-club league position history + market value
================================================================================
Scrapes a club's historical final league placement (one position per
season, going back `num_years` seasons) directly from Transfermarkt,
plus a current squad market value (reused from tools.get_club_squad --
sum of listed players' market values, as a practical proxy for "current
club value" without a second, separate scraper).

Built for training-data collection (league_position_playground.py) --
NOT tested against Transfermarkt's live page structure from this
environment (same caveat as every other Transfermarkt tool in this
project: the "Platzierungen" (placements) URL pattern below is a
best-effort guess, test with ONE club first and report back the actual
column names if it comes back empty/wrong -- same debug loop we've used
for the squad/transfer-history scrapers).
"""

import datetime
import io
import re
import time
from typing import List, Optional

import pandas as pd

import tools


def get_club_league_history(club_name: str, num_years: int = 21) -> dict:
    """Scrapes a club's final league placement for each of the last
    `num_years` seasons, plus a current total squad market value.

    Example return:
        {"found": True, "club": "Liverpool FC",
         "history": [{"season": 2005, "position": 5}, ..., {"season": 2025, "position": 2}],
         "current_market_value_eur": 987000000.0,
         "seasons_failed": [2013]}
    """
    club_url = tools.search_club(club_name)
    if club_url is None:
        return {"found": False, "reason": f"No Transfermarkt result found for club '{club_name}'."}

    # Guessed URL pattern -- Transfermarkt's "Platzierungen" (placements)
    # page for a club, listing one final league position per season.
    # UNTESTED against the live site -- verify with one club first.
    placements_url = club_url.replace("/startseite/verein/", "/platzierungen/verein/")

    try:
        soup = tools._get_soup(placements_url)
        tables = pd.read_html(io.StringIO(str(soup)))
    except Exception as e:
        return {"found": False, "reason": f"Error while reading the placements page: {e}"}

    history = []
    for t in tables:
        cols = [str(c) for c in t.columns]
        season_col = next((c for c in cols if c.lower() == "season"), None)

        # NOTE on the column shift below: Transfermarkt's placements
        # table has a "League Level" column that's often empty/icon-only
        # in the real HTML, which shifts every column AFTER it one slot
        # to the right when pandas.read_html parses it -- so the column
        # literally labeled "Points" actually holds the real final
        # league position, and the column labeled "Rank" actually holds
        # the manager's name. Verified against real known seasons
        # (Liverpool 24/25: 25W-9D-4L, 84pts, rank 1, Arne Slot; 19/20:
        # 32W-3D-3L, 99pts, rank 1, Klopp -- both matched this exact
        # shift). If this stops matching for older seasons / other
        # clubs, re-verify against a season you know the real table for.
        position_col = next((c for c in cols if c.lower() == "points"), None)

        if not season_col or not position_col:
            continue

        for _, row in t.iterrows():
            season_raw = str(row.get(season_col, "")).strip()
            position_raw = str(row.get(position_col, "")).strip()

            # Season format is "25/26", "24/25" etc -- two-digit
            # start-year. Same century-inference trick used earlier in
            # this project for the finance notebook's transfer_season column.
            season_match = re.match(r"^(\d{2})/\d{2}$", season_raw)
            position_match = re.match(r"^(\d{1,2})$", position_raw)
            if not season_match or not position_match:
                continue

            two_digit_year = int(season_match.group(1))
            season_year = 2000 + two_digit_year if two_digit_year <= 30 else 1900 + two_digit_year

            position = int(position_match.group(1))
            if not (1 <= position <= 24):  # sanity check -- a real final position, not stray data
                continue

            # Sanity check: a season can't start in the future -- if it
            # does, this row is almost certainly a parsing artifact
            # (e.g. a summary/total row at the table's edge), not a
            # real season. Cap at current year + 1 (allows the
            # currently-in-progress season, e.g. 25/26 in 2026).
            current_year = datetime.date.today().year
            if season_year > current_year + 1:
                continue

            history.append({"season": season_year, "position": position})

    if not history:
        return {
            "found": False,
            "reason": "No placement rows parsed -- page structure likely differs from what was assumed. "
                      f"Tables found on page: {[list(t.columns) for t in tables]}",
        }

    history.sort(key=lambda h: h["season"])
    history = history[-num_years:]

    # Reuse the already-working squad tool for a "current value" proxy,
    # instead of writing a second, separate scraper for it.
    squad_data = tools.get_club_squad(club_name)
    current_value = None
    if squad_data.get("found"):
        total = 0.0
        for p in squad_data.get("players", []):
            parsed = _parse_market_value(p.get("market_value", ""))
            if parsed is not None:
                total += parsed
        current_value = total if total > 0 else None

    return {
        "found": True,
        "club": club_name,
        "history": history,
        "current_market_value_eur": current_value,
    }


def _parse_market_value(text: str) -> Optional[float]:
    """Same "€25.00m" -> 25000000.0 style parsing as calculator.py's
    parse_money_string_to_eur -- duplicated here (not imported) to keep
    this file's only dependency on the rest of the project as `tools`."""
    if not text:
        return None
    cleaned = text.strip().lower().replace("€", "").replace(",", "").strip()
    if cleaned in ("", "-", "unknown", "nan", "n/a"):
        return None
    multiplier = 1.0
    if cleaned.endswith("bn"):
        multiplier, cleaned = 1_000_000_000, cleaned[:-2]
    elif cleaned.endswith("m"):
        multiplier, cleaned = 1_000_000, cleaned[:-1]
    elif cleaned.endswith("k"):
        multiplier, cleaned = 1_000, cleaned[:-1]
    try:
        return float(cleaned) * multiplier
    except ValueError:
        return None


if __name__ == "__main__":
    # Quick manual test:
    #   python league_history_scraper.py "Liverpool"
    import sys
    import json

    club = " ".join(sys.argv[1:]) or "Liverpool"
    result = get_club_league_history(club)
    print(json.dumps(result, indent=2, ensure_ascii=False))