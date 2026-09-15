"""
transfer_history.py — self-scraped club transfer history
=============================================================
Scrapes a club's historical transfer record (incoming and/or outgoing)
DIRECTLY from Transfermarkt, season by season, instead of depending on
a third-party pre-built dataset.

Why this exists: the Kaggle dataset we were using before
(davidcariboo/player-scores) auto-updates weekly, and a real player
transfer that showed up in one download had silently vanished in the
next -- with no warning, no version pin, nothing. This module trades
"someone else already built a giant global dataset" for "we control
exactly when the snapshot is taken, and we only pull the club(s) we
actually care about" -- a much smaller scraping job than what the
Kaggle maintainer's pipeline does (400+ clubs, 60k+ games), so it's a
reasonable trade.

This is a separate file from tools.py (rather than another function
in it) because it's a different USAGE PATTERN: tools.py's functions
are "look up one player/club right now, on demand, as part of a live
conversation". This one is "pull a club's multi-year transfer history
occasionally, to build/refresh an offline analysis dataset" -- more of
a batch job than a live-conversation tool.

It reuses tools.py's already-proven scraping primitives (the curl_cffi
session, search_club, and the name/position-splitting fix) rather than
duplicating them.

The exact column names assumed below (Left/Joined/Fee/Market value)
are a best-effort guess at Transfermarkt's real transfer-page table
structure -- I couldn't verify them against the live site from this
environment. Test with a single season first
(get_club_transfer_history("Liverpool", 2024, 2024)) and report back
the actual output if columns come back as "unknown", same test-and-fix
loop used for every other Transfermarkt tool in this project.

Setup: no new dependencies beyond what tools.py already needs
(curl_cffi, beautifulsoup4, lxml, pandas) -- this file just imports
tools.py for its scraping primitives.
"""

import io
from typing import Optional

import pandas as pd

import tools


def get_club_transfer_history(
    club_name: str, start_year: int, end_year: int, direction: str = "arrivals"
) -> dict:
    """Scrapes a club's transfer history (incoming by default) season by
    season, directly from Transfermarkt's own club transfer page.

    Args:
        club_name: the club to scrape.
        start_year: first season to include (e.g. 2016 for the 2016/17 season).
        end_year: last season to include (inclusive).
        direction: "arrivals" (incoming/buys) or "departures" (outgoing/sells).

    Example return:
        {"found": True, "club": "Liverpool FC", "direction": "arrivals",
         "seasons_covered": "2016-2025",
         "transfers": [{"player": "...", "position": "...", "age": "...",
                          "market_value": "...", "fee": "...",
                          "other_club": "...", "season": 2024}, ...],
         "seasons_failed": [2019],
         "source_url_pattern": "https://www.transfermarkt.com/.../transfers/verein/31/saison_id/<year>"}
    """
    if direction not in ("arrivals", "departures"):
        return {"found": False, "reason": "direction must be 'arrivals' or 'departures'"}

    club_url = tools.search_club(club_name)
    if club_url is None:
        return {"found": False, "reason": f"No Transfermarkt result found for club '{club_name}'."}

    # club_url looks like https://www.transfermarkt.com/{slug}/startseite/verein/{id}
    transfers_base_url = club_url.replace("/startseite/verein/", "/transfers/verein/")

    all_transfers = []
    seasons_failed = []

    for year in range(start_year, end_year + 1):
        season_url = f"{transfers_base_url}/saison_id/{year}"
        try:
            soup = tools._get_soup(season_url)
            tables = pd.read_html(io.StringIO(str(soup)))
        except Exception:
            seasons_failed.append(year)
            continue

        # Transfermarkt's transfer page typically shows TWO tables per
        # season -- arrivals and departures -- but we don't rely on
        # table order being guaranteed. Instead we look at each table's
        # own column names to guess which one it is. Best-effort
        # heuristic, may need adjusting once tested against the real page.
        for t in tables:
            cols = [str(c) for c in t.columns]
            if not any("Player" in c or "Name" in c for c in cols):
                continue

            is_arrivals_table = any("left" in c.lower() or "joined from" in c.lower() for c in cols)
            is_departures_table = (
                any("joined" in c.lower() and "from" not in c.lower() for c in cols)
                or any("left to" in c.lower() for c in cols)
            )

            if direction == "arrivals" and is_departures_table and not is_arrivals_table:
                continue
            if direction == "departures" and is_arrivals_table and not is_departures_table:
                continue

            if hasattr(t.columns, "droplevel") and t.columns.nlevels > 1:
                t.columns = t.columns.get_level_values(-1)

            for _, row in t.iterrows():
                raw_name = str(row.get("Player", row.get("Name", "unknown"))).strip()
                if not raw_name or raw_name.lower() == "nan":
                    continue

                fee_col = next((c for c in t.columns if "fee" in str(c).lower()), None)
                mv_col = next(
                    (c for c in t.columns if "market" in str(c).lower() or "value" in str(c).lower()), None
                )
                club_col = next(
                    (c for c in t.columns if "left" in str(c).lower() or "joined" in str(c).lower()
                     or "club" in str(c).lower()),
                    None,
                )
                age_col = next((c for c in t.columns if str(c).strip().lower() == "age"), None)

                # Phantom-row filter (same fix as get_club_squad's earlier
                # bug): Transfermarkt stacks name + position + club + league
                # across multiple lines within ONE cell, and pandas.read_html
                # splits that into several "leftover" rows -- only the real
                # row carries actual data in age/fee/market value, the rest
                # are empty fragments (e.g. a whole row where "Player" is
                # just the league name that leaked out of the club cell).
                # We keep only the row that carries at least one real value.
                def _has_real_value(col):
                    if not col:
                        return False
                    v = row.get(col)
                    return not (v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip().lower() in ("", "nan"))

                if not (_has_real_value(age_col) or _has_real_value(fee_col) or _has_real_value(mv_col)):
                    continue

                name, position = tools._split_name_position(raw_name)

                all_transfers.append({
                    "player": name,
                    "position": position,
                    "age": str(row.get(age_col, "unknown")) if age_col else "unknown",
                    "market_value": str(row.get(mv_col, "unknown")) if mv_col else "unknown",
                    "fee": str(row.get(fee_col, "unknown")) if fee_col else "unknown",
                    "other_club": str(row.get(club_col, "unknown")) if club_col else "unknown",
                    "season": year,
                })

    return {
        "found": True,
        "club": club_name,
        "direction": direction,
        "seasons_covered": f"{start_year}-{end_year}",
        "transfers": all_transfers,
        "seasons_failed": seasons_failed,
        "source_url_pattern": transfers_base_url + "/saison_id/<year>",
    }


if __name__ == "__main__":
    # Quick manual test:
    #   python transfer_history.py "Liverpool" 2024 2024
    #   python transfer_history.py "Liverpool" 2016 2025 departures
    import sys
    import json

    club = sys.argv[1] if len(sys.argv) > 1 else "Liverpool"
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 2024
    end = int(sys.argv[3]) if len(sys.argv) > 3 else start
    direction = sys.argv[4] if len(sys.argv) > 4 else "arrivals"

    result = get_club_transfer_history(club, start, end, direction=direction)
    print(json.dumps(result, indent=2, ensure_ascii=False))