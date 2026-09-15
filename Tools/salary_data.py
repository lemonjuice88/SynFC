"""
salary_data.py — real player salary data via SalaryLeaks
==============================================================
Fills the long-standing gap noted in project docs: no wage/salary data
source existed anywhere in the project (Capology's official pages were
too bot-protected to scrape reliably from this environment). SalaryLeaks
publishes the same category of data (weekly/annual wage, bonus,
contract length) via plain, unprotected HTML tables -- no login, no
bot-blocking encountered.

Setup: no API key needed, just curl_cffi (already a project dependency).
"""

import io
import re
import time
from functools import lru_cache
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup

from curl_cffi import requests as cffi_requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
}
REQUEST_DELAY_SECONDS = 2


def _slugify(name: str) -> str:
    """Best-effort club-name -> URL-slug conversion (e.g. "Manchester City"
    -> "manchester-city"). SalaryLeaks' actual slugs mostly follow this
    pattern, but a few clubs use a different official name on the site
    (e.g. "Tottenham Hotspurs", with an extra "s") -- if a lookup fails,
    that's the first thing to check/adjust manually."""
    slug = name.strip().lower()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug)
    return slug


def _get_html(url: str) -> str:
    resp = cffi_requests.get(url, headers=HEADERS, timeout=15, impersonate="chrome124")
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY_SECONDS)
    return resp.text


@lru_cache(maxsize=64)
def get_club_salaries(club_name: str, slug_override: Optional[str] = None) -> dict:
    """Fetches a club's full salary table from SalaryLeaks.

    Example return:
        {"found": True, "club": "Liverpool", "total_wage_bill": "£204.7M",
         "players": [{"name": "Virgil van Dijk", "weekly": "£350,000",
                       "annual": "£18.2M", "bonus": "+£5.2M", "age": "35",
                       "contract_until": "2027"}, ...]}
    """
    slug = slug_override or _slugify(club_name)
    url = f"https://www.salaryleaks.com/football/teams/{slug}"

    try:
        html = _get_html(url)
    except Exception as e:
        return {
            "found": False,
            "reason": f"Error fetching SalaryLeaks page for '{club_name}' (tried slug '{slug}'): {e}",
        }

    try:
        tables = pd.read_html(io.StringIO(html))
    except Exception as e:
        return {"found": False, "reason": f"No parseable table found on page: {e}"}

    # The page repeats the same salary table twice (desktop view with a
    # "#" rank column, and a simplified duplicate) -- prefer the one
    # with a "#" column since it's the more complete version.
    target_table = None
    for t in tables:
        cols = [str(c) for c in t.columns]
        if any(c.strip() == "#" for c in cols) and "Player" in cols:
            target_table = t
            break
    if target_table is None:
        for t in tables:
            if "Player" in [str(c) for c in t.columns]:
                target_table = t
                break

    if target_table is None:
        return {"found": False, "reason": "Salary table structure not found/changed -- inspect the page manually."}

    players = []
    for _, row in target_table.iterrows():
        name = str(row.get("Player", "")).strip()
        if not name or name.lower() in ("nan", "total"):
            continue
        players.append({
            "name": name,
            "weekly": str(row.get("Weekly", "unknown")),
            "annual": str(row.get("Annual", "unknown")),
            "bonus": str(row.get("Bonus", "unknown")),
            "age": str(row.get("Age", "unknown")),
            "contract_until": str(row.get("Contract", "unknown")),
        })

    if not players:
        return {"found": False, "reason": "Table found but no player rows parsed -- check column names."}

    return {"found": True, "club": club_name, "players": players, "source_url": url}


def get_player_salary_via_club(player_name: str, club_name: str) -> dict:
    """Convenience wrapper: pulls the whole club table (cached) and
    filters to one player -- avoids a second scrape if you already
    fetched the club's table for something else this run. Use
    search_player_salary() instead if you don't already know the club."""
    club_data = get_club_salaries(club_name)
    if not club_data.get("found"):
        return club_data

    name_lower = player_name.lower()
    for p in club_data["players"]:
        if name_lower in p["name"].lower():
            return {"found": True, "player": p, "club": club_name}

    return {"found": False, "reason": f"'{player_name}' not found in {club_name}'s salary table."}


def _extract_label_value(lines: list, label: str) -> Optional[str]:
    """SalaryLeaks' player page renders fields like "Position" / "Age" /
    "Nationality" as a label on its own line, followed by the value on
    the next non-empty line -- this walks the page's plain-text lines
    looking for that pattern. Best-effort: if the page's layout differs
    from this, it just returns None rather than a wrong guess."""
    for i, line in enumerate(lines):
        if line.strip() == label:
            for next_line in lines[i + 1:]:
                if next_line.strip():
                    return next_line.strip()
    return None


@lru_cache(maxsize=128)
def search_player_salary(player_name: str, slug_override: Optional[str] = None) -> dict:
    """Direct player lookup -- NO club needed first (unlike
    get_player_salary_via_club). Fetches SalaryLeaks' own per-player
    page directly.

    Example return:
        {"found": True, "name": "Virgil van Dijk", "position": "Centre-Back (CB)",
         "age": "35", "nationality": "Dutch", "current_club": "Liverpool",
         "current_weekly": "£350,000", "current_annual": "£18.2M",
         "summary": "Virgil van Dijk signed 2 years contract at Liverpool
         until 2027. His base salary is £18.2 million per-year (£350,000/week).",
         "contract_history": [{"date": "2025", "weekly": "£350,000",
         "yearly": "£18.2M", "team": "Liverpool"}, ...]}
    """
    slug = slug_override or _slugify(player_name)
    url = f"https://www.salaryleaks.com/football/{slug}"

    try:
        html = _get_html(url)
    except Exception as e:
        return {"found": False, "reason": f"Error fetching SalaryLeaks page for '{player_name}' (tried slug '{slug}'): {e}"}

    soup = BeautifulSoup(html, "lxml")

    # The <meta name="description"> tag is a reliable, human-written
    # one-sentence summary present on every player page -- cheap and
    # sturdy compared to guessing at the surrounding div structure.
    summary = None
    meta_tag = soup.find("meta", attrs={"name": "description"})
    if meta_tag and meta_tag.get("content"):
        summary = meta_tag["content"].strip()

    if not summary:
        # No description meta tag found at all -- the page likely doesn't
        # exist / the slug guess was wrong, rather than a parsing issue.
        return {"found": False, "reason": f"No player page content found at {url} -- slug '{slug}' may be wrong."}

    text_lines = soup.get_text("\n").split("\n")
    position = _extract_label_value(text_lines, "Position")
    age = _extract_label_value(text_lines, "Age")
    nationality = _extract_label_value(text_lines, "Nationality")

    contract_history = []
    current_club = None
    current_weekly = None
    current_annual = None
    try:
        tables = pd.read_html(io.StringIO(html))
        for t in tables:
            cols = [str(c) for c in t.columns]
            if "Date" in cols and "Weekly" in cols and "Yearly" in cols and "Team" in cols:
                for _, row in t.iterrows():
                    contract_history.append({
                        "date": str(row.get("Date", "unknown")),
                        "weekly": str(row.get("Weekly", "unknown")),
                        "yearly": str(row.get("Yearly", "unknown")),
                        "team": str(row.get("Team", "unknown")),
                    })
                break  # first matching table is the more detailed one (has a Source column too)
    except Exception:
        pass  # contract_history stays empty -- summary/meta-description is still returned

    if contract_history:
        current_club = contract_history[0]["team"]
        current_weekly = contract_history[0]["weekly"]
        current_annual = contract_history[0]["yearly"]

    return {
        "found": True,
        "name": player_name,
        "position": position or "unknown",
        "age": age or "unknown",
        "nationality": nationality or "unknown",
        "current_club": current_club or "unknown",
        "current_weekly": current_weekly or "unknown",
        "current_annual": current_annual or "unknown",
        "summary": summary,
        "contract_history": contract_history,
        "source_url": url,
    }


def _run_cli(argv: list) -> dict:
    """The actual CLI decision logic, pulled into its own function so
    it's directly testable without needing to exec the __main__ block."""
    if len(argv) > 1 and argv[1] == "club":
        club = argv[2] if len(argv) > 2 else "Liverpool"
        if len(argv) > 3:
            return get_player_salary_via_club(argv[3], club)
        return get_club_salaries(club)

    # Default mode: everything after the script name IS the player
    # query, no "player" prefix needed -- this is the common case.
    player = " ".join(argv[1:]) if len(argv) > 1 else "Virgil van Dijk"
    return search_player_salary(player)


if __name__ == "__main__":
    # Quick manual test:
    #   python salary_data.py Osimhen                  -> direct player lookup (default mode)
    #   python salary_data.py Virgil van Dijk           -> multi-word names work too, no quotes needed
    #   python salary_data.py club Liverpool            -> whole club's table
    #   python salary_data.py club Liverpool "Van Dijk" -> one player, via the club's table
    import sys
    import json

    result = _run_cli(sys.argv)
    print(json.dumps(result, indent=2, ensure_ascii=False)[:3000])