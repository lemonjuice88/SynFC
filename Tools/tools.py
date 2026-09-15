"""
SynFC — External data tools
==============================
Helper functions that let the CFO and Analyst roles ground their opinions
in real data instead of speaking off the cuff. Everything here scrapes
Transfermarkt (no official API exists):

  - get_player_financials(name)   -> market value / contract / current club
  - get_player_positions(name)    -> main + secondary playing position(s)
  - get_club_squad(club, filter)  -> a club's squad, optionally filtered
                                      to players in one position

IMPORTANT NOTES:
- Transfermarkt (and FBref, kept below but currently unused) don't offer
  an official API, so we scrape HTML. This is fragile: if a site changes
  its markup, this code can break, and the exact table/column names below
  are a best effort that HAVE NOT been verified against the live site
  from this environment (network access here is restricted) — test
  locally and report back the real column names if parsing fails.
- Neither site is happy about heavy/automated bulk scraping. These
  functions are meant for "look up one player / one club", NOT for
  crawling. REQUEST_DELAY_SECONDS adds a deliberate pause after each
  request.
- Respecting each site's Terms of Service is your responsibility; check
  before using this in anything commercial or high-volume.

Setup:
    pip install curl_cffi beautifulsoup4 lxml pandas
"""

import os
import re
import time
import io
from functools import lru_cache
from typing import Optional, List

import pandas as pd

# curl_cffi impersonates a real browser at the TLS/JA3 fingerprint level
# (not just HTTP headers), which is what actually got past FBref's bot
# protection when a plain requests/cloudscraper approach didn't.
from curl_cffi import requests as cffi_requests
from bs4 import BeautifulSoup, Comment

HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

REQUEST_DELAY_SECONDS = 2  # be polite: pause after every request

# Transfermarkt frequently shows a player's name and position stacked in
# the SAME table cell (name on one line, position on the next). Several
# scraped tables below need to split a combined "Name Position" string
# back into its two parts using this list of known position labels.
KNOWN_POSITIONS = sorted([
    "Goalkeeper",
    "Sweeper",
    "Centre-Back", "Left-Back", "Right-Back",
    "Defensive Midfield", "Central Midfield", "Attacking Midfield",
    "Right Midfield", "Left Midfield",
    "Left Winger", "Right Winger",
    "Second Striker", "Centre-Forward",
], key=len, reverse=True)


def _split_name_position(combined: str):
    combined = combined.strip()
    for pos in KNOWN_POSITIONS:
        if combined.endswith(pos):
            return combined[: -len(pos)].strip(), pos
    return combined, "unknown"


# Generic section/category header words that sometimes get parsed as if
# they were a player row (e.g. a table has a divider row reading just
# "Injuries" or "Suspended" to group entries below it).
_NON_PLAYER_ROW_WORDS = {
    "injuries", "injury", "suspended", "suspensions", "locked out",
    "personal reasons", "other", "illness", "ban", "bans",
}


def _get_soup(url: str, referer: Optional[str] = None) -> BeautifulSoup:
    headers = dict(HEADERS)
    if referer:
        headers["Referer"] = referer
    resp = cffi_requests.get(url, headers=headers, timeout=15, impersonate="chrome124")
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY_SECONDS)
    return BeautifulSoup(resp.text, "lxml")


# =======================================================================
# TRANSFERMARKT — player lookups
# =======================================================================
@lru_cache(maxsize=256)
def search_player(player_name: str) -> Optional[str]:
    """Uses Transfermarkt's own search engine and returns the first
    matching player's profile URL. Returns None if nothing is found.

    Instead of relying on a specific CSS class name (which changes
    often), this matches on the stable URL pattern "/profil/spieler/<id>".

    Cached (@lru_cache): multiple team modules often look up the SAME
    player within one conversation (e.g. analytics_team needs it for
    both position lookup and injury history) -- without this, that's
    two redundant round-trips (with their built-in politeness delays)
    to Transfermarkt for identical information.
    """
    query = player_name.replace(" ", "+")
    search_url = f"https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query={query}"
    soup = _get_soup(search_url)

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if "/profil/spieler/" in href:
            return "https://www.transfermarkt.com" + href if href.startswith("/") else href
    return None


def get_player_financials(player_name: str) -> dict:
    """For the CFO role: current market value, contract expiry, and club.

    Example return:
        {"found": True, "name": "...", "market_value": "€25.00m",
         "contract_expires": "Dec 31, 2026", "current_club": "...",
         "source_url": "..."}
    """
    try:
        profile_url = search_player(player_name)
        if profile_url is None:
            return {"found": False, "reason": f"No Transfermarkt result found for '{player_name}'."}

        soup = _get_soup(profile_url)

        mv_tag = soup.find("a", {"class": "data-header__market-value-wrapper"})
        market_value = mv_tag.get_text(strip=True).split("Last")[0].strip() if mv_tag else "unknown"

        contract_expires = "unknown"
        for span in soup.find_all("span", {"class": "data-header__label"}):
            if "Contract expires" in span.get_text():
                value_span = span.find("span", {"class": "data-header__content"})
                if value_span:
                    contract_expires = value_span.get_text(strip=True)
                break

        club_tag = soup.find("span", {"class": "data-header__club"})
        current_club = club_tag.get_text(strip=True) if club_tag else "unknown"

        return {
            "found": True,
            "name": player_name,
            "market_value": market_value,
            "contract_expires": contract_expires,
            "current_club": current_club,
            "source_url": profile_url,
        }
    except Exception as e:  # covers search_player() failing too, not just page parsing
        return {"found": False, "reason": f"Error while reading the Transfermarkt page: {e}"}


def get_player_positions(player_name: str) -> dict:
    """For the Analyst/Manager roles: which position(s) a player plays.

    Transfermarkt's profile header lists a "Position" (main) and,
    when applicable, "Other position(s)". We treat "main" as the most
    frequently played position and "other" as secondary ones — true
    per-match position frequency isn't exposed on this page, so this is
    an APPROXIMATION, not an exact appearance count.

    Example return:
        {"found": True, "name": "...", "main_position": "Centre-Forward",
         "other_positions": ["Second Striker"], "source_url": "..."}
    """
    try:
        profile_url = search_player(player_name)
        if profile_url is None:
            return {"found": False, "reason": f"No Transfermarkt result found for '{player_name}'."}

        soup = _get_soup(profile_url)

        main_position = "unknown"
        other_positions: List[str] = []

        position_label = soup.find(string=re.compile(r"^\s*Position\s*:?\s*$"))
        container = None
        if position_label:
            container = position_label.find_parent("li") or position_label.find_parent("span")

        if container is None:
            candidate = soup.find(string=re.compile("Position"))
            if candidate:
                container = candidate.find_parent("li") or candidate.find_parent("div")

        if container:
            text = container.get_text(" ", strip=True)
            parts = re.split(r"Other\s+position\(?s?\)?\s*:", text)
            main_part = re.sub(r"^Position\s*:?\s*", "", parts[0]).strip()
            main_position = main_part or "unknown"
            if len(parts) > 1:
                other_positions = [p.strip() for p in re.split(r",|/", parts[1]) if p.strip()]

        return {
            "found": True,
            "name": player_name,
            "main_position": main_position,
            "other_positions": other_positions,
            "source_url": profile_url,
        }
    except Exception as e:
        return {"found": False, "reason": f"Error while reading the Transfermarkt page: {e}"}


def get_player_injury_history(player_name: str) -> dict:
    """For the Analyst role: the TARGET PLAYER's own injury history (not
    the buying club's squad) -- e.g. is this specific player currently
    injured, and how injury-prone have they been recently.

    Transfermarkt exposes this at the player's "verletzungen" (injuries)
    subpage, at the same URL as the player's profile but with that path
    segment swapped in.

    Example return:
        {"found": True, "name": "...", "source_url": "...",
         "injuries": [{"injury": "Hamstring strain", "from": "Jun 2026",
                        "until": "Jul 2026", "days_missed": "21"}, ...]}
        (an empty "injuries" list means no injury history is listed --
        that's a valid, meaningful result, not a failure)
    """
    try:
        profile_url = search_player(player_name)
        if profile_url is None:
            return {"found": False, "reason": f"No Transfermarkt result found for '{player_name}'."}

        injury_url = profile_url.replace("/profil/spieler/", "/verletzungen/spieler/")

        soup = _get_soup(injury_url)
        import pandas as pd
        import io

        tables = pd.read_html(io.StringIO(str(soup)))
        injury_table = None
        for t in tables:
            cols = [str(c) for c in t.columns]
            if any("injury" in c.lower() for c in cols):
                injury_table = t
                break

        if injury_table is None:
            # No injury table can legitimately mean "no injury history listed".
            return {"found": True, "name": player_name, "injuries": [], "source_url": injury_url}

        if hasattr(injury_table.columns, "droplevel") and injury_table.columns.nlevels > 1:
            injury_table.columns = injury_table.columns.get_level_values(-1)

        injury_col = next((c for c in injury_table.columns if "injury" in c.lower()), None)
        from_col = next((c for c in injury_table.columns if c.strip().lower() == "from"), None)
        until_col = next((c for c in injury_table.columns if c.strip().lower() == "until"), None)
        days_col = next(
            (c for c in injury_table.columns if "days" in c.lower() or "missed" in c.lower()), None
        )

        injuries = []
        for _, row in injury_table.iterrows():
            injury = str(row.get(injury_col, "unknown")).strip() if injury_col else "unknown"
            if not injury or injury.lower() in ("nan", "unknown", *_NON_PLAYER_ROW_WORDS):
                continue
            injuries.append({
                "injury": injury,
                "from": str(row.get(from_col, "unknown")).strip() if from_col else "unknown",
                "until": str(row.get(until_col, "unknown")).strip() if until_col else "unknown",
                "days_missed": str(row.get(days_col, "unknown")).strip() if days_col else "unknown",
            })

        return {
            "found": True,
            "name": player_name,
            "injuries": injuries,
            "source_url": injury_url,
        }
    except Exception as e:
        return {"found": False, "reason": f"Error while reading the Transfermarkt injury page: {e}"}


@lru_cache(maxsize=128)
def search_club(club_name: str) -> Optional[str]:
    """Uses Transfermarkt's search engine to find a club's squad page.
    Matches on the stable URL pattern "/startseite/verein/<id>".

    To avoid grabbing an unrelated club/country link (which can happen if
    the search results page also contains "most valuable clubs" style
    widgets), we prefer a link whose visible text/title actually contains
    a significant word from `club_name`.
    """
    query = club_name.replace(" ", "+")
    search_url = f"https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query={query}"
    soup = _get_soup(search_url)

    # Use the most distinctive word in the club name for matching (skips
    # generic words like "FC"/"CF"/"United" that appear in many names).
    generic_words = {"fc", "cf", "sc", "afc", "cfc", "united", "city", "club", "the"}
    name_words = [w.lower() for w in club_name.split() if w.lower() not in generic_words]
    key_word = name_words[0] if name_words else club_name.lower()

    candidates = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if "/startseite/verein/" not in href:
            continue
        label = (a_tag.get("title", "") + " " + a_tag.get_text(" ", strip=True)).lower()
        full_url = "https://www.transfermarkt.com" + href if href.startswith("/") else href
        candidates.append((label, full_url))

    # First pass: prefer a candidate whose label actually mentions the club.
    for label, full_url in candidates:
        if key_word in label:
            return full_url

    # Fallback: no confident match found; return the first raw candidate
    # (better than nothing, but the caller should treat this as uncertain).
    if candidates:
        return candidates[0][1]
    return None


def get_club_squad(club_name: str, position_filter: Optional[str] = None) -> dict:
    """For the Manager/Analyst roles: the club's full squad (name,
    position, age, market value), optionally filtered to players whose
    listed position contains `position_filter` as a case-insensitive
    substring (e.g. "Forward" matches "Centre-Forward").

    Example return:
        {"found": True, "club": "...", "source_url": "...",
         "players": [{"name": "...", "position": "...", "age": "...",
                       "market_value": "..."}, ...]}
    """
    try:
        squad_url = search_club(club_name)
        if squad_url is None:
            return {"found": False, "reason": f"No Transfermarkt result found for club '{club_name}'."}

        soup = _get_soup(squad_url)

        # Safety check: if we somehow ended up on the Transfermarkt
        # homepage or some other generic page instead of a club's squad
        # page, fail loudly instead of silently parsing garbage. A real
        # club page's <title> always includes the club's own name.
        page_title = soup.title.get_text(strip=True) if soup.title else ""
        if "transfermarkt - the football" in page_title.lower() or page_title.lower().startswith("football transfers, rumours"):
            return {
                "found": False,
                "reason": (
                    f"Landed on the Transfermarkt homepage instead of {club_name}'s squad "
                    f"page (URL tried: {squad_url}) -- likely bot detection or a bad club match."
                ),
            }

        import pandas as pd
        import io

        # pd.read_html can misinterpret a very long HTML string as a file
        # path in some pandas versions, raising "No such file or
        # directory". Wrapping it in StringIO forces it to be treated as
        # in-memory text instead.
        tables = pd.read_html(io.StringIO(str(soup)))
        squad_table = None
        for t in tables:
            cols = [str(c) for c in t.columns]
            if any("Player" in c or "Name" in c for c in cols):
                squad_table = t
                break
        if squad_table is None:
            return {"found": False, "reason": "Squad table not found (page layout may have changed)."}

        if hasattr(squad_table.columns, "droplevel") and squad_table.columns.nlevels > 1:
            squad_table.columns = squad_table.columns.get_level_values(-1)

        # Transfermarkt squad tables show the player's name and position
        # stacked in the SAME cell (name on one line, position on the
        # next). pandas.read_html splits this into three consecutive rows
        # per player instead of one: "Name Position" (with the real
        # market value), then "Name" alone, then "Position" alone (both
        # with a NaN market value). We recover the real rows by keeping
        # only the ones with an actual market value, then split the
        # combined "Name Position" string using _split_name_position.

        players = []
        for _, row in squad_table.iterrows():
            raw_name = str(row.get("Player", row.get("Name", "unknown"))).strip()
            market_value = str(row.get("Market value", row.get("Market Value", "unknown"))).strip()

            # Skip the "leftover" split rows (NaN market value) -- only
            # the row that still has the combined "Name Position" string
            # carries the real market value.
            if market_value.lower() == "nan":
                continue

            name, position = _split_name_position(raw_name)

            # Best-effort age extraction: Transfermarkt typically shows
            # birth date + age together like "Jun 22, 1999 (26)" in some
            # column; scan every cell in the row for that "(NN)" pattern.
            age = "unknown"
            age_match = re.search(r"\((\d{1,2})\)", " ".join(str(v) for v in row.values))
            if age_match:
                age = age_match.group(1)

            if position_filter and position_filter.lower() not in position.lower():
                continue

            players.append({
                "name": name,
                "position": position,
                "age": age,
                "market_value": market_value,
            })

        return {
            "found": True,
            "club": club_name,
            "players": players,
            "source_url": squad_url,
        }
    except Exception as e:
        return {"found": False, "reason": f"Error while reading the Transfermarkt squad page: {e}"}


# =======================================================================
# TRANSFERMARKT — club injuries & suspensions (for the Manager role)
# =======================================================================
def get_club_injuries_and_suspensions(club_name: str) -> dict:
    """For the Manager role: who is currently injured or suspended at the
    club, so "we already have 3 centre-backs" claims can be checked
    against "...but 2 of them are out injured right now".

    Transfermarkt exposes this at the club's "sperrenundverletzungen"
    (bans & injuries) subpage, at the same URL as the squad page but with
    that path segment swapped in.

    Example return:
        {"found": True, "club": "...", "source_url": "...",
         "players": [{"name": "...", "reason": "Knee injury",
                       "status": "Since Jun 3, 2026"}, ...]}
        (an empty "players" list means no current injuries/suspensions
        are listed -- that's a valid, meaningful result, not a failure)
    """
    try:
        squad_url = search_club(club_name)
        if squad_url is None:
            return {"found": False, "reason": f"No Transfermarkt result found for club '{club_name}'."}

        injury_url = squad_url.replace("/startseite/verein/", "/sperrenundverletzungen/verein/")

        soup = _get_soup(injury_url)
        import pandas as pd
        import io

        tables = pd.read_html(io.StringIO(str(soup)))
        injury_table = None
        for t in tables:
            cols = [str(c) for c in t.columns]
            if any("Player" in c or "Name" in c for c in cols):
                injury_table = t
                break

        if injury_table is None:
            # No table at all can legitimately mean "nobody is out right now".
            return {"found": True, "club": club_name, "players": [], "source_url": injury_url}

        if hasattr(injury_table.columns, "droplevel") and injury_table.columns.nlevels > 1:
            injury_table.columns = injury_table.columns.get_level_values(-1)

        name_col = "Player" if "Player" in injury_table.columns else (
            "Name" if "Name" in injury_table.columns else None
        )
        reason_col = next(
            (c for c in injury_table.columns if c in ("Injury", "Reason", "Ban")), None
        )

        players = []
        for _, row in injury_table.iterrows():
            raw_name = str(row.get(name_col, "unknown")).strip() if name_col else "unknown"
            if not raw_name or raw_name.lower() == "nan":
                continue

            # Filter out section/category divider rows that pandas can
            # mistake for a player row (e.g. a lone "Injuries" heading).
            if raw_name.lower() in _NON_PLAYER_ROW_WORDS:
                continue

            name, _position = _split_name_position(raw_name)
            reason = str(row.get(reason_col, "unknown")).strip() if reason_col else "unknown"

            # A phantom header row often has name == reason (both read
            # "Injuries", "Suspended", etc.) -- skip those too.
            if name.lower() == reason.strip().lower():
                continue

            # We don't know the exact column name for "since/return date"
            # across all Transfermarkt layouts, so instead of guessing one
            # name, collect whatever OTHER non-empty columns exist (besides
            # name/reason) and join them -- this degrades gracefully
            # instead of always showing "unknown".
            other_bits = []
            for col in injury_table.columns:
                if col in (name_col, reason_col):
                    continue
                val = str(row.get(col, "")).strip()
                if val and val.lower() != "nan":
                    other_bits.append(val)
            status = " / ".join(other_bits) if other_bits else "unknown"

            players.append({"name": name, "reason": reason, "status": status})

        return {
            "found": True,
            "club": club_name,
            "players": players,
            "source_url": injury_url,
        }
    except Exception as e:
        return {"found": False, "reason": f"Error while reading the Transfermarkt injuries page: {e}"}


# =======================================================================
# TRANSFERMARKT — league standing (for the Manager role)
# =======================================================================
def get_club_league_standing(club_name: str) -> dict:
    """For the Manager role: where the club currently sits in its
    league table, so urgency can be judged realistically (fighting
    relegation vs. chasing a title vs. mid-table comfort).

    We reuse the club's own homepage (the same page get_club_squad
    fetches) and look for a generic-looking league table among all its
    tables (columns like a rank marker and points), then find the row
    that matches the club.

    Example return:
        {"found": True, "club": "...", "position": "4", "points": "58",
         "played": "28", "source_url": "..."}
    """
    try:
        home_url = search_club(club_name)
        if home_url is None:
            return {"found": False, "reason": f"No Transfermarkt result found for club '{club_name}'."}

        soup = _get_soup(home_url)
        import pandas as pd
        import io

        tables = pd.read_html(io.StringIO(str(soup)))

        generic_words = {"fc", "cf", "sc", "afc", "cfc", "united", "city", "club", "the"}
        name_words = [w.lower() for w in club_name.split() if w.lower() not in generic_words]
        key_word = name_words[0] if name_words else club_name.lower()

        def _looks_like_rank_col(col_name: str, series) -> bool:
            if col_name.strip().lower() in ("#", "pos", "pos.", "position", "rk", "rank", "unnamed: 0"):
                return True
            # Header-less rank columns sometimes come through as a plain
            # sequential-looking numeric column; treat that as a rank too.
            try:
                vals = [int(str(v).strip().rstrip(".")) for v in series.head(5)]
                return vals == sorted(vals) and vals[0] in (1, 0)
            except (ValueError, TypeError, IndexError):
                return False

        for t in tables:
            cols = [str(c) for c in t.columns]
            has_points = any("pt" in c.lower() for c in cols)
            has_club_col = any("club" in c.lower() or "team" in c.lower() for c in cols)
            if not (has_points and has_club_col):
                continue

            club_col = next(c for c in cols if "club" in c.lower() or "team" in c.lower())
            points_col = next(c for c in cols if "pt" in c.lower())
            rank_col = next((c for c in cols if _looks_like_rank_col(c, t[c])), None)
            played_col = next(
                (c for c in cols if c.strip().lower() in ("pl.", "played", "mp", "matches")), None
            )

            for _, row in t.iterrows():
                cell = str(row.get(club_col, "")).lower()
                if key_word in cell:
                    return {
                        "found": True,
                        "club": club_name,
                        "position": str(row.get(rank_col, "unknown")) if rank_col else "unknown",
                        "points": str(row.get(points_col, "unknown")),
                        "played": str(row.get(played_col, "unknown")) if played_col else "unknown",
                        "source_url": home_url,
                    }

        # Nothing matched -- rather than a bare "not found", report what
        # WAS on the page so the next debugging round has something
        # concrete to go on instead of guessing blind again.
        table_summaries = [
            [str(c) for c in t.columns][:8] for t in tables[:10]
        ]
        return {
            "found": False,
            "reason": (
                f"No league table with '{key_word}' found on {club_name}'s Transfermarkt "
                f"homepage. Tables seen on page (first 8 columns each): {table_summaries}"
            ),
        }
    except Exception as e:
        return {"found": False, "reason": f"Error while reading the Transfermarkt homepage: {e}"}


# =======================================================================
# FBREF — for the Analytics Team. Previously blocked by FBref's bot
# protection when using cloudscraper; retrying now with curl_cffi's
# TLS-level browser impersonation (the same fix that got Transfermarkt
# working reliably).
# =======================================================================
def search_fbref_player(player_name: str) -> Optional[str]:
    query = player_name.replace(" ", "+")
    search_url = f"https://fbref.com/en/search/search.fcgi?search={query}"

    fbref_headers = dict(HEADERS)
    fbref_headers["Referer"] = "https://fbref.com/en/"

    # Use a Session (not a one-off request) so a first "visit the
    # homepage" hit can pick up cookies before the real search request
    # -- looks more like an actual browser session, less like a script
    # making a single cold request straight to a search endpoint.
    with cffi_requests.Session(impersonate="chrome124") as session:
        try:
            session.get("https://fbref.com/en/", headers=HEADERS, timeout=15)
            time.sleep(1)
        except Exception:
            pass  # if the homepage warm-up fails, still try the real request below

        resp = session.get(search_url, headers=fbref_headers, timeout=15, allow_redirects=True)
        resp.raise_for_status()
        time.sleep(REQUEST_DELAY_SECONDS)

    if re.search(r"/en/players/[0-9a-f]{8}/", resp.url):
        return resp.url

    soup = BeautifulSoup(resp.text, "lxml")
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if re.match(r"^/en/players/[0-9a-f]{8}/", href):
            return "https://fbref.com" + href
    return None


def _read_tables_with_comment_fallback(soup: BeautifulSoup, match: str):
    """FBref hides some tables inside HTML comments to deter simple
    scrapers. If pandas can't find the table in the plain HTML, we also
    try inside the comments."""
    import pandas as pd
    import io

    html = str(soup)
    try:
        tables = pd.read_html(io.StringIO(html), match=match)
        if tables:
            return tables
    except ValueError:
        pass

    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for comment in comments:
        if match in comment:
            try:
                return pd.read_html(io.StringIO(str(comment)), match=match)
            except ValueError:
                continue
    return []


def get_fbref_player_stats(player_name: str) -> dict:
    try:
        profile_url = search_fbref_player(player_name)
        if profile_url is None:
            return {"found": False, "reason": f"No FBref result found for '{player_name}'."}

        soup = _get_soup(profile_url, referer="https://fbref.com/en/search/search.fcgi")
        tables = _read_tables_with_comment_fallback(soup, match="Standard Stats")
        if not tables:
            return {"found": False, "reason": "Stats table not found (page layout may have changed)."}

        df = tables[0]
        if hasattr(df.columns, "droplevel") and df.columns.nlevels > 1:
            df.columns = df.columns.get_level_values(-1)

        last_row = df.iloc[-1]

        def _safe(col):
            return str(last_row[col]) if col in last_row.index else "unknown"

        return {
            "found": True,
            "name": player_name,
            "season": _safe("Season"),
            "minutes": _safe("Min"),
            "goals": _safe("Gls"),
            "assists": _safe("Ast"),
            "xg": _safe("xG"),
            "xag": _safe("xAG"),
            "source_url": profile_url,
        }
    except Exception as e:
        # Catches failures from search_fbref_player() too (e.g. an
        # HTTPError from FBref's bot protection), not just from reading
        # the profile page -- previously an unhandled exception here
        # crashed the whole analytics_team node instead of degrading to
        # a normal "found: False" result.
        return {"found": False, "reason": f"Error while fetching FBref data: {e}"}


# =======================================================================
# FAN SENTIMENT — local, user-curated X/Twitter comment files
# (for the Fans role)
# =======================================================================
# Unlike every tool above, this does NOT scrape anything live. X's
# current anti-scraping measures make reliable, free, live scraping
# impractical to build directly into this codebase. Instead, YOU
# populate one or more plain .txt files yourself -- one fan
# comment/post per line -- using whatever collection method you choose
# (e.g. Scweet, a paid scraper service, manual copy-paste), and this
# tool just searches those local files for lines relevant to whatever
# club/player is under discussion.
#
# Expected setup:
#   fan_data/                      <- FAN_DATA_DIR, any folder you like
#     liverpool_x_posts.txt        <- one fan comment per line
#     osimhen_x_posts.txt
#     ...as many .txt files as you want, named however you want
#
# File names and which club/player they're "about" don't matter to the
# code -- it searches every .txt file in the directory for lines
# containing the club name and/or player name(s) from the current topic.

# This file (tools.py) lives in a Tools/ folder now, not the project
# root -- default fan_data/ to a SIBLING of Tools/ (i.e. the project
# root), so it doesn't matter which directory you happen to run
# `python engine.py` from.
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))     # .../SynFC/Tools
_PROJECT_ROOT = os.path.dirname(_TOOLS_DIR)                  # .../SynFC

FAN_DATA_DIR = os.environ.get("FAN_DATA_DIR", os.path.join(_PROJECT_ROOT, "fan_data"))


def search_fan_comments(query_terms: List[str], data_dir: Optional[str] = None, max_results: int = 25) -> dict:
    """Searches every .txt file in `data_dir` (default: FAN_DATA_DIR) for
    lines that mention ANY of `query_terms` (case-insensitive substring
    match), and returns a sample of matching lines plus the total count.

    Example return:
        {"found": True, "total_matches": 340,
         "sample": ["line 1", "line 2", ...],
         "files_searched": ["fan_data/liverpool_x_posts.txt", ...]}
    """
    directory = data_dir or FAN_DATA_DIR
    if not os.path.isdir(directory):
        return {
            "found": False,
            "reason": (
                f"Fan data directory '{directory}' does not exist. Create it and add "
                f"one or more .txt files (one fan comment per line) to use this tool. "
                f"You can point elsewhere via the FAN_DATA_DIR environment variable."
            ),
        }

    txt_filenames = [f for f in os.listdir(directory) if f.lower().endswith(".txt")]
    if not txt_filenames:
        return {"found": False, "reason": f"No .txt files found in '{directory}'."}

    terms_lower = [t.lower() for t in query_terms if t and t.strip()]
    matches = []
    files_searched = []

    for filename in txt_filenames:
        path = os.path.join(directory, filename)
        files_searched.append(path)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if not terms_lower or any(term in line.lower() for term in terms_lower):
                        matches.append(line)
        except Exception:
            continue  # skip unreadable files, don't fail the whole search

    if not matches:
        return {"found": True, "total_matches": 0, "sample": [], "files_searched": files_searched}

    # Take an evenly-spaced sample across all matches (instead of just
    # the first N) so the persona sees a representative spread rather
    # than always the same handful of earliest-added lines.
    if len(matches) > max_results:
        step = len(matches) / max_results
        sample = [matches[int(i * step)] for i in range(max_results)]
    else:
        sample = matches

    return {
        "found": True,
        "total_matches": len(matches),
        "sample": sample,
        "files_searched": files_searched,
    }


# =======================================================================
# PLANNED TOOLS — signatures only, NOT implemented yet.
# Tracked here so team modules can already be written against a stable
# interface once these are filled in, without needing to touch the
# calling code again later. Calling any of these right now raises
# NotImplementedError on purpose (so a team accidentally depending on
# one fails loudly, instead of silently getting empty/fake data).
# =======================================================================
def get_transfer_rumors(club_name: str) -> dict:
    """Pulls a club's current transfer rumors from Transfermarkt's own
    rumor-mill page for that club.

    UNTESTED against the live site -- URL pattern ("/geruechte/verein/")
    is a best-effort guess following the same URL-suffix-swap pattern
    used by get_club_league_history()/get_club_transfer_history()
    (Transfermarkt's German-language URL segments are consistent site-
    wide: "geruechte" = "rumours"). Verify with one club first; column
    names may need adjustment, same iterative pattern as every other
    Transfermarkt tool in this project.

    Returns:
        {"found": True, "club": "...", "source_url": "...",
         "rumors": [{"player": "...", "position": "...", "linked_club": "...",
                      "probability": "...", "date": "..."}, ...]}

    The "source" column originally assumed to carry a journalist's name
    (e.g. "Fabrizio Romano") turned out, in live testing, to actually be
    a date column -- renamed to "date" accordingly. No journalist-name
    column was found on this page at all, and "linked_club" also came
    back empty in live testing (all "nan") since the assumed
    column-name keywords ("club"/"from") didn't match anything real on
    the page. Both are open items to re-diagnose with the real column
    names (print `[list(t.columns) for t in tables]` on a live page).
    """
    club_url = search_club(club_name)
    if club_url is None:
        return {"found": False, "reason": f"No Transfermarkt result found for club '{club_name}'."}

    rumors_url = club_url.replace("/startseite/verein/", "/geruechte/verein/")

    try:
        soup = _get_soup(rumors_url)
        tables = pd.read_html(io.StringIO(str(soup)))
    except Exception as e:
        return {"found": False, "reason": f"Error while reading the rumors page: {e}"}

    rumors = []
    for t in tables:
        cols = [str(c) for c in t.columns]
        player_col = next((c for c in cols if "player" in c.lower()), None)
        club_col = next((c for c in cols if "club" in c.lower() or "from" in c.lower()), None)
        probability_col = next((c for c in cols if "probability" in c.lower() or "%" in c), None)
        source_col = next((c for c in cols if "source" in c.lower() or "date" in c.lower()), None)

        if not player_col:
            continue

        for _, row in t.iterrows():
            raw_name = str(row.get(player_col, "")).strip()
            if not raw_name or raw_name.lower() == "nan":
                continue

            # Phantom-row filter (same fix as get_club_squad/get_club_transfer_history's
            # earlier bug): Transfermarkt stacks name + position in ONE
            # cell, and pandas.read_html splits that into multiple
            # "leftover" rows -- only the row that also carries a real
            # source/date is the genuine entry, the other two (bare name,
            # bare position) are fragments with everything else NaN.
            source_val = row.get(source_col) if source_col else None
            has_real_source = source_val is not None and str(source_val).strip().lower() not in ("", "nan")
            if not has_real_source:
                continue

            name, position = _split_name_position(raw_name)

            rumors.append({
                "player": name,
                "position": position,
                "linked_club": str(row.get(club_col, "unknown")) if club_col else "unknown",
                "probability": str(row.get(probability_col, "unknown")) if probability_col else "unknown",
                "date": str(source_val),
            })

    if not rumors:
        return {
            "found": False,
            "reason": "No rumor rows parsed -- page structure likely differs from what was assumed. "
                      f"Tables found on page: {[list(t.columns) for t in tables]}",
        }

    return {"found": True, "club": club_name, "source_url": rumors_url, "rumors": rumors}


def web_search(query: str) -> dict:
    """PLANNED: a general-purpose web search (not tied to any single
    site's HTML structure, unlike everything else in this file) for
    questions that don't fit a fixed Transfermarkt/FBref page -- e.g.
    "what's the latest news on this player's transfer saga". Meant to
    be far less fragile than page-scraping since it doesn't depend on
    one site's markup staying the same.

    Intended return shape (once implemented):
        {"found": True, "query": "...",
         "results": [{"title": "...", "snippet": "...", "url": "..."}, ...]}
    """
    raise NotImplementedError("web_search is a planned tool, not implemented yet.")


def get_next_opponent_analysis(club_name: str) -> dict:
    """PLANNED: identify a club's next fixture and pull basic info
    about that opponent (recent form, key players, playing style) --
    lets the technical team argue "we need THIS player specifically for
    THIS matchup" instead of only generic squad-depth reasoning.

    Intended return shape (once implemented):
        {"found": True, "club": "...", "opponent": "...", "date": "...",
         "competition": "...", "opponent_recent_form": "...",
         "source_url": "..."}
    """
    raise NotImplementedError("get_next_opponent_analysis is a planned tool, not implemented yet.")


if __name__ == "__main__":
    # Quick manual test:
    #   python tools.py player "Lionel Messi"
    #   python tools.py club "Galatasaray"
    #   python tools.py fans "Liverpool" "Osimhen"
    #   python tools.py fbref "Bruno Guimaraes"
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "player"
    query = " ".join(sys.argv[2:]) or "Lionel Messi"

    if mode == "club":
        print("=== CLUB SQUAD ===")
        print(get_club_squad(query))
    elif mode == "fans":
        print("=== FAN COMMENTS ===")
        print(search_fan_comments(sys.argv[2:]))
    elif mode == "fbref":
        print("=== FBREF SEARCH URL ===")
        try:
            print(search_fbref_player(query))
        except Exception as e:
            print(f"ERROR: {e}")
        print("\n=== FBREF STATS ===")
        print(get_fbref_player_stats(query))
    else:
        print("=== FINANCIALS ===")
        print(get_player_financials(query))
        print("\n=== POSITIONS ===")
        print(get_player_positions(query))