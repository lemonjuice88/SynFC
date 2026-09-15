"""
sql_database.py — real SQL database for the Data Scientist agent
======================================================================
Converts the project's local CSV (Top_5_European_Leagues_2024_25...)
into a real SQLite database, and exposes TEXT-TO-SQL: the Data
Scientist agent gives a natural-language question, an LLM writes the
actual SQL query, and we execute it for real against real data.

Safety: only SELECT statements are ever executed (regex-checked before
running anything) -- the LLM-generated SQL can read data but never
modify the database.

Setup: no extra packages needed -- sqlite3 and pandas are already
project dependencies.
"""

import os
import re
import sqlite3
from functools import lru_cache
from typing import List

import pandas as pd

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_TOOLS_DIR)

DEFAULT_CSV_PATH = os.environ.get(
    "PLAYER_STATS_CSV_PATH",
    os.path.join(_PROJECT_ROOT, "data", "Top_5_European_Leagues_2024_25_Complete_Player_Stats.csv"),
)
DEFAULT_DB_PATH = os.environ.get(
    "PLAYER_STATS_DB_PATH",
    os.path.join(_PROJECT_ROOT, "data", "player_stats.sqlite"),
)

TABLE_NAME = "player_stats"

# Real columns from the CSV, several with spaces/special characters
# (e.g. "Playing Time_MP", "Performance_G+A") -- SQLite needs these
# double-quoted in any query, both when WE build the table and when the
# LLM writes SQL against it (the schema text given to the LLM shows
# this explicitly).
_SCHEMA_DESCRIPTION = """\
Table: player_stats
Columns (double-quote any column name containing a space or +/-):
  league TEXT, season INTEGER, team TEXT, player TEXT, nation TEXT,
  pos TEXT, age REAL, born REAL,
  "Playing Time_MP" INTEGER, "Playing Time_Starts" INTEGER,
  "Playing Time_Min" INTEGER, "Playing Time_90s" REAL,
  "Performance_Gls" INTEGER, "Performance_Ast" INTEGER,
  "Performance_G+A" INTEGER, "Performance_G-PK" INTEGER,
  "Performance_PK" INTEGER, "Performance_PKatt" INTEGER,
  "Performance_CrdY" INTEGER, "Performance_CrdR" INTEGER,
  "Per 90 Minutes_Gls" REAL, "Per 90 Minutes_Ast" REAL,
  "Per 90 Minutes_G+A" REAL, "Per 90 Minutes_G-PK" REAL,
  "Per 90 Minutes_G+A-PK" REAL,
  np_xg REAL, xg REAL, xa REAL, np_goals INTEGER, key_passes INTEGER,
  xg_chain REAL, xg_buildup REAL

IMPORTANT -- the `league` column's ACTUAL values are country-code
prefixed, NOT plain league names. The only 5 valid values are exactly:
  'ENG-Premier League', 'ESP-La Liga', 'GER-Bundesliga',
  'ITA-Serie A', 'FRA-Ligue 1'
If the question mentions "Premier League"/"La Liga"/etc. by their
common name, you MUST translate that to the exact matching value above
in your WHERE clause (e.g. WHERE league = 'ENG-Premier League') -- a
plain 'Premier League' will match ZERO rows and silently return nothing.

Data covers the 2024/25 season across these 5 leagues. One row per
player per team-stint that season (a player transferred mid-season can
have more than one row).
"""


@lru_cache(maxsize=1)
def _get_connection(csv_path: str = DEFAULT_CSV_PATH, db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Builds the SQLite DB from the CSV on first use (cached after
    that -- the CSV -> DB conversion only happens once per process,
    not once per query)."""
    if not os.path.exists(db_path):
        df = pd.read_csv(csv_path)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
    else:
        conn = sqlite3.connect(db_path, check_same_thread=False)
    return conn


_SELECT_ONLY_PATTERN = re.compile(r"^\s*SELECT\b", re.IGNORECASE)
_FORBIDDEN_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|PRAGMA)\b", re.IGNORECASE
)


def run_sql_query(sql: str, max_rows: int = 50) -> dict:
    """Executes a raw SQL query against the real database. SELECT-only
    -- rejects anything else (see _FORBIDDEN_PATTERN) before ever
    touching the connection, so a malformed/malicious generated query
    can't modify data.

    Example return:
        {"found": True, "sql": "...", "columns": [...], "rows": [...]}
    """
    if not _SELECT_ONLY_PATTERN.match(sql) or _FORBIDDEN_PATTERN.search(sql):
        return {"found": False, "reason": "Only SELECT queries are allowed.", "sql": sql}

    try:
        conn = _get_connection()
        cursor = conn.execute(sql)
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchmany(max_rows)
        rows_as_dicts = [dict(zip(columns, row)) for row in rows]
        return {"found": True, "sql": sql, "columns": columns, "rows": rows_as_dicts}
    except sqlite3.Error as e:
        return {"found": False, "reason": f"SQL error: {e}", "sql": sql}


TEXT_TO_SQL_SYSTEM = f"""\
You write SQLite SELECT queries against this exact schema:

{_SCHEMA_DESCRIPTION}

Given a natural-language question, reply with ONLY the raw SQL query --
no explanation, no markdown code fences, no extra text. Quote any
column name containing a space or +/- character with double quotes,
exactly as shown in the schema above. Use LIMIT to keep results
reasonable (20 rows max) unless the question clearly asks for an
aggregate (COUNT/AVG/SUM) that returns one row anyway.
"""


def ask_database(question: str) -> dict:
    """TEXT-TO-SQL: an LLM writes the actual SQL query for `question`,
    then it's actually executed against the real database. Returns both
    the generated SQL (for transparency) and the real results.

    Example return:
        {"found": True, "question": "...", "sql": "SELECT ...",
         "columns": [...], "rows": [...]}
    """
    import core  # local import to avoid a circular import at module load time

    generated_sql = core.call_llm(TEXT_TO_SQL_SYSTEM, question).strip()
    # Strip markdown code fences if the LLM added them despite instructions.
    generated_sql = re.sub(r"^```(?:sql)?\s*|\s*```$", "", generated_sql, flags=re.IGNORECASE).strip()

    result = run_sql_query(generated_sql)
    result["question"] = question
    return result


if __name__ == "__main__":
    # Quick manual test:
    #   python sql_database.py "Which 5 players had the highest xG in 2024/25?"
    import sys
    import json

    question = " ".join(sys.argv[1:]) or "Which 5 players had the highest xG in 2024/25?"
    result = ask_database(question)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str)[:2000])