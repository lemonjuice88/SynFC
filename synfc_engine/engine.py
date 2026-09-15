"""
engine.py — the main engine
==============================
Wires together the Main Router, the conflict-detection/debate loop, and
the Board of Directors' final decision, with each of the 7 departments'
own logic living in its own module as a self-contained sub-graph
(technical_team.py, finance_team.py, media_team.py, data_team.py,
scout_team.py, legal_team.py, health_team.py). Shared infrastructure
(config, call_llm, state schema, extraction helpers, and the generic
sub-team factory every department is built from) lives in core.py.

Setup:
    pip install langgraph openai python-dotenv cloudscraper beautifulsoup4 lxml pandas
    # create a .env file in the project folder with:
    #   DEEPSEEK_API_KEY=sk-xxxxxxx

Run:
    python engine.py "Would Salah suit RB Leipzig?"

Flow:
    START -> router -> (fan-out: technical_team / finance_team / media_team
                         / data_team / scout_team / legal_team / health_team)
           -> collect -> is there a conflict + round < 5 ?
                 yes -> debate_dispatch (round_count++) -> departments run again -> collect
                 no / round limit hit -> [good_cop + bad_cop in parallel] -> president -> END

Each department listed above is itself a compiled LangGraph sub-graph
(SubRouter -> its own agents -> collect -> debate -> Director) built by
core.build_subteam() -- see SYNFC_VISION.md for the full picture and
core.py's docstring for why the module split avoids circular imports.

File map:
    core.py               shared config, call_llm, state schema, extraction
                           helpers, and the generic build_subteam() factory
    technical_team.py      Translator / Psychologist / Technical Analyst -> Head Coach
    finance_team.py        Accounting / Investor -> Finance Director
    media_team.py           Fans / Reporter -> Media Director
    data_team.py            Data Scientist / Transfer Analyst / Match Analyst -> Data Director
    scout_team.py           5 regional scouts -> Scout Director
    legal_team.py           Administrative / Financial-Legal agents -> Legal Director
    health_team.py          Physio / Psychologist -> Club Doctor
    board_of_directors.py  Good Cop / Bad Cop / President (final decision stage)
    sentiment_scorer.py    background opinion-stance scoring (invisible to the user)
    tools.py               Transfermarkt/local-data helpers used by department modules
    calculator.py          pure-math financial calculations
    player_stats_dataset.py local top-5-league player stats lookup
    engine.py              this file: Main Router, collect/debate loop, graph, CLI
"""

import sys
import os

# ---------------------------------------------------------------------
# Path bootstrap for the split folder layout (synfc_engine/ + Tools/ +
# Sub_Teams/ as SIBLING folders instead of everything flat in one
# directory). Every file still does plain "import core", "import tools",
# "from finance_team import ..." etc. -- unchanged -- this block just
# makes sure Python's import system can FIND those files regardless of
# which sibling folder they physically live in.
#
# "Sub_Teams" (underscore) is a valid Python package name, unlike
# "Sub-Teams" (hyphen), so this folder could later become a real
# package with __init__.py + dotted imports if you want. For now it
# still uses plain sys.path insertion, since that requires zero changes
# to any of the team files' own import statements -- simplest option
# unless you specifically want proper package-style imports later.
_ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../SynFC/synfc_engine
_PROJECT_ROOT = os.path.dirname(_ENGINE_DIR)                       # .../SynFC

for _sibling_folder in ("Tools", "Sub_Teams"):
    _path = os.path.join(_PROJECT_ROOT, _sibling_folder)
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)
# ---------------------------------------------------------------------

import json

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

import core
from core import SynFCState, call_llm, ACTIVE_ROLE_NAMES, MAX_DEBATE_ROUNDS

from technical_team import technical_team_node
from finance_team import finance_team_node
from media_team import media_team_node
from data_team import data_team_node
from scout_team import scout_team_node
from legal_team import legal_team_node
from health_team import health_team_node
from board_of_directors import good_cop_node, bad_cop_node, president_node


# =======================================================================
# ENGINE-LEVEL PROMPTS (Main Router / conflict detection -- these belong
# to the engine itself, not to any one department, so they stay here
# rather than in core.py or a department module. The president's own
# prompt lives in board_of_directors.py now.)
# =======================================================================
MODE_CLASSIFIER_SYSTEM = """\
You are the FIRST step of the SynFC Main Router. Your ONLY job: classify
whether this message needs a JUDGMENT CALL or is a FACTUAL/INFORMATION
request. Do NOT pick departments here -- that's a separate step.

- "karar" (decision): the message asks for a recommendation, evaluation,
  or judgment call -- "should we sign X", "is this a good idea", "what
  do you think about Y", weighing a transfer/decision. Needs the full
  Board (debate rounds if departments disagree, then Good Cop/Bad Cop/
  President).
- "bilgi" (information): the message asks for a FACT -- "who is
  interested in X", "what happened with Y", "what does the regulation
  say about Z", "what is X's contract status", "who is the club's
  current captain", "who is our club interested in / kimlerle
  ilgileniyor" (this is a transfer-rumor FACT request, not a judgment
  call). No judgment call needed, no Board -- just the departments'
  factual findings. When genuinely unsure between the two, prefer
  "karar" -- it's the safer default (more scrutiny, not less).

Reply ONLY in this exact JSON format, with no other text:
{"mode": "karar"}
"""

DEPARTMENT_LIST_TEXT = """\
You may ONLY choose from these departments:
- "technical_team": tactics, squad, transfers, matches, player performance,
  language/cultural fit, dressing-room psychology
- "finance_team": budget, wages, transfer fees, revenue/expense, FFP,
  investment return, opportunity cost
- "media_team": supporter sentiment/reaction, club image, matchday
  atmosphere, press coverage, transfer rumors
- "data_team": statistics, advanced metrics, transfer-target data
  analysis, match/opponent analysis
- "scout_team": scouting a player/market in a specific world region
- "legal_team": procedure, discipline, registration, governing-body
  correspondence (e.g. "draft a letter to UEFA"), FFP/wage-cap compliance,
  OR any question about what FIFA/UEFA's actual regulations say on a
  topic
- "health_team": physical condition, injury risk, psychological/mental
  health considerations
"""

DEPARTMENT_SELECTOR_KARAR_SYSTEM = f"""\
You are the SECOND step of the SynFC Main Router. The message has
already been classified as a DECISION (a judgment call is being asked
for). Decide which department(s) it concerns.

{DEPARTMENT_LIST_TEXT}
CRITICAL: don't over-select. An entity (player, club) appearing in the
message does NOT automatically mean every department is relevant --
judge what the question is actually ASKING, not just which entities it
mentions.
- A full TRANSFER DECISION ("should we sign/buy this named player") is
  a real-world due-diligence process that ALWAYS includes a medical/
  injury-risk check -- no real club skips this. So for a named-player
  transfer decision, include technical (fit), finance (cost), data
  (data case), media (public reaction), AND health (injury-risk due
  diligence) BY DEFAULT -- health is NOT conditional on the question
  mentioning fitness explicitly, the same way cost isn't conditional on
  the question mentioning money. Only scout and legal stay genuinely
  conditional: scout only if the player is from a specific scouted
  region, legal only if there's an actual compliance question.
- A pure TACTICAL/STATISTICAL FIT question ("would this player solve
  our low xG") is about football reasoning and data ONLY -- leave out
  finance_team and media_team even if a specific player is named, unless
  cost or public sentiment is actually part of the question.
- An ADMINISTRATIVE/COMPLIANCE action ("draft an email to UEFA") belongs
  to legal_team, not technical/finance/media/data/scout/health.

If the topic clearly concerns none of them (a general/administrative
matter for the president alone), return an empty list — the decision
goes straight to the Board.

Reply ONLY in this exact JSON format, with no other text:
{{"roles": ["technical_team", "finance_team", "media_team", "data_team", "scout_team", "legal_team", "health_team"]}}
"""

DEPARTMENT_SELECTOR_BILGI_SYSTEM = f"""\
You are the SECOND step of the SynFC Main Router. The message has
already been classified as an INFORMATION request (a plain fact is
being asked for, not a judgment call). Decide which department(s) can
answer it.

{DEPARTMENT_LIST_TEXT}
Since this is a FACTUAL question, match it to whichever department(s)
would actually HOLD that fact -- think "who has this data", not "whose
judgment matters here". Some common patterns:
- "who is our club interested in" / "kimlerle ilgileniyor" / any
  transfer-rumor-style question -> media_team (transfer rumors are
  media_team's territory) -- pick this even if the phrasing sounds like
  it could be a scouting/technical question.
- A player's contract/wage/market value/FINANCIAL BURDEN (cost,
  amortization, total spend) -> finance_team ONLY. Don't also add
  technical_team just because a player is named -- technical_team's
  squad data is about "who plays this position now", not cost, so
  adding it here just produces an unhelpful "we don't have data"
  response, same reasoning as the pure-stats case below. Only add
  technical_team too if the question ALSO asks about squad fit/depth,
  not just cost.
- A player's stats/season performance -> data_team and/or technical_team
- A pure statistical/ranking/aggregate question with NO named player and
  NO tactical-fit framing (e.g. "top 5 assist providers this season",
  "who has the highest xG") -> data_team ONLY. Don't also add
  technical_team just because it's football-related -- technical_team
  doesn't have this raw stats database, only data_team does, so adding
  it here just produces an unhelpful "we don't have data" response.
- A regulation/rule question -> legal_team
- Injury history/fitness status -> health_team
- Fan/press reaction to something -> media_team

It's fine to select multiple departments if the fact could come from
more than one place. If genuinely nothing applies, return an empty list.

Reply ONLY in this exact JSON format, with no other text:
{{"roles": ["technical_team", "finance_team", "media_team", "data_team", "scout_team", "legal_team", "health_team"]}}
"""

CONFLICT_SYSTEM = """\
You'll be given the opinions of several departments on the same topic.
Your job is to detect whether there's a REAL disagreement between them.
A difference in tone or emphasis alone does NOT count as a conflict; a
genuine clash of recommendations/conclusions (one says "do it", another
says "don't") DOES count.

Reply ONLY in this exact JSON format:
{"conflict": true or false, "reason": "brief justification"}
"""

INFO_FORMATTER_SYSTEM = """\
You'll be given one or more departments' raw findings on an
INFORMATION-mode question (not a decision) -- these are written in each
department's own persona/voice (e.g. a "Media Director" giving their
take), NOT as neutral information.

Rewrite this into a clean, neutral INFORMATION ANSWER for the user:
- Strip away persona voice, board-member framing, and any "our
  recommendation is..." language -- this isn't a decision, don't make
  it sound like one.
- Organize the actual facts/findings clearly (a short list or a few
  plain sentences, whatever fits the content).
- PRESERVE all honesty markers -- if the departments said something is
  unconfirmed/a rumor/uncertain, keep that framing; don't launder
  uncertainty into false confidence.
- Do NOT add any new facts, names, or numbers that weren't in the
  original findings. Do NOT drop any of the substantive facts either --
  reformat/reorganize, don't summarize away detail.
- SPECIAL CASE: if one department gave a concrete, data-backed answer
  and another department said it has no data/couldn't answer, this is
  NOT a real disagreement -- it just means only one department had the
  right tool/data for this question. Lead with the concrete answer as
  the actual answer; only briefly mention (if at all) that another
  department didn't have relevant data, don't frame this as "two
  differing sets of information" or a conflict.
"""


# =======================================================================
# ENGINE NODES
# =======================================================================
def mode_classifier_node(state: SynFCState) -> dict:
    """Step 1 of 2: ONLY classifies mode (bilgi/karar) -- doesn't pick
    departments yet. Split into its own call (rather than one combined
    call) specifically so department_selector_node can use a MODE-
    SPECIFIC selection prompt next -- a dedicated, simpler prompt for
    "bilgi" queries (see DEPARTMENT_SELECTOR_BILGI_SYSTEM) turned out to
    classify departments more reliably than one prompt trying to cover
    both mode's selection logic at once (e.g. "kimlerle ilgileniyor"
    kept getting mis-routed away from media_team when both criteria sets
    were mixed into a single prompt)."""
    raw = call_llm(
        MODE_CLASSIFIER_SYSTEM,
        f"User message: {state['user_message']}",
        json_mode=True,
    )
    mode = "karar"  # safe default: if parsing fails, treat it as a decision
    #                 (more scrutiny -- full board -- not less)
    try:
        data = json.loads(raw)
        parsed_mode = data.get("mode", "karar")
        mode = parsed_mode if parsed_mode in ("karar", "bilgi") else "karar"
    except (json.JSONDecodeError, AttributeError):
        pass

    return {"mode": mode}


def department_selector_node(state: SynFCState) -> dict:
    """Step 2 of 2: picks departments using the prompt matching the mode
    ALREADY decided by mode_classifier_node -- a mode-specific prompt,
    not one prompt trying to serve both."""
    mode = state.get("mode", "karar")
    selector_prompt = DEPARTMENT_SELECTOR_BILGI_SYSTEM if mode == "bilgi" else DEPARTMENT_SELECTOR_KARAR_SYSTEM

    raw = call_llm(
        selector_prompt,
        f"User message: {state['user_message']}",
        json_mode=True,
    )
    roles = []
    try:
        data = json.loads(raw)
        roles = [r for r in data.get("roles", []) if r in ACTIVE_ROLE_NAMES]
    except (json.JSONDecodeError, AttributeError):
        roles = []  # if parsing fails, stay safe: no roles, straight to the Board

    # Extracted ONCE here (sequentially, before the parallel fan-out) so
    # every department sees the exact same club -- if each extracted it
    # independently in parallel, small LLM variance could have them
    # disagree on which club is even being discussed.
    club_name = core.extract_club_name(state["user_message"])

    return {
        "topic": state["user_message"],
        "club_name": club_name,
        "active_roles": roles,
        "round_count": 0,
        "role_opinions": {},
        "tool_contexts": {},
        "board_notes": {},
    }


def dispatch_to_roles(state: SynFCState):
    """Conditional edge: fans out to every department in active_roles in
    parallel via Send. If the list is empty, goes straight to collect."""
    roles = state.get("active_roles", [])
    if not roles:
        return "collect"
    return [Send(role, state) for role in roles]


def collect_node(state: SynFCState) -> dict:
    if state.get("mode") == "bilgi":
        # Information-mode: no conflict detection, no debate -- saves an
        # LLM call entirely. Routing decision happens in after_collect.
        return {"conflict": False, "conflict_reason": "Information-mode query, conflict detection skipped."}

    opinions = state.get("role_opinions", {})

    if len(opinions) <= 1:
        return {
            "conflict": False,
            "conflict_reason": "Not enough opinions to compare.",
            "active_roles": list(opinions.keys()),
        }

    # ------------------------------------------------------------------
    # Background sentiment scoring & neutral-opinion filtering.
    # Before checking for a real conflict, score each opinion on a 1-10
    # for/against STANCE scale using an INDEPENDENT scorer (see
    # sentiment_scorer.py). Opinions landing in the "neutral zone" don't
    # carry a real position either way, so they're quietly dropped from
    # further debate rounds. This entire step is invisible to the user.
    # ------------------------------------------------------------------
    decisive_opinions = dict(opinions)
    if core.SENTIMENT_SCORING_AVAILABLE:
        try:
            for role_name, text in opinions.items():
                score = core.sentiment_scorer.score_opinion(text)
                if core.sentiment_scorer.is_neutral(score):
                    decisive_opinions.pop(role_name, None)
        except Exception:
            decisive_opinions = dict(opinions)

    if len(decisive_opinions) <= 1:
        return {
            "conflict": False,
            "conflict_reason": "Remaining decisive opinions too few to compare (the rest were neutral).",
            "active_roles": list(decisive_opinions.keys()),
        }

    joined = "\n".join(f"{r}: {o}" for r, o in decisive_opinions.items())
    raw = call_llm(CONFLICT_SYSTEM, f"Opinions:\n{joined}", json_mode=True)

    try:
        data = json.loads(raw)
        conflict = bool(data.get("conflict", False))
        reason = data.get("reason", "")
    except (json.JSONDecodeError, AttributeError):
        conflict = False
        reason = "Conflict detection could not be parsed, assuming no conflict."

    return {
        "conflict": conflict,
        "conflict_reason": reason,
        # Only decisive departments get re-invoked if the debate
        # continues; neutral departments' round-0 opinion stays on
        # record for the Board, they just stop actively debating.
        "active_roles": list(decisive_opinions.keys()),
    }


def after_collect(state: SynFCState):
    """Conditional edge: for "bilgi" (information) mode, skip straight to
    the confidence-check node -- no debate, no Board, that's the whole
    point of this mode. For "karar" (decision) mode, unchanged: checks
    the debate round limit and conflict flag, fans out to Good Cop + Bad
    Cop when the debate is over."""
    if state.get("mode") == "bilgi":
        return "confidence_check"
    if state.get("round_count", 0) >= MAX_DEBATE_ROUNDS:
        return [Send("good_cop", state), Send("bad_cop", state)]
    if state.get("conflict"):
        return "debate_dispatch"
    return [Send("good_cop", state), Send("bad_cop", state)]


def confidence_check_node(state: SynFCState) -> dict:
    """"bilgi" (information) mode's whole job: present what the
    departments actually found (round 0 only, no debate, no Board) as
    CLEAN NEUTRAL INFORMATION -- not as a department's persona-voiced
    opinion. One LLM call (INFO_FORMATTER_SYSTEM) to strip persona/
    board-member framing while preserving every fact and every honesty
    marker (unconfirmed/rumor/uncertain language stays intact). Still
    far cheaper than "karar" mode: no debate rounds, no Good Cop/Bad
    Cop/President -- just this one lightweight formatting pass."""
    opinions = state.get("role_opinions", {})

    if not opinions:
        return {"final_decision": "[No department had information on this topic]"}

    joined = "\n\n".join(f"[{ROLE_LABELS.get(r, r.upper())}]\n{o}" for r, o in opinions.items())

    try:
        formatted = call_llm(INFO_FORMATTER_SYSTEM, f"Raw findings:\n{joined}")
    except Exception:
        formatted = joined  # if formatting fails for any reason, fall back to the raw findings

    return {"final_decision": formatted}


def debate_dispatch_node(state: SynFCState) -> dict:
    return {"round_count": state.get("round_count", 0) + 1}


# =======================================================================
# GRAPH
# =======================================================================
ROLE_NODE_NAMES = ACTIVE_ROLE_NAMES

ROLE_NODE_FUNCTIONS = {
    "technical_team": technical_team_node,
    "finance_team": finance_team_node,
    "media_team": media_team_node,
    "data_team": data_team_node,
    "scout_team": scout_team_node,
    "legal_team": legal_team_node,
    "health_team": health_team_node,
}


def build_graph():
    graph = StateGraph(SynFCState)

    graph.add_node("mode_classifier", mode_classifier_node)
    graph.add_node("department_selector", department_selector_node)
    for role_name, node_fn in ROLE_NODE_FUNCTIONS.items():
        graph.add_node(role_name, node_fn)
    graph.add_node("collect", collect_node)
    graph.add_node("confidence_check", confidence_check_node)
    graph.add_node("debate_dispatch", debate_dispatch_node)
    graph.add_node("good_cop", good_cop_node)
    graph.add_node("bad_cop", bad_cop_node)
    graph.add_node("president", president_node)

    graph.add_edge(START, "mode_classifier")
    graph.add_edge("mode_classifier", "department_selector")

    graph.add_conditional_edges(
        "department_selector", dispatch_to_roles, ROLE_NODE_NAMES + ["collect"]
    )

    for role_name in ROLE_NODE_NAMES:
        graph.add_edge(role_name, "collect")

    graph.add_conditional_edges(
        "collect", after_collect, ["debate_dispatch", "confidence_check", "good_cop", "bad_cop"]
    )

    graph.add_conditional_edges(
        "debate_dispatch", dispatch_to_roles, ROLE_NODE_NAMES + ["collect"]
    )

    graph.add_edge("confidence_check", END)
    graph.add_edge("good_cop", "president")
    graph.add_edge("bad_cop", "president")
    graph.add_edge("president", END)

    return graph.compile()


app = build_graph()


# =======================================================================
# CLI
# =======================================================================
ROLE_LABELS = {
    "technical_team": "TECHNICAL TEAM",
    "finance_team": "FINANCE TEAM",
    "media_team": "MEDIA TEAM",
    "data_team": "DATA TEAM",
    "scout_team": "SCOUT TEAM",
    "legal_team": "LEGAL TEAM",
    "health_team": "HEALTH TEAM",
}


def run(user_message: str):
    """Uses stream() instead of invoke(): as soon as a node finishes, its
    state update is delivered immediately, so we can watch the board's
    discussion unfold live, step by step."""
    initial_state = {"user_message": user_message}

    for step in app.stream(initial_state, stream_mode="updates"):
        for node_name, update in step.items():
            if node_name == "mode_classifier":
                print(f"\n[MODE] {update.get('mode', '')}")

            elif node_name == "department_selector":
                roles = update.get("active_roles", [])
                print(f"[MAIN ROUTER] Relevant departments: {roles or '(none, straight to the Board)'}")

            elif node_name in ROLE_LABELS:
                opinion = update.get("role_opinions", {}).get(node_name, "")
                print(f"\n[{ROLE_LABELS[node_name]}]\n{opinion}")

            elif node_name == "collect":
                if update.get("conflict"):
                    print(f"\n[CONFLICT DETECTED] {update.get('conflict_reason', '')}")
                else:
                    print(f"\n[NO CONFLICT] {update.get('conflict_reason', '')}")

            elif node_name == "confidence_check":
                print(f"\n=== INFORMATION ===\n{update.get('final_decision', '')}")

            elif node_name == "debate_dispatch":
                print(f"\n--- Starting debate round {update.get('round_count')} ---")

            elif node_name == "good_cop":
                print(f"\n[GOOD COP]\n{update.get('board_notes', {}).get('good_cop', '')}")

            elif node_name == "bad_cop":
                print(f"\n[BAD COP]\n{update.get('board_notes', {}).get('bad_cop', '')}")

            elif node_name == "president":
                print(f"\n=== PRESIDENT'S DECISION ===\n{update.get('final_decision', '')}")


if __name__ == "__main__":
    # If an argument is given, use it (e.g. python engine.py "Should we sign Messi?")
    # otherwise ask interactively.
    if len(sys.argv) > 1:
        msg = " ".join(sys.argv[1:])
    else:
        msg = input("What would you like to consult SynFC about? ").strip()

    if not msg:
        print("Cannot run with an empty message.")
        sys.exit(1)

    run(msg)