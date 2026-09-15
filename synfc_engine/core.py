"""
core.py — SynFC shared infrastructure
=========================================
Everything the main engine (engine.py) AND every individual team module
(technical_team.py, finance_team.py, analytics_team.py, fans.py) depend
on: config, the LLM wrapper, the shared graph state schema, and the
handful of extraction helpers (player name / club / position) that more
than one team needs.

This file exists specifically to avoid circular imports: team modules
need call_llm() and the extraction helpers, while engine.py (the
engine) needs to import the team modules' *_node functions to build the
graph. If team modules imported engine.py directly, that would be
circular. Team modules import core.py instead; only engine.py imports
both core.py and the team modules.
"""

import os
import sys
import json
from typing import TypedDict, Annotated, List, Dict, Optional, Callable

from dotenv import load_dotenv, find_dotenv

_dotenv_path = find_dotenv(usecwd=True)
load_dotenv(_dotenv_path)

from openai import OpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

import tools  # tools.py -- Transfermarkt scraping helpers

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")

if not DEEPSEEK_API_KEY:
    print("ERROR: DEEPSEEK_API_KEY not found.\n")
    if _dotenv_path:
        print(f"A .env file was found at: {_dotenv_path}")
        print("But no DEEPSEEK_API_KEY=... line could be read from it. Open the file in a")
        print("text editor and make sure there's no stray space/quote/invisible character.")
    else:
        print(f"No .env file found in the working directory (searched from: {os.getcwd()}).")
        print("Check:")
        print("  1) Is the .env file in the EXACT SAME folder as engine.py?")
        print("  2) Windows sometimes hides file extensions: the file might actually be")
        print("     named '.env.txt' instead of '.env'. In Explorer, enable 'View > File")
        print("     name extensions' and check the real file name.")
    sys.exit(1)

# The set of valid team/role keys the router can select and the graph
# can dispatch to. Every team module's node MUST be registered under
# one of these exact keys (see engine.py's build_graph()).
ACTIVE_ROLE_NAMES = ["technical_team", "finance_team", "media_team", "data_team",
                      "scout_team", "legal_team", "health_team"]

MAX_DEBATE_ROUNDS = 5
TEMPERATURE = 0.4

# Background opinion-stance scoring (see sentiment_scorer.py), used by
# EVERY collect node (main engine's and every sub-team's) to quietly
# filter out neutral/undecided opinions before conflict detection.
# Imported ONCE here, centrally, so a missing/renamed file only prints
# one warning instead of one per module that used to import it separately.
try:
    import sentiment_scorer
    SENTIMENT_SCORING_AVAILABLE = True
except ImportError:
    sentiment_scorer = None
    SENTIMENT_SCORING_AVAILABLE = False
    print(
        "[WARNING] Could not import sentiment_scorer -- neutral-opinion "
        "filtering is DISABLED everywhere (main engine and every sub-team). "
        "Check that sentiment_scorer.py exists next to this file with that "
        "exact name.",
        file=sys.stderr,
    )

# There's no fixed default club -- the club under discussion is
# extracted dynamically from each message (see extract_club_name below),
# e.g. "would Salah suit Leipzig?" resolves to "RB Leipzig" for that
# conversation, without any config needed.


# =======================================================================
# LLM WRAPPER
# =======================================================================
_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def call_llm(system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = _client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        **kwargs,
    )
    return response.choices[0].message.content


# =======================================================================
# PERSONA PROMPT HELPERS
# =======================================================================
def club_phrase(club_name: Optional[str]) -> str:
    """Small helper so persona prompts read naturally whether or not a
    specific club could be identified for this conversation."""
    return (
        f"of {club_name} football club" if club_name
        else "of a football club (no specific club was identified in this conversation)"
    )


# =======================================================================
# EXTRACTION PROMPTS & HELPERS
# Used by multiple team modules (any team that needs to know which
# player/club/position is under discussion), so they live here rather
# than being duplicated per-team.
# =======================================================================
PLAYER_NAME_EXTRACT_SYSTEM = """\
You'll be given a message. Find ALL football player names mentioned in
it (could be one, could be several — e.g. "Messi or Ronaldo" mentions
both).
IMPORTANT: names are often misspelled, mistyped, or missing accents/
capitalization (e.g. "UgoChokwou" instead of "Ugochukwu"). If you can
confidently identify which real football player is meant, correct the
spelling to that player's proper/official name before returning it —
downstream tools will search for the exact name, so a corrected,
searchable spelling matters more than preserving the user's typo.
Reply ONLY with the (corrected) names separated by commas (e.g. "Lionel
Messi, Cristiano Ronaldo"), no explanation, numbering, or quotation
marks. If no clear player name appears in the message, reply ONLY "NONE".
"""

CLUB_EXTRACT_SYSTEM = """\
You'll be given a message about a football transfer/squad decision.
Identify the ONE real football club that is MAKING the decision (the
buying/deciding club) -- NOT a player's current club. For example, in
"would Salah suit Leipzig?" the deciding club is "RB Leipzig", not
Liverpool (Salah's current club).
Reply with ONLY the club's full official name (e.g. "RB Leipzig",
"Galatasaray"), nothing else. If no deciding club can be identified,
reply ONLY "NONE".
"""

POSITION_EXTRACT_SYSTEM = """\
You'll be given a message about a football transfer or squad decision.
Identify the ONE playing position being discussed (e.g. "Centre-Forward",
"Right-Back", "Central Midfield"). If the message doesn't name a position
directly but the intent is still about strengthening a general area
(e.g. "we need more goals"), infer the most fitting single position.
If nothing position-related can be determined, reply "UNKNOWN".
Reply with ONLY the position name, or "UNKNOWN" — nothing else.
"""


def extract_player_names(topic: str) -> List[str]:
    """Extracts ALL football player names mentioned in the raw user
    message. Returns an empty list if none are found."""
    raw = call_llm(PLAYER_NAME_EXTRACT_SYSTEM, topic).strip()
    if not raw or raw.upper() == "NONE":
        return []
    return [name.strip() for name in raw.split(",") if name.strip()]


def extract_club_name(topic: str) -> Optional[str]:
    """Extracts the deciding club (the one making the transfer/squad
    decision) from the raw user message, e.g. "would Salah suit Leipzig?"
    -> "RB Leipzig". Returns None if no club can be identified."""
    raw = call_llm(CLUB_EXTRACT_SYSTEM, topic).strip()
    if not raw or raw.upper() == "NONE":
        return None
    return raw


def determine_target_position(topic: str, player_names: List[str]) -> Optional[str]:
    """Figures out which playing position is under discussion. If a
    specific player was named, uses THAT player's main position (via
    Transfermarkt); otherwise asks the LLM to infer a position directly
    from the message text. If multiple players are named, only the
    first one's position is used (a reasonable v0 simplification for
    comparison-style questions).
    """
    if player_names:
        data = tools.get_player_positions(player_names[0])
        if data.get("found"):
            return data.get("main_position")

    raw = call_llm(POSITION_EXTRACT_SYSTEM, topic).strip()
    if not raw or raw.upper() == "UNKNOWN":
        return None
    return raw


# =======================================================================
# STATE
# =======================================================================
def merge_dicts(left: dict, right: dict) -> dict:
    """Custom reducer for dict-typed state fields (like role_opinions)
    that get written by multiple parallel nodes in the same fan-out step.
    operator.add doesn't work on dicts, hence this."""
    merged = dict(left or {})
    merged.update(right or {})
    return merged


class SynFCState(TypedDict, total=False):
    user_message: str
    topic: str
    club_name: Optional[str]
    # "karar" (decision) or "bilgi" (information) -- decided by the
    # Router in the SAME LLM call as department selection (no extra
    # round-trip). "bilgi" skips conflict-detection/debate/the Board
    # entirely -- see engine.py's confidence_check_node.
    mode: str
    active_roles: List[str]
    role_opinions: Annotated[Dict[str, str], merge_dicts]
    tool_contexts: Annotated[Dict[str, str], merge_dicts]
    round_count: int
    conflict: bool
    conflict_reason: str
    board_notes: Annotated[Dict[str, str], merge_dicts]  # Good Cop / Bad Cop notes
    final_decision: str


# =======================================================================
# REUSABLE NODE FACTORY
# For teams whose grounding is "look up each named player via one
# Transfermarkt tool function" (currently only finance_team, but any
# future team with the same simple pattern can reuse this instead of
# hand-writing the round-0/debate-round prompt plumbing again).
# =======================================================================
def make_tool_backed_role_node(role_name: str, system_prompt_builder, tool_fn, tool_label: str):
    """Builds a role node that, on round 0, looks up EVERY player named
    in the topic via `tool_fn`, then keeps reusing that exact same data
    across debate rounds (stored in state) instead of re-deriving it —
    otherwise the role tends to "forget" the real numbers and starts
    making new ones up each round.

    `system_prompt_builder` is a function(club_name) -> str.
    """

    def node(state: SynFCState) -> dict:
        opinions_so_far = state.get("role_opinions", {})
        tool_contexts_so_far = state.get("tool_contexts", {})
        round_count = state.get("round_count", 0)
        topic = state["topic"]
        system_prompt = system_prompt_builder(state.get("club_name"))

        state_update = {}

        if round_count == 0:
            player_names = extract_player_names(topic)
            if not player_names:
                tool_context = f"[{tool_label}: no clear player name found in the message, no data pulled]"
            else:
                sections = []
                for player_name in player_names:
                    data = tool_fn(player_name)
                    if data.get("found"):
                        details = "\n".join(
                            f"    - {k}: {v}" for k, v in data.items()
                            if k not in ("found", "source_url")
                        )
                        sections.append(
                            f"  {player_name} ({tool_label} - {data.get('source_url', '')}):\n{details}"
                        )
                    else:
                        sections.append(
                            f"  {player_name}: [{tool_label} data not found: {data.get('reason', 'unknown')}]"
                        )
                tool_context = "\n".join(sections)
            state_update["tool_contexts"] = {role_name: tool_context}
        else:
            tool_context = tool_contexts_so_far.get(role_name, "")

        if round_count == 0:
            user_prompt = (
                f"Topic: {topic}\n"
                f"Give your opinion on this, short and clear.\n\n"
                f"{tool_context}\n"
                f"Ground your opinion in this concrete data."
            )
        else:
            others = "\n".join(
                f"- {r}: {o}" for r, o in opinions_so_far.items() if r != role_name
            )
            own_previous = opinions_so_far.get(role_name, "(your previous opinion is missing)")
            user_prompt = (
                f"Round {round_count + 1}. Topic: {topic}\n\n"
                f"Reference data (use the numbers, but don't open by restating "
                f"'this hasn't changed' — go straight into your argument):\n"
                f"{tool_context or '(no data)'}\n\n"
                f"Your own opinion from the previous round:\n{own_previous}\n\n"
                f"Other roles' opinions:\n{others or '(none yet)'}\n\n"
                f"Respond SPECIFICALLY to what the other roles said and move the "
                f"discussion forward (a new angle, a counter-argument, or a "
                f"compromise). Keep your numbers consistent with previous rounds, "
                f"but don't repeat your own sentences verbatim."
            )

        opinion = call_llm(system_prompt, user_prompt)
        state_update["role_opinions"] = {role_name: opinion}
        return state_update

    return node


# =======================================================================
# GENERIC SUB-TEAM FACTORY
# This is what finance_team.py's SubRouter/agents/collect/debate/
# Director machinery generalizes into. Every future team (Technical,
# Media, Data, Scout, Legal, Health) is built by calling this ONCE with
# that team's agent list + personas, instead of hand-copying ~150 lines
# of router/collect/debate boilerplate for each one. The Board of
# Directors is the one deliberate exception -- its Good Cop/Bad Cop/
# President shape doesn't fit this "N agents -> Director synthesis"
# pattern on purpose (see board_of_directors.py).
# =======================================================================
def build_subteam(
    team_key: str,
    agent_keys: List[str],
    agent_system_builders: Dict[str, Callable[[Optional[str]], str]],
    subrouter_system: str,
    director_system_builder: Callable[[Optional[str]], str],
    agent_context_builders: Optional[Dict[str, Callable[[str, Optional[str]], str]]] = None,
    agent_output_prefixers: Optional[Dict[str, Callable[[str, Optional[str]], str]]] = None,
    director_output_prefixer: Optional[Callable[[str, Optional[str]], str]] = None,
    max_debate_rounds: int = 1,
) -> Callable[["SynFCState"], dict]:
    """Builds an entire sub-team as its own compiled LangGraph sub-graph
    (SubRouter -> N agents in parallel -> collect w/ neutral-opinion
    filtering -> a capped debate loop -> Director synthesis), and
    returns a single wrapper function ready to be registered as one
    ordinary node in the MAIN engine -- from the outside it looks
    exactly like any other team, e.g. finance_team_node.

    agent_context_builders: optional {agent_key: function(topic, club_name) -> str}
    -- injects extra tool-grounded context into that agent's ROUND-0
    prompt only. Agents with no entry here are pure-reasoning skeletons
    for now; real tools get wired in later just by adding an entry.

    agent_output_prefixers: optional {agent_key: function(topic, club_name) -> str}
    -- for content that must appear VERBATIM in the final output (e.g. a
    generated document/letter) and should NOT depend on the LLM
    faithfully reproducing it. Asking an LLM to transcribe a long block
    of text unchanged is unreliable -- it tends to summarize/paraphrase
    even when told not to. This prepends the prefixer's return value to
    the agent's LLM-generated opinion IN CODE, at round 0 only, so the
    exact text is guaranteed present regardless of what the LLM does.

    director_output_prefixer: optional function(topic, club_name) -> str
    -- same idea, one layer up. An agent_output_prefixer only guarantees
    the text reaches the DIRECTOR's input -- the Director's own LLM
    output (what actually becomes visible to the user) can still
    summarize it away. Use this when verbatim content must survive all
    the way to the final team output.
    """
    agent_context_builders = agent_context_builders or {}
    agent_output_prefixers = agent_output_prefixers or {}

    class SubTeamState(TypedDict, total=False):
        topic: str
        club_name: Optional[str]
        # "karar" or "bilgi" -- passed down from the MAIN engine's Router
        # decision (see engine.py). Lets agent personas speak in a
        # decision/judgment voice for "karar" and a plain factual voice
        # for "bilgi" -- optional for each agent_system_builder to use
        # (2-arg call tried first, falls back to 1-arg for builders that
        # don't care about mode).
        mode: str
        active_subroles: List[str]
        subrole_opinions: Annotated[Dict[str, str], merge_dicts]
        agent_contexts: Annotated[Dict[str, str], merge_dicts]
        # Cross-team reports (currently just Health's, if it ran first) --
        # passed in from the MAIN engine, read-only from this sub-graph's
        # own perspective. See engine.py's dispatch_to_roles for how this
        # gets populated.
        external_context: Dict[str, str]
        final_opinion: str
        discussion_round_count: int
        conflict: bool
        conflict_reason: str

    def subrouter_node(state: SubTeamState) -> dict:
        raw = call_llm(subrouter_system, f"Topic: {state['topic']}", json_mode=True)
        roles = []
        try:
            data = json.loads(raw)
            roles = [r for r in data.get("roles", []) if r in agent_keys]
        except (json.JSONDecodeError, AttributeError):
            roles = []
        return {
            "active_subroles": roles,
            "discussion_round_count": 0,
            "subrole_opinions": {},
            "agent_contexts": {},
        }

    def dispatch(state: SubTeamState):
        roles = state.get("active_subroles", [])
        if not roles:
            return "collect"
        return [Send(role, state) for role in roles]

    def make_agent_node(agent_key: str):
        system_builder = agent_system_builders[agent_key]
        context_builder = agent_context_builders.get(agent_key)
        output_prefixer = agent_output_prefixers.get(agent_key)

        def node(state: SubTeamState) -> dict:
            opinions_so_far = state.get("subrole_opinions", {})
            contexts_so_far = state.get("agent_contexts", {})
            round_count = state.get("discussion_round_count", 0)
            topic = state["topic"]
            club_name = state.get("club_name")
            mode = state.get("mode", "karar")
            try:
                # Try the mode-aware 2-arg form first (club_name, mode) --
                # lets a persona speak in a plain factual voice for
                # "bilgi" queries instead of always sounding like it's
                # making a judgment call. Falls back to the older 1-arg
                # form for any agent that doesn't care about mode yet.
                system_prompt = system_builder(club_name, mode)
            except TypeError:
                system_prompt = system_builder(club_name)

            state_update = {}

            if round_count == 0:
                # Tool-grounded context (if any) is fetched ONCE here and
                # CACHED in state -- otherwise later debate rounds only
                # have the agent's own prior NATURAL-LANGUAGE PARAPHRASE
                # of the data to work from, and numbers can drift each
                # time the LLM re-summarizes its own summary (e.g. an
                # injury count silently changing between rounds). This
                # mirrors the fix already applied to every hand-written
                # team node before this factory existed.
                extra = ""
                if context_builder:
                    external_context = state.get("external_context", {})
                    try:
                        # Try the 3-arg form first (topic, club_name,
                        # external_context) for context builders that want
                        # cross-team reports (e.g. technical_team's
                        # psychologist reading Health's report); fall back
                        # to the older 2-arg form for every other existing
                        # context builder, unchanged, no need to touch them.
                        extra = context_builder(topic, club_name, external_context)
                    except TypeError:
                        try:
                            extra = context_builder(topic, club_name)
                        except Exception as e:
                            extra = f"[Tool context unavailable: {e}]"
                    except Exception as e:
                        extra = f"[Tool context unavailable: {e}]"
                state_update["agent_contexts"] = {agent_key: extra}
                round0_instruction = (
                    "Report what you actually know, short and clear -- this is an "
                    "information request, not a decision, so don't frame it as a "
                    "recommendation or verdict."
                    if mode == "bilgi" else
                    "Give your assessment, short and clear."
                )
                user_prompt = (
                    f"Topic: {topic}\n{round0_instruction}\n\n{extra}"
                ).strip()
            else:
                extra = contexts_so_far.get(agent_key, "")
                others = "\n".join(
                    f"- {r}: {o}" for r, o in opinions_so_far.items() if r != agent_key
                )
                own_previous = opinions_so_far.get(agent_key, "(your previous opinion is missing)")
                user_prompt = (
                    f"Round {round_count + 1}. Topic: {topic}\n\n"
                    f"Reference data (use the numbers, but don't open by restating "
                    f"'this hasn't changed' — go straight into your argument):\n"
                    f"{extra or '(no data)'}\n\n"
                    f"Your own opinion from the previous round:\n{own_previous}\n\n"
                    f"Other agents' opinions:\n{others or '(none yet)'}\n\n"
                    f"Respond SPECIFICALLY to what others said, move the discussion "
                    f"forward, but keep any numbers EXACTLY consistent with the "
                    f"reference data and your previous round -- don't repeat your "
                    f"own sentences verbatim, but don't silently change a figure either."
                )

            opinion = call_llm(system_prompt, user_prompt)

            # Round-0-only: if this agent has a registered output
            # prefixer, splice its EXACT return value onto the front of
            # the LLM's opinion IN CODE -- not by asking the LLM to
            # reproduce it, which is unreliable for long verbatim text.
            if round_count == 0 and output_prefixer:
                try:
                    prefix = output_prefixer(topic, club_name)
                except Exception as e:
                    prefix = f"[Output prefix unavailable: {e}]"
                if prefix:
                    opinion = f"{prefix}\n\n---\n\n{opinion}"

            state_update["subrole_opinions"] = {agent_key: opinion}
            return state_update

        return node

    def collect_node(state: SubTeamState) -> dict:
        opinions = state.get("subrole_opinions", {})
        if len(opinions) <= 1:
            return {
                "conflict": False,
                "conflict_reason": "Not enough opinions to compare.",
                "active_subroles": list(opinions.keys()),
            }

        decisive = dict(opinions)
        if SENTIMENT_SCORING_AVAILABLE:
            try:
                for k, text in opinions.items():
                    score = sentiment_scorer.score_opinion(text)
                    if sentiment_scorer.is_neutral(score):
                        decisive.pop(k, None)
            except Exception:
                decisive = dict(opinions)

        if len(decisive) <= 1:
            return {
                "conflict": False,
                "conflict_reason": "Remaining decisive opinions too few to compare (the rest were neutral).",
                "active_subroles": list(decisive.keys()),
            }

        joined = "\n".join(f"{r}: {o}" for r, o in decisive.items())
        raw = call_llm(
            f"You'll be given opinions from the '{team_key}' sub-team's agents on "
            f"the same topic. Detect whether there's a REAL disagreement (not just "
            f"a difference in tone/emphasis).\nReply ONLY as JSON: "
            f'{{"conflict": true or false, "reason": "brief justification"}}',
            f"Opinions:\n{joined}",
            json_mode=True,
        )
        try:
            data = json.loads(raw)
            conflict = bool(data.get("conflict", False))
            reason = data.get("reason", "")
        except (json.JSONDecodeError, AttributeError):
            conflict, reason = False, "Conflict detection could not be parsed, assuming no conflict."

        return {"conflict": conflict, "conflict_reason": reason, "active_subroles": list(decisive.keys())}

    def after_collect(state: SubTeamState) -> str:
        if state.get("discussion_round_count", 0) >= max_debate_rounds:
            return "director"
        if state.get("conflict"):
            return "debate_dispatch"
        return "director"

    def debate_dispatch_node(state: SubTeamState) -> dict:
        return {"discussion_round_count": state.get("discussion_round_count", 0) + 1}

    def director_node(state: SubTeamState) -> dict:
        opinions = state.get("subrole_opinions", {})
        joined = "\n".join(f"{r}: {o}" for r, o in opinions.items()) if opinions else "(no agents were consulted)"
        mode = state.get("mode", "karar")
        closing_instruction = (
            "Summarize the team's factual findings clearly -- do NOT phrase "
            "this as a recommendation, verdict, or judgment call; this is an "
            "information request, not a decision."
            if mode == "bilgi" else
            "State this team's final position with a brief justification."
        )
        user_prompt = (
            f"Topic: {state['topic']}\n\n"
            f"Agent opinions:\n{joined}\n\n"
            f"Conflict note: {state.get('conflict_reason', '(none)')}\n\n"
            f"{closing_instruction}"
        )
        try:
            director_system = director_system_builder(state.get("club_name"), mode)
        except TypeError:
            director_system = director_system_builder(state.get("club_name"))
        decision = call_llm(director_system, user_prompt)
        return {"final_opinion": decision}

    graph = StateGraph(SubTeamState)
    graph.add_node("subrouter", subrouter_node)
    for key in agent_keys:
        graph.add_node(key, make_agent_node(key))
    graph.add_node("collect", collect_node)
    graph.add_node("debate_dispatch", debate_dispatch_node)
    graph.add_node("director", director_node)

    graph.add_edge(START, "subrouter")
    graph.add_conditional_edges("subrouter", dispatch, agent_keys + ["collect"])
    for key in agent_keys:
        graph.add_edge(key, "collect")
    graph.add_conditional_edges("collect", after_collect, ["debate_dispatch", "director"])
    graph.add_conditional_edges("debate_dispatch", dispatch, agent_keys + ["collect"])
    graph.add_edge("director", END)

    compiled = graph.compile()

    def team_node(state: "SynFCState") -> dict:
        topic = state["topic"]
        club_name = state.get("club_name")
        mode = state.get("mode", "karar")  # safe default matches router_node's own default

        # Round 0: every department runs in TRUE parallel (all dispatched
        # from the same state snapshot), so role_opinions["health_team"]
        # genuinely doesn't exist yet -- nobody can see it, by design.
        # By the time a DEBATE round runs (conflict detected), round 0
        # has already finished for everyone, so Health's report (if
        # Health was selected) is already sitting in role_opinions --
        # this naturally attaches it then, no explicit round-tracking
        # or special sequencing needed.
        external_context = {}
        health_report = state.get("role_opinions", {}).get("health_team")
        if health_report and team_key != "health_team":
            external_context = {"health_team": health_report}

        result = compiled.invoke({
            "topic": topic, "club_name": club_name,
            "external_context": external_context, "mode": mode,
        })
        final_opinion = result.get("final_opinion", "")

        # Same reasoning as agent_output_prefixers, one layer up: the
        # Director's OWN LLM output is what actually becomes visible to
        # the user (role_opinions[team_key]) -- the agent's opinion
        # (even if code-prefixed) is only an INPUT to the Director, not
        # the final output. If verbatim content must survive all the
        # way to what the user sees, it has to be spliced here too.
        if director_output_prefixer:
            try:
                prefix = director_output_prefixer(topic, club_name)
            except Exception as e:
                prefix = f"[Output prefix unavailable: {e}]"
            if prefix:
                final_opinion = f"{prefix}\n\n---\n\n{final_opinion}"

        return {"role_opinions": {team_key: final_opinion}}

    return team_node