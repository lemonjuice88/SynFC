"""
finance_team.py — SynFC's Finance department (sub-graph)
=============================================================
This is no longer a single CFO persona -- it's a small, self-contained
LangGraph of its own (Accounting + Investor + a Finance SubRouter + a
Finance Director), compiled once and wrapped so the MAIN engine
(synfc.py) still sees it as a single ordinary node called
"finance_team", exactly like before. This is the template the other 7
teams in the long-term vision will eventually follow.

Internal flow (mirrors the main engine's own router/collect/debate
pattern, just scoped down to 2 agents and a 2-round cap instead of 4
agents / 5 rounds):

    START -> finance_router -> (fan-out: accounting / investor)
           -> finance_collect -> conflict (after neutral-opinion filtering)?
                 yes, round < 1 -> finance_debate_dispatch -> agents again -> finance_collect
                 no / round limit hit -> finance_director -> END

The rest of SynFC never sees any of this -- it only calls
finance_team_node(state) and gets back a single opinion string, same as
every other team.
"""

import json
import datetime
from typing import TypedDict, Annotated, Dict, List, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

import core
import calculator
import salary_data
from urllib.parse import urlparse


def _transfermarkt_url_to_slug(transfermarkt_url: Optional[str]) -> Optional[str]:
    """Extracts the player-name slug from a Transfermarkt profile URL
    (e.g. "https://www.transfermarkt.com/victor-osimhen/profil/spieler/401923"
    -> "victor-osimhen") -- reused AS-IS as salary_data.py's SalaryLeaks
    slug, since both sites happen to use the same "firstname-lastname"
    convention. This avoids guessing the slug a second time from
    whatever fragment of the name the user actually typed (e.g.
    "osimhen") -- we let Transfermarkt's own search do that
    name-resolution work once, then reuse its answer.
    """
    if not transfermarkt_url:
        return None
    path = urlparse(transfermarkt_url).path  # "/victor-osimhen/profil/spieler/401923"
    parts = [p for p in path.split("/") if p]  # ["victor-osimhen", "profil", "spieler", "401923"]
    return parts[0] if parts else None


# =======================================================================
# FINANCE SUB-GRAPH STATE
# Deliberately separate from core.SynFCState -- the main graph has no
# idea "accounting" or "investor" exist, it only ever sees the single
# "finance_team" node. This state is private to this file.
# =======================================================================
class FinanceState(TypedDict, total=False):
    topic: str
    mode: str  # "karar" or "bilgi", passed down from the main engine's Router
    # Cross-team edge: Health department's report, IF it's already
    # available (only true during a debate round). Same pattern as
    # core.py's build_subteam -- see investor_node for the only current
    # reader.
    external_context: Dict[str, str]
    active_finance_roles: List[str]
    finance_role_opinions: Annotated[Dict[str, str], core.merge_dicts]
    total_finance_decision: str
    discussion_round_count: int
    conflict: bool
    conflict_reason: Optional[str]


ACTIVE_FINANCE_ROLES = ["accounting", "investor"]
MAX_FINANCE_DEBATE_ROUNDS = 1  # round 0 + at most 1 debate round = 2 total, per design


# =======================================================================
# PROMPTS
# =======================================================================
FINANCE_ROUTER_SYSTEM = """\
You are the router for the club's finance team. Decide which finance
agent(s) a topic concerns. You may ONLY choose from:
- "accounting": the club's current financial reality -- market value,
  historical transfer fees, wage/amortization cost calculations
- "investor": the potential return/risk of a decision, and the
  opportunity cost of NOT acting (e.g. a rival securing the same asset)

A topic can concern both (e.g. "should we sign this player" needs both
a cost reality check from accounting and a risk/return read from the
investor). If neither genuinely applies, return an empty list.

Reply ONLY in this exact JSON format, with no other text:
{"roles": ["accounting", "investor"]}
"""

def build_accounting_system(mode: str = "karar") -> str:
    if mode == "bilgi":
        return f"""\
You are the club's accountant, part of the finance team. This is an
INFORMATION request, not a decision -- report the club's actual
financial data (market value, transfer fees, wage figures, amortization
numbers) exactly as given to you, without framing it as a
recommendation.
CRITICAL: if you weren't given real financial data for what's actually
being asked, do NOT substitute your own memorized/training knowledge
(e.g. recalling an approximate market value from what you learned
during training) and present it as if it were a real, current data
lookup -- say plainly you don't have real data on this specific
question, rather than confidently fabricating numbers.
Keep your answer to 3-6 sentences.
"""
    return f"""\
You are the club's accountant, part of the finance team. Character:
precise, fact-based, allergic to speculating without real numbers.
Your job: report the club's current financial reality for whatever is
being discussed -- market value, historical transfer fees, and cost
calculations, grounded in real data when it's available to you. If you
don't have real data for something, say so plainly rather than guessing.
Keep your answer to 3-6 sentences.
"""


def build_investor_system(mode: str = "karar") -> str:
    if mode == "bilgi":
        return f"""\
You are the club's investment analyst, part of the finance team. This is
an INFORMATION request, not a decision -- report what's actually known
(e.g. the accountant's real figures, if visible to you) without framing
it as a risk/return recommendation.
CRITICAL: dedicated revenue-projection and opportunity-cost models
aren't built yet -- do NOT substitute your own memorized/training
knowledge for a real number and present it as verified fact. Say
plainly you don't have real data on this, rather than confidently
fabricating a specific figure.
Keep your answer to 3-6 sentences.
"""
    return f"""\
You are the club's investment analyst, part of the finance team.
Character: opportunity-focused, thinks in terms of risk/return and what
the club stands to lose by NOT acting.
Your job: assess the potential return and risk profile of a decision,
and weigh the cost of missing out against a rival securing the same
asset. NOTE: dedicated revenue-projection and opportunity-cost models
aren't built yet -- reason qualitatively from whatever context you have
(including the accountant's figures, once you can see them) rather than
inventing precise numbers you don't actually have. When you're given
the Health department's report (this only happens once there's an
active debate -- not on your very first pass), factor real injury risk
directly into your risk assessment -- an injury-prone player is a
higher-risk investment regardless of upside, and you should say so
plainly when the data supports it.
Keep your answer to 3-6 sentences.
"""


def build_finance_director_system(mode: str = "karar") -> str:
    if mode == "bilgi":
        return f"""\
You are the Finance Director. This is an INFORMATION request, not a
decision -- relay what the Accountant and Investor actually reported,
organized clearly, without framing it as a recommendation/verdict.
CRITICAL: never fill a gap in their reports with your own memorized/
training knowledge presented as if it were their real finding -- if they
had no real data on something, say so plainly instead of fabricating
specific numbers.
"""
    return f"""\
You are the Finance Director. You synthesize the Accountant's and the
Investor's input into ONE final financial position for the club's
Board. Character: decisive -- you weigh both fiscal caution and
opportunity cost, but you must commit to a clear recommendation, not a
wishy-washy summary of both sides.
Given both reports (and any conflict between them), state the finance
team's single position with a brief justification.
"""

FINANCE_CONFLICT_SYSTEM = """\
You'll be given the opinions of the club's Accountant and/or Investor
on the same topic. Detect whether there's a REAL disagreement between
them (a genuine clash of recommendations, not just a difference in tone
or emphasis).

Reply ONLY in this exact JSON format:
{"conflict": true or false, "reason": "brief justification"}
"""


# =======================================================================
# SUB-GRAPH NODES
# =======================================================================
def _estimate_contract_years(contract_expires: Optional[str]) -> Optional[int]:
    """Reads remaining contract length from a Transfermarkt "contract
    expires" date string (format varies, commonly DD.MM.YYYY).

    Returns the real number of years if the date was actually parsed,
    or None if it wasn't (missing/unparseable) -- NO fallback default
    anymore. The caller must skip the amortization calculation entirely
    when this returns None, rather than silently guessing a number."""
    if not contract_expires:
        return None
    # "%d/%m/%Y" is tried before "%m/%d/%Y" deliberately: for an
    # ambiguous date like "05/06/2029", trying month-first could
    # silently misread day/month-order data (Transfermarkt is European,
    # so day/month is the far more likely real convention here) as the
    # wrong date instead of failing loudly. Day-first minimizes that
    # silent-misparse risk.
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            expiry_date = datetime.datetime.strptime(contract_expires.strip(), fmt).date()
            years_remaining = round((expiry_date - datetime.date.today()).days / 365.25)
            return max(years_remaining, 1)
        except ValueError:
            continue
    return None


def finance_team_router_node(state: FinanceState) -> dict:
    raw = core.call_llm(
        FINANCE_ROUTER_SYSTEM,
        f"Topic: {state['topic']}",
        json_mode=True,
    )
    roles = []
    try:
        data = json.loads(raw)
        roles = [r for r in data.get("roles", []) if r in ACTIVE_FINANCE_ROLES]
    except (json.JSONDecodeError, AttributeError):
        roles = []

    return {
        "active_finance_roles": roles,
        "discussion_round_count": 0,
        "finance_role_opinions": {},
    }


def dispatch_to_finance_roles(state: FinanceState):
    """Conditional edge: fans out to every agent in active_finance_roles
    in parallel via Send. If the list is empty, goes straight to collect."""
    roles = state.get("active_finance_roles", [])
    if not roles:
        return "finance_collect"
    return [Send(role, state) for role in roles]


def accounting_node(state: FinanceState) -> dict:
    opinions_so_far = state.get("finance_role_opinions", {})
    round_count = state.get("discussion_round_count", 0)
    topic = state["topic"]
    mode = state.get("mode", "karar")

    if round_count == 0:
        # Ground the first-round opinion in real data if a player is named.
        player_names = core.extract_player_names(topic)
        if player_names:
            data = core.tools.get_player_financials(player_names[0])
            if data.get("found"):
                tool_context = (
                    f"[Transfermarkt data for {player_names[0]}]\n"
                    f"  market value: {data.get('market_value')}\n"
                    f"  contract expires: {data.get('contract_expires')}\n"
                    f"  current club: {data.get('current_club')}"
                )

                # Turn the raw market value into a real amortization
                # breakdown (calculator.py) -- contract LENGTH comes
                # directly from "contract expires". If that date isn't
                # available/unparseable, we DON'T guess -- the
                # amortization calculation is skipped entirely and we
                # say so plainly. This is fee-only (transfer
                # amortization); REAL wage data is added separately
                # right below, via SalaryLeaks (salary_data.py).
                fee_eur = calculator.parse_money_string_to_eur(data.get("market_value", ""))
                contract_years = _estimate_contract_years(data.get("contract_expires"))
                if fee_eur is not None and contract_years is not None:
                    amortization = calculator.calculate_transfer_amortization(
                        transfer_fee_eur=fee_eur,
                        contract_years=contract_years,
                    )
                    tool_context += (
                        f"\n\n[Amortization — transfer-fee component only]\n"
                        f"  contract length: {contract_years} year(s) (from the real contract expiry date)\n"
                        f"  annual amortization: {amortization['annual_amortization_eur']:,.0f} EUR/year\n"
                        f"  NOTE: this specific figure covers ONLY the transfer-fee amortization -- "
                        f"check the wage data section below (if present) for the salary component "
                        f"before stating a total annual cost."
                    )
                else:
                    tool_context += (
                        f"\n\n[Amortization calculation skipped: no real contract length data available]"
                    )

                # Real wage data (SalaryLeaks) -- reuse the SAME slug
                # Transfermarkt's own search already resolved (e.g.
                # "victor-osimhen"), instead of re-guessing it from
                # whatever fragment of the name the user actually typed
                # (e.g. just "osimhen", which wouldn't match on its own).
                slug = _transfermarkt_url_to_slug(data.get("source_url"))
                salary_result = salary_data.search_player_salary(player_names[0], slug_override=slug)
                if salary_result.get("found"):
                    tool_context += (
                        f"\n\n[Wage data — SalaryLeaks]\n"
                        f"  weekly wage: {salary_result.get('current_weekly')}\n"
                        f"  annual wage: {salary_result.get('current_annual')}\n"
                        f"  summary: {salary_result.get('summary')}"
                    )
                else:
                    tool_context += (
                        f"\n\n[Wage data unavailable: {salary_result.get('reason', 'unknown')}]"
                    )
            else:
                tool_context = f"[No Transfermarkt data found for {player_names[0]}: {data.get('reason', 'unknown')}]"
        else:
            tool_context = "[No specific player identified for this topic, no data pulled]"

        round0_instruction = (
            "Report what you actually know, short and clear -- this is an "
            "information request, not a decision."
            if mode == "bilgi" else
            "Give your accounting assessment, short and clear."
        )
        user_prompt = (
            f"Topic: {topic}\n"
            f"{round0_instruction}\n\n"
            f"{tool_context}\n"
            f"Ground your numbers in this data where possible."
        )
    else:
        others = "\n".join(
            f"- {r}: {o}" for r, o in opinions_so_far.items() if r != "accounting"
        )
        own_previous = opinions_so_far.get("accounting", "(your previous opinion is missing)")
        user_prompt = (
            f"Round {round_count + 1}. Topic: {topic}\n\n"
            f"Your own opinion from the previous round:\n{own_previous}\n\n"
            f"The investor's opinion:\n{others or '(none yet)'}\n\n"
            f"Respond specifically to it, don't repeat your own sentences verbatim."
        )

    opinion = core.call_llm(build_accounting_system(mode), user_prompt)
    return {"finance_role_opinions": {"accounting": opinion}}


def investor_node(state: FinanceState) -> dict:
    opinions_so_far = state.get("finance_role_opinions", {})
    round_count = state.get("discussion_round_count", 0)
    topic = state["topic"]
    mode = state.get("mode", "karar")

    # Cross-team edge: Health's report, if already available (only true
    # during a debate round -- round 0 runs every MAIN-engine department
    # in parallel, so nothing to read yet). Injury risk is directly
    # relevant to investment risk, hence wiring this into Investor
    # specifically (not Accounting, which stays purely financial-data-
    # driven).
    health_report = state.get("external_context", {}).get("health_team")
    health_context = (
        f"\n\n[Health department's report]\n{health_report}"
        if health_report else
        "\n\n[No Health department report available yet -- reasoning without injury-risk context]"
    )

    if round_count == 0:
        round0_instruction = (
            "Report what you actually know, short and clear -- this is an "
            "information request, not a decision."
            if mode == "bilgi" else
            "Give your investment/opportunity-cost assessment, short and clear."
        )
        user_prompt = f"Topic: {topic}\n{round0_instruction}{health_context}"
    else:
        others = "\n".join(
            f"- {r}: {o}" for r, o in opinions_so_far.items() if r != "investor"
        )
        own_previous = opinions_so_far.get("investor", "(your previous opinion is missing)")
        user_prompt = (
            f"Round {round_count + 1}. Topic: {topic}\n\n"
            f"Your own opinion from the previous round:\n{own_previous}\n\n"
            f"The accountant's opinion:\n{others or '(none yet)'}\n"
            f"{health_context}\n\n"
            f"Respond specifically to it, don't repeat your own sentences verbatim."
        )

    opinion = core.call_llm(build_investor_system(mode), user_prompt)
    return {"finance_role_opinions": {"investor": opinion}}


def finance_collect_node(state: FinanceState) -> dict:
    opinions = state.get("finance_role_opinions", {})

    if len(opinions) <= 1:
        return {
            "conflict": False,
            "conflict_reason": "Not enough opinions to compare.",
            "active_finance_roles": list(opinions.keys()),
        }

    # Same background neutral-opinion filtering as the main engine (see
    # sentiment_scorer.py) -- invisible to the user, only affects
    # whether an agent keeps debating.
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
            "active_finance_roles": list(decisive_opinions.keys()),
        }

    joined = "\n".join(f"{r}: {o}" for r, o in decisive_opinions.items())
    raw = core.call_llm(FINANCE_CONFLICT_SYSTEM, f"Opinions:\n{joined}", json_mode=True)

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
        "active_finance_roles": list(decisive_opinions.keys()),
    }


def after_finance_collect(state: FinanceState) -> str:
    if state.get("discussion_round_count", 0) >= MAX_FINANCE_DEBATE_ROUNDS:
        return "finance_director"
    if state.get("conflict"):
        return "finance_debate_dispatch"
    return "finance_director"


def finance_debate_dispatch_node(state: FinanceState) -> dict:
    return {"discussion_round_count": state.get("discussion_round_count", 0) + 1}


def finance_director_node(state: FinanceState) -> dict:
    opinions = state.get("finance_role_opinions", {})
    joined = "\n".join(f"{r}: {o}" for r, o in opinions.items()) if opinions else "(no agents were consulted)"
    mode = state.get("mode", "karar")

    closing_instruction = (
        "Summarize the finance team's factual findings clearly -- do NOT "
        "phrase this as a recommendation or verdict; this is an "
        "information request, not a decision."
        if mode == "bilgi" else
        "State the finance team's final position with a brief justification."
    )
    user_prompt = (
        f"Topic: {state['topic']}\n\n"
        f"Agent opinions:\n{joined}\n\n"
        f"Conflict note: {state.get('conflict_reason', '(none)')}\n\n"
        f"{closing_instruction}"
    )
    decision = core.call_llm(build_finance_director_system(mode), user_prompt)
    return {"total_finance_decision": decision}


# =======================================================================
# SUB-GRAPH ASSEMBLY (compiled once, at import time)
# =======================================================================
def build_finance_subgraph():
    graph = StateGraph(FinanceState)

    graph.add_node("finance_router", finance_team_router_node)
    graph.add_node("accounting", accounting_node)
    graph.add_node("investor", investor_node)
    graph.add_node("finance_collect", finance_collect_node)
    graph.add_node("finance_debate_dispatch", finance_debate_dispatch_node)
    graph.add_node("finance_director", finance_director_node)

    graph.add_edge(START, "finance_router")

    graph.add_conditional_edges(
        "finance_router", dispatch_to_finance_roles, ACTIVE_FINANCE_ROLES + ["finance_collect"]
    )

    for role in ACTIVE_FINANCE_ROLES:
        graph.add_edge(role, "finance_collect")

    graph.add_conditional_edges(
        "finance_collect", after_finance_collect, ["finance_debate_dispatch", "finance_director"]
    )

    graph.add_conditional_edges(
        "finance_debate_dispatch", dispatch_to_finance_roles, ACTIVE_FINANCE_ROLES + ["finance_collect"]
    )

    graph.add_edge("finance_director", END)

    return graph.compile()


_finance_subgraph = build_finance_subgraph()


# =======================================================================
# THE WRAPPER — this is the ONLY thing the main engine (synfc.py) ever
# imports/calls. From the outside, "finance_team" still looks like a
# single ordinary node, exactly like technical_team/analytics_team/fans.
# =======================================================================
def finance_team_node(state: core.SynFCState) -> dict:
    # Cross-team edge: reuse the SAME pattern core.py's build_subteam
    # uses -- Health's report, if it's already finished (only true
    # during a debate round, not round 0, since round 0 runs every
    # MAIN-engine department in parallel).
    health_report = state.get("role_opinions", {}).get("health_team")
    external_context = {"health_team": health_report} if health_report else {}

    result = _finance_subgraph.invoke({
        "topic": state["topic"],
        "mode": state.get("mode", "karar"),
        "external_context": external_context,
    })
    return {"role_opinions": {"finance_team": result.get("total_finance_decision", "")}}