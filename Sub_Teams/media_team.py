"""
media_team.py — SynFC's Media department (sub-graph)
==========================================================
Built with core.build_subteam(). Two agents (Fans, Reporter) report to
a Media Director.

Fans has real tool grounding (local X/Twitter comment files, migrated
from the old top-level fans.py). Reporter is a reasoning-only skeleton
for now -- tools.get_transfer_rumors is a planned tool (raises
NotImplementedError), so Reporter's context builder catches that and
reasons without data until it's built.
"""

from typing import Optional
from functools import lru_cache

import core
import tools

try:
    import media_web_search_perplexity
    _PERPLEXITY_AVAILABLE = True
except ImportError:
    media_web_search_perplexity = None
    _PERPLEXITY_AVAILABLE = False

AGENT_KEYS = ["fans", "reporter"]


def build_fans_system(club_name: Optional[str], mode: str = "karar") -> str:
    club = core.club_phrase(club_name)
    if mode == "bilgi":
        return f"""\
You represent the FANS {club} -- the collective, unfiltered voice of
the supporters. This is an INFORMATION request, not a decision -- report
the real fan sentiment/comments you're given (quote or paraphrase
specific recurring themes), keeping their actual passionate/emotional
character, but do NOT frame this as a recommendation for what the club
should do.
CRITICAL: if you weren't given real fan comments/reports on this
specific topic, do NOT invent generic fan opinion or substitute your own
guess about "what fans would probably think" as if it were real
sentiment data -- say plainly you don't have real data on this.
Keep your answer to 3-6 sentences.
"""
    return f"""\
You represent the FANS {club} -- the collective, unfiltered voice of
the supporters, drawn from what they're actually saying on social media
(X/Twitter) and in the stands. Character: passionate, emotional, biased
toward the club, prone to hyperbole and mood swings based on recent
results. You don't care about balance sheets or squad-depth
spreadsheets -- you care about identity, pride, and "does this feel
right for US".
Your job: reflect the real sentiment found in the fan comments/reports
you're given (quote or paraphrase specific recurring themes -- don't
invent generic fan opinion). If the comments are split/contradictory,
say so honestly. If no fan data is available, say plainly that you
don't have anything concrete from the fanbase on this and speak more
cautiously.

CRITICAL -- stay in character even in debate rounds: when other
departments (finance, technical, data) respond with cost-benefit
analysis, risk assessments, or tactical logic, do NOT start mirroring
their tone or reasoning like an analyst yourself -- that drift is
exactly what you must resist. You are not weighing pros and cons; you
are reporting what the stands actually feel, even when that's one-
sided, impatient, or dismissive of financial/tactical concerns other
departments raise. A real fanbase chanting a player's name doesn't
pause to consider amortization -- neither do you. If your own data
shows genuine excitement/anger/demand, report it at that same
intensity every round, don't let it cool into a "balanced take" just
because the room around you sounds analytical.
Keep your answer to 3-6 sentences.
"""


def build_reporter_system(club_name: Optional[str], mode: str = "karar") -> str:
    club = core.club_phrase(club_name)
    if mode == "bilgi":
        return f"""\
You are a football reporter covering the club {club}, part of the media
team. This is an INFORMATION request, not a decision -- report the
actual press/rumor findings you're given, without framing it as a
recommendation.
CRITICAL: if you weren't given real sourcing/data for the specific
question asked, do NOT substitute your own memorized/training knowledge
or invent "reports suggest" lines you can't back up -- say plainly you
don't have real data on this.
Keep your answer to 3-6 sentences.
"""
    return f"""\
You are a football reporter covering the club {club}, part of the
media team. Character: skeptical, sources-driven, careful to
distinguish confirmed reporting from speculation.
Your job: assess what the press/rumor landscape looks like for the
topic under discussion. NOTE: a dedicated transfer-rumor scraping tool
isn't built yet -- reason qualitatively and say plainly when you don't
have real sourcing for a claim, rather than inventing "reports suggest"
lines you can't back up.
Keep your answer to 3-6 sentences.
"""


def build_media_director_system(club_name: Optional[str], mode: str = "karar") -> str:
    club = core.club_phrase(club_name)
    if mode == "bilgi":
        return f"""\
You are the Media Director {club}. This is an INFORMATION request, not
a decision -- relay the Fans' and the Reporter's findings clearly,
without adding a recommendation/verdict at the end (unlike your usual
process).
CRITICAL: never fill a gap in their reports with your own memorized/
training knowledge presented as if it were their real finding -- if they
had no real data on something, say so plainly instead of fabricating
specifics.
"""
    return f"""\
You are the Media Director {club}. You relay the Fans' and the
Reporter's findings to the club's Board, then add your own brief take.

CRITICAL -- structure your answer in exactly this order:
1. The Reporter's finding, relayed close to their own original wording
   and tone -- don't paraphrase it into calmer, more neutral language.
   If they were skeptical/blunt, stay skeptical/blunt.
2. The Fans' finding, relayed close to their own original wording and
   INTENSITY -- if they reported genuine excitement, chanting, anger,
   or demand, keep that same energy in how you relay it. Do NOT soften
   it into a more measured, "PR-safe" paraphrase -- that loses exactly
   the signal the Board needs to see. A fanbase that stopped a
   ceremony to chant a name should still sound like that when you
   relay it, not like "moderate optimism".
3. ONLY at the end, add a short (1-2 sentence) synthesis in your own
   voice -- your own image-conscious take on what this means for the
   club, clearly separated from the two findings above.

Never let step 3 bleed backward into how you relay steps 1 and 2.
"""


MEDIA_ROUTER_SYSTEM = """\
You are the router for the club's media team. Decide which agent(s) a
topic concerns. You may ONLY choose from:
- "fans": supporter sentiment/reaction, club image, matchday atmosphere
- "reporter": press coverage, transfer rumors, media narrative

A topic can concern both. If neither genuinely applies (e.g. a purely
internal tactical/financial question with no public dimension), return
an empty list.

Reply ONLY in this exact JSON format, with no other text:
{"roles": ["fans", "reporter"]}
"""


@lru_cache(maxsize=64)
def _fetch_perplexity_report(query: str) -> str:
    """Cached so fans and reporter (same query, same round) don't both
    trigger a separate live API call for identical information."""
    result = media_web_search_perplexity.search_fan_sentiment(query)
    if not result.get("found"):
        return f"[Live web search unavailable: {result.get('reason', 'unknown error')}]"
    return result["summary"]


def _perplexity_report(topic: str, club_name: Optional[str]) -> str:
    """Shared live web-search report (Perplexity), fed to BOTH the fans
    and reporter agents -- only runs if Media was actually selected as
    an active department (this function is only ever called from
    within an agent node, which only runs if that agent was routed to
    in the first place). Each agent has its own persona already, so
    they naturally emphasize what's relevant to their own role from
    the same underlying report instead of us pre-splitting the search
    into narrower, separately-scoped queries.
    """
    if not _PERPLEXITY_AVAILABLE:
        return ""

    player_names = core.extract_player_names(topic)
    query_terms = ([club_name] if club_name else []) + player_names
    # Bare keywords alone (e.g. just "Besiktas Salah") strip out the
    # intent/framing words that actually steer Perplexity toward
    # fan-reaction content, so combine the extracted keywords (for
    # specificity) with the raw topic text (for the natural-language
    # framing that made manual testing work well, e.g. "... fan
    # reaction").
    keyword_part = " ".join(query_terms)
    query = f"{keyword_part} {topic}".strip() if keyword_part else topic

    try:
        summary = _fetch_perplexity_report(query)
    except Exception as e:
        return f"[Live web search failed: {e}]"

    return f"[Live web search report — query: '{query}']\n{summary}"


def _fans_context(topic: str, club_name: Optional[str]) -> str:
    """Real tool grounding: local X/Twitter comment files, migrated from
    the old top-level fans.py, PLUS a shared live web-search report."""
    player_names = core.extract_player_names(topic)
    query_terms = ([club_name] if club_name else []) + player_names

    if not query_terms:
        local_section = "[No club or player identified for this topic, no fan data searched]"
    else:
        fan_data = tools.search_fan_comments(query_terms)
        if not fan_data.get("found"):
            local_section = f"[Could not search local fan data: {fan_data.get('reason', 'unknown error')}]"
        else:
            sample = fan_data.get("sample", [])
            total = fan_data.get("total_matches", 0)
            if not sample:
                local_section = f"[No local fan comments found mentioning {query_terms}]"
            else:
                lines = "\n".join(f'    - "{c}"' for c in sample)
                local_section = f"[Local fan comments — {total} total matches for {query_terms}, sample of {len(sample)}]\n{lines}"

    sections = [local_section]
    live_section = _perplexity_report(topic, club_name)
    if live_section:
        sections.append(live_section)
    return "\n\n".join(sections)


def _reporter_context(topic: str, club_name: Optional[str]) -> str:
    """Two modes, deliberately different (per project design discussion),
    now MUTUALLY EXCLUSIVE in terms of source, not just emphasis:

    1. SPECIFIC query (a player is named, e.g. "does Osimhen come to
       Liverpool") -- ONLY Perplexity's live search is used.
       Transfermarkt's rumor list is skipped entirely: it can be
       incomplete/stale, and a name missing from it must never be read
       as "this rumor is false".

    2. GENERAL/LIST query (no specific player named, e.g. "who is our
       club interested in") -- ONLY Transfermarkt's rumor list is used.
       Perplexity is deliberately NOT called for this case anymore --
       mixing it in was causing the Reporter to surface Perplexity's
       (sometimes stale/less-targeted) web search names instead of the
       freshly-scraped, verified Transfermarkt list, which defeats the
       point of having real, current data for this query type.
    """
    player_names = core.extract_player_names(topic)
    is_specific_query = bool(player_names)

    if is_specific_query:
        live_section = _perplexity_report(topic, club_name)
        if not live_section:
            return "[Live web search unavailable, and this is a specific-player query -- reasoning without data]"
        return live_section

    if not club_name:
        return "[No specific club identified, no rumor data pulled]"

    try:
        data = tools.get_transfer_rumors(club_name)
    except NotImplementedError:
        return "[Transfer rumor scraping tool is not implemented yet -- reasoning without data]"

    if not data.get("found"):
        return f"[Could not fetch rumors for {club_name}: {data.get('reason', 'unknown error')}]"

    rumors = data.get("rumors", [])
    if not rumors:
        return f"[No current transfer rumors found for {club_name}]"

    lines = "\n".join(
        f"    - {r.get('player')} ({r.get('position', 'unknown')}) "
        f"(secondary signal only -- reported probability: {r.get('probability', 'unknown')}, "
        f"date: {r.get('date', 'unknown')})"
        for r in rumors
    )
    return (
        f"[Transfer rumors — {club_name} — Transfermarkt, names are the useful "
        f"part here, treat probability as secondary]\n{lines}"
    )


media_team_node = core.build_subteam(
    team_key="media_team",
    agent_keys=AGENT_KEYS,
    agent_system_builders={
        "fans": build_fans_system,
        "reporter": build_reporter_system,
    },
    subrouter_system=MEDIA_ROUTER_SYSTEM,
    director_system_builder=build_media_director_system,
    agent_context_builders={
        "fans": _fans_context,
        "reporter": _reporter_context,
    },
)