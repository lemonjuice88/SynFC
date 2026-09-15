"""
scout_team.py — SynFC's Scouting department (sub-graph)
=============================================================
Built with core.build_subteam(). Five regional agents (Asia, Europe,
Africa, South America, North America) report to a Scout Director.

Real tool grounding: when a specific player is named, ALL five agents
can see their real Transfermarkt market value/contract data + real
2024/25 performance stats (same sources as Data/Technical teams use).
This ISN'T region-filtered (we don't have a reliable way to map a
player to "their region" from current data sources) -- whichever
regional scout the router picks for a topic gets the same real data,
grounded in fact rather than reasoning from general knowledge alone.
True region-specific news/potential-assessment tools are still a v2
goal (see project notes).
"""

from typing import Optional

import core
import tools
import player_stats_dataset

AGENT_KEYS = ["scout_asia", "scout_europe", "scout_africa", "scout_south_america", "scout_north_america"]

_REGION_LABELS = {
    "scout_asia": "Asia",
    "scout_europe": "Europe",
    "scout_africa": "Africa",
    "scout_south_america": "South America",
    "scout_north_america": "North America",
}


def _build_regional_scout_system(agent_key: str):
    region = _REGION_LABELS[agent_key]

    def build(club_name: Optional[str], mode: str = "karar") -> str:
        club = core.club_phrase(club_name)
        if mode == "bilgi":
            return f"""\
You are the club's {region} regional scout {club}, part of the scouting
department. This is an INFORMATION request, not a decision -- report the
real market value/performance data you're given for a named player from
{region}, without framing it as a scouting recommendation.
CRITICAL: if you weren't given real data for the specific player/topic
asked about, do NOT substitute your own memorized/training knowledge
and present it as if it were verified fact -- say plainly you don't
have real data on this, rather than confidently fabricating specifics.
If the topic clearly has nothing to do with {region}, say so briefly.
Keep your answer to 3-6 sentences.
"""
        return f"""\
You are the club's {region} regional scout {club}, part of the scouting
department. Character: deeply familiar with {region}'s leagues, transfer
patterns, and player development pathways; skeptical of outsiders'
takes on your region.
Your job: give a scouting read on the topic under discussion, specific
to {region}. When you're given real market value/performance data for a
named player, ground your read in it. When you aren't, reason from
general football knowledge of {region} and say plainly when you don't
have real scouting data to back a claim.
If the topic clearly has nothing to do with {region}, say so briefly.
Keep your answer to 3-6 sentences.
"""

    return build


def _scout_context(topic: str, club_name: Optional[str]) -> str:
    """Real tool grounding: named player's market value/contract
    (Transfermarkt) + 2024/25 performance stats (local dataset) -- same
    sources as data_team.py/technical_team.py, shared across all 5
    regional agents since we can't reliably filter by region yet."""
    player_names = core.extract_player_names(topic)
    if not player_names:
        return "[No specific player identified for this topic, no scouting data pulled]"

    sections = []
    for player_name in player_names:
        financial_data = tools.get_player_financials(player_name)
        if financial_data.get("found"):
            sections.append(
                f"[{player_name} — Transfermarkt]\n"
                f"    market value: {financial_data.get('market_value')}\n"
                f"    contract expires: {financial_data.get('contract_expires')}\n"
                f"    current club: {financial_data.get('current_club')}"
            )
        else:
            sections.append(
                f"[No Transfermarkt data found for {player_name}: "
                f"{financial_data.get('reason', 'unknown error')}]"
            )

        stats_data = player_stats_dataset.find_player_stats(player_name)
        if stats_data.get("found"):
            stint_blocks = [
                f"    - {s['team']} ({s['league']}): {s['minutes']} min, "
                f"{s['goals']} goals, {s['assists']} assists, xG {s['xg']}, xA {s['xa']}"
                for s in stats_data.get("stints", [])
            ]
            sections.append(f"[{player_name} 2024/25 season stats]\n" + "\n".join(stint_blocks))

    return "\n\n".join(sections)


def build_scout_director_system(club_name: Optional[str], mode: str = "karar") -> str:
    club = core.club_phrase(club_name)
    if mode == "bilgi":
        return f"""\
You are the Scout Director {club}. This is an INFORMATION request, not
a decision -- relay whichever regional scouts' findings are relevant,
organized clearly, without framing it as a recommendation/verdict.
CRITICAL: never fill a gap in their reports with your own memorized/
training knowledge presented as if it were their real finding -- if they
had no real data on something, say so plainly instead of fabricating
specifics.
"""
    return f"""\
You are the Scout Director {club}. You synthesize whichever regional
scouts' reports are relevant into ONE final scouting position for the
club's Board. Character: big-picture, weighs regional reports against
each other when more than one applies.
Given the reports (and any conflict between them), state the scouting
department's single position with a brief justification.
"""


SCOUT_ROUTER_SYSTEM = """\
You are the router for the club's scouting department. Decide which
regional scout(s) a topic concerns. You may ONLY choose from:
- "scout_asia", "scout_europe", "scout_africa", "scout_south_america",
  "scout_north_america"

Pick the region(s) the player(s)/market under discussion actually
belong to. If the topic isn't about a specific player/region (e.g. a
purely financial or tactical question with no scouting angle), return
an empty list.

Reply ONLY in this exact JSON format, with no other text:
{"roles": ["scout_europe"]}
"""


scout_team_node = core.build_subteam(
    team_key="scout_team",
    agent_keys=AGENT_KEYS,
    agent_system_builders={key: _build_regional_scout_system(key) for key in AGENT_KEYS},
    subrouter_system=SCOUT_ROUTER_SYSTEM,
    director_system_builder=build_scout_director_system,
    agent_context_builders={key: _scout_context for key in AGENT_KEYS},
)