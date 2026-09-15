"""
data_team.py — SynFC's Data department (sub-graph)
========================================================
Built with core.build_subteam(). Three agents (Data Scientist, Transfer
Analyst, Match Analyst) report to a Data Director.

Data Scientist now has real tool grounding too: text-to-SQL against a
real SQLite database (sql_database.py) built from the local top-5-league
CSV -- an LLM writes the actual SQL query for whatever the topic is
asking, and it's really executed (SELECT-only, see that file's safety
notes). Transfer Analyst inherits the full tool chain the old top-level
analytics_team.py had built up (squad depth, injury history, local
top-5-league stats dataset). Match Analyst (no match-data tool yet) is
still a reasoning-only skeleton.
"""

from typing import Optional

import core
import tools
import player_stats_dataset
import sql_database

AGENT_KEYS = ["data_scientist", "transfer_analyst", "match_analyst"]


def build_data_scientist_system(club_name: Optional[str], mode: str = "karar") -> str:
    club = core.club_phrase(club_name)
    if mode == "bilgi":
        return f"""\
You are the club's data scientist {club}, part of the data team. This is
an INFORMATION request, not a decision -- report the real SQL query
results you're given exactly as they are, without framing it as a
judgment call. The database only covers the 2024/25 season across the
top 5 European leagues, so say plainly when a question falls outside
that scope (older seasons, lower leagues) rather than guessing.
CRITICAL: if the SQL query returned no real results (or wasn't run),
do NOT substitute your own memorized/training knowledge (e.g. recalling
approximate stats from what you learned during training) and present it
as if it were a real, current database lookup -- say plainly you don't
have real data on this specific question, rather than confidently
fabricating numbers or rankings.
Keep your answer to 3-6 sentences.
"""
    return f"""\
You are the club's data scientist {club}, part of the data team.
Character: deeply technical, fluent in advanced metrics (xG, xA, xG
chain/buildup, and beyond), impatient with non-data-driven reasoning.
Your job: give a technically rigorous statistical read on the topic
under discussion. When you're given real SQL query results, ground your
answer in those exact numbers -- cite them directly. The database only
covers the 2024/25 season across the top 5 European leagues, so say
plainly when a question falls outside that scope (older seasons, lower
leagues) rather than guessing.
Keep your answer to 3-6 sentences.
"""


def build_transfer_analyst_system(club_name: Optional[str], mode: str = "karar") -> str:
    club = core.club_phrase(club_name)
    if mode == "bilgi":
        return f"""\
You are the club's transfer analyst {club}, part of the data team. This
is an INFORMATION request, not a decision -- report the real squad
depth/injury/performance data you're given, without framing it as a
transfer judgment.
CRITICAL: if you weren't given real data for what's actually being
asked, do NOT substitute your own memorized/training knowledge and
present it as if it were verified fact -- say plainly you don't have
real data on this, rather than confidently fabricating specifics.
Keep your answer to 3-6 sentences.
"""
    return f"""\
You are the club's transfer analyst {club}, part of the data team.
Character: cool-headed, neutral, translates raw statistics into
transfer-relevant conclusions.
Your job: give an objective read on a potential transfer target using
squad depth, injury history, and season performance data.
Keep your answer to 3-6 sentences.
"""


def build_match_analyst_system(club_name: Optional[str], mode: str = "karar") -> str:
    club = core.club_phrase(club_name)
    if mode == "bilgi":
        return f"""\
You are the club's match analyst {club}, part of the data team. This is
an INFORMATION request, not a decision -- report what you actually know
about the match/opponent context, without framing it as a tactical
judgment.
CRITICAL: no dedicated match/opponent-data tool exists yet -- do NOT
substitute your own memorized/training knowledge for real data and
present it as verified fact. Say plainly you don't have real data on
this, rather than confidently fabricating specifics.
Keep your answer to 3-6 sentences.
"""
    return f"""\
You are the club's match analyst {club}, part of the data team.
Character: tactically minded, focused on in-match/opponent-specific
context rather than season-long trends.
Your job: give a match-context read on the topic under discussion (e.g.
how a player/tactic would perform against likely upcoming opposition).
NOTE: a dedicated match/opponent-data tool isn't built yet -- reason
qualitatively and say plainly when you're missing real match data.
Keep your answer to 3-6 sentences.
"""


def build_data_director_system(club_name: Optional[str], mode: str = "karar") -> str:
    club = core.club_phrase(club_name)
    if mode == "bilgi":
        return f"""\
You are the Data Director {club}. This is an INFORMATION request, not a
decision -- relay whichever of the Data Scientist's, Transfer Analyst's,
and Match Analyst's findings are relevant, organized clearly, without
framing it as a recommendation/verdict.
CRITICAL: never fill a gap in their reports with your own memorized/
training knowledge presented as if it were their real finding -- if they
had no real data on something, say so plainly instead of fabricating
specific numbers/rankings.
"""
    return f"""\
You are the Data Director {club}. You synthesize whichever of the Data
Scientist's, Transfer Analyst's, and Match Analyst's reports are
relevant into ONE final data-driven position for the club's Board.
Character: rigorous, insists conclusions follow from the evidence
actually presented, not from vibes.
Given the reports (and any conflict between them), state the data
team's single position with a brief justification.
"""


DATA_ROUTER_SYSTEM = """\
You are the router for the club's data team. Decide which agent(s) a
topic concerns. You may ONLY choose from:
- "data_scientist": deep statistical/advanced-metric questions
- "transfer_analyst": transfer-target evaluation (squad fit, injury
  history, season stats)
- "match_analyst": match/opponent-specific tactical context

Most transfer questions concern "transfer_analyst" at minimum. Only
include the others if the topic genuinely calls for that specific lens.
If nothing here applies, return an empty list.

Reply ONLY in this exact JSON format, with no other text:
{"roles": ["data_scientist", "transfer_analyst", "match_analyst"]}
"""


def _transfer_analyst_context(topic: str, club_name: Optional[str]) -> str:
    """Real tool grounding: squad depth + local top-5-league stats
    dataset. Injury history was REMOVED from here -- that's now
    exclusively Health department's territory (see health_team.py's
    Physio, which enriches it with research-grounded context via
    injury_profiles.py). Data team focuses on squad/performance data;
    Technical team's psychologist can pull Health's synthesized report
    during a debate round via the cross-team edge -- no need to
    duplicate raw injury data fetching here too."""
    sections = []

    # 1) Squad depth in the target position (needs a club)
    if not club_name:
        sections.append("[No specific club could be identified for this conversation, no squad data pulled]")
    else:
        player_names = core.extract_player_names(topic)
        target_position = core.determine_target_position(topic, player_names)
        if target_position:
            squad_data = tools.get_club_squad(club_name, position_filter=target_position)
            if squad_data.get("found"):
                players = squad_data.get("players", [])
                if players:
                    lines = "\n".join(
                        f"    - {p['name']} (age {p['age']}, market value {p['market_value']})"
                        for p in players
                    )
                    sections.append(f"[{club_name} squad — position '{target_position}']\n{lines}")
                else:
                    sections.append(
                        f"[{club_name} currently has NO listed player in position "
                        f"'{target_position}' per Transfermarkt]"
                    )
            else:
                sections.append(
                    f"[Could not fetch {club_name}'s squad: {squad_data.get('reason', 'unknown error')}]"
                )
        else:
            sections.append("[No specific position could be identified for this topic, no squad data pulled]")

    # 2) The target player's real season performance stats
    player_names = core.extract_player_names(topic)
    if not player_names:
        sections.append("[No specific player found in the message, no stats data pulled]")
    else:
        for player_name in player_names:
            stats_data = player_stats_dataset.find_player_stats(player_name)
            if stats_data.get("found"):
                stint_blocks = []
                for stint in stats_data.get("stints", []):
                    stint_blocks.append(
                        f"    - {stint['team']} ({stint['league']}): "
                        f"{stint['minutes']} min, {stint['goals']} goals, "
                        f"{stint['assists']} assists, xG {stint['xg']}, "
                        f"xA {stint['xa']}, key passes {stint['key_passes']}, "
                        f"xG chain {stint['xg_chain']}, xG buildup {stint['xg_buildup']} "
                        f"(chain-minus-buildup = direct involvement in the final action, "
                        f"not just being part of the move)"
                    )
                sections.append(
                    f"[{player_name} 2024/25 season stats — top-5-league dataset]\n"
                    + "\n".join(stint_blocks)
                )
            else:
                sections.append(
                    f"[Could not find {player_name}'s 2024/25 stats: "
                    f"{stats_data.get('reason', 'unknown error')}]"
                )

    return "\n\n".join(sections)


def _data_scientist_context(topic: str, club_name: Optional[str]) -> str:
    """Real tool grounding: text-to-SQL against the real SQLite database
    (sql_database.py) -- an LLM writes the actual SQL query for `topic`,
    and it's really executed against real 2024/25 top-5-league data."""
    try:
        result = sql_database.ask_database(topic)
    except Exception as e:
        return f"[SQL query failed: {e}]"

    if not result.get("found"):
        return f"[SQL query unsuccessful: {result.get('reason', 'unknown error')}]\n(Generated SQL: {result.get('sql', 'n/a')})"

    if not result.get("rows"):
        return f"[Query ran successfully but returned no rows]\n(SQL: {result['sql']})"

    lines = "\n".join(f"    - {row}" for row in result["rows"])
    return f"[SQL query result]\n(SQL: {result['sql']})\n{lines}"


data_team_node = core.build_subteam(
    team_key="data_team",
    agent_keys=AGENT_KEYS,
    agent_system_builders={
        "data_scientist": build_data_scientist_system,
        "transfer_analyst": build_transfer_analyst_system,
        "match_analyst": build_match_analyst_system,
    },
    subrouter_system=DATA_ROUTER_SYSTEM,
    director_system_builder=build_data_director_system,
    agent_context_builders={
        "data_scientist": _data_scientist_context,
        "transfer_analyst": _transfer_analyst_context,
    },
)