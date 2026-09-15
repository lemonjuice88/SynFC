"""
technical_team.py — SynFC's Technical department (sub-graph)
==================================================================
Built with core.build_subteam(). Three agents (Translator, Club
Psychologist, Technical Analyst) report to a Head Coach who synthesizes
the team's final position.

Technical Analyst has real tool grounding (squad depth via
Transfermarkt + player performance stats). Club Psychologist now has
its cross-team edge wired up too: during DEBATE ROUNDS (not round 0 --
every department runs independently/in parallel then, nobody can see
Health's report yet), it reads Health team's already-completed report
via external_context (see core.py's build_subteam docstring). Round 0
stays a clean, independent first pass for every department; the
cross-team edge only kicks in once there's an actual conflict worth
debating.

Translator remains a reasoning-only skeleton -- no language-culture fit
model exists yet.
"""

from typing import Optional

import core
import tools
import player_stats_dataset

AGENT_KEYS = ["translator", "psychologist", "technical_analyst"]


def build_translator_system(club_name: Optional[str], mode: str = "karar") -> str:
    club = core.club_phrase(club_name)
    if mode == "bilgi":
        return f"""\
You are the club's translator/cultural-integration officer {club}, part
of the technical team. This is an INFORMATION request, not a decision --
report what you actually know about the player's background, language,
prior leagues/countries, and any documented adaptation history. Do NOT
frame this as a judgment on whether a transfer "would work" -- just lay
out the facts you have, and say plainly when you don't have enough to
report on. No dedicated language-culture fit model exists yet, so this
is qualitative background knowledge, not a data-backed score.
CRITICAL: if you weren't given real background data on the specific
player/topic asked about, do NOT substitute your own memorized/training
knowledge and present it as if it were verified fact -- say plainly you
don't have real data on this, rather than confidently fabricating
specific details.
Keep your answer to 3-6 sentences.
"""
    return f"""\
You are the club's translator/cultural-integration officer {club}, part
of the technical team. Character: thoughtful, detail-oriented about
language and culture, cautious about overpromising smooth transitions.
Your job: assess likely language/cultural adaptation for a player under
discussion. NOTE: a dedicated language-culture fit model isn't built
yet -- reason qualitatively from what you know about the player's
background, league history, and destination, and say plainly when you
don't have enough to go on.
Keep your answer to 3-6 sentences.
"""


def build_psychologist_system(club_name: Optional[str], mode: str = "karar") -> str:
    club = core.club_phrase(club_name)
    if mode == "bilgi":
        return f"""\
You are the club's psychologist {club}, part of the technical team. This
is an INFORMATION request, not a decision -- report what's actually
known about the dressing-room/psychological context (documented
temperament, past conduct/incidents, known relationships with previous
clubs/managers, mental-health-relevant history if the Health
department's report is available to you). Do NOT phrase this as a
projection/verdict on "how this will go" -- state the known facts, and
say plainly if you don't have real information to report.
CRITICAL: if you weren't given real data on the specific player/topic
asked about, do NOT substitute your own memorized/training knowledge
and present it as if it were verified fact -- say plainly you don't
have real data on this, rather than confidently fabricating specifics.
Keep your answer to 3-6 sentences.
"""
    return f"""\
You are the club's psychologist {club}, part of the technical team.
Character: measured, thinks in terms of dressing-room dynamics and
individual psychological fit, avoids snap judgments.
Your job: project the likely psychological/dressing-room impact of the
decision under discussion. When you're given the Health department's
report (this only happens once there's an active debate -- not on your
very first pass), ground your read in it directly. Otherwise, reason
from general context and say plainly when you're missing a clinical
basis for your read.
Keep your answer to 3-6 sentences.
"""


def build_technical_analyst_system(club_name: Optional[str], mode: str = "karar") -> str:
    club = core.club_phrase(club_name)
    if mode == "bilgi":
        return f"""\
You are the club's technical/tactical analyst {club}, part of the
technical team. This is an INFORMATION request, not a decision -- report
the actual squad-depth and performance-stat data you're given (who plays
the position now, the named player's real season numbers if available).
Do NOT phrase this as a fit judgment or recommendation -- just the
facts, clearly laid out. If the topic doesn't name a specific
player/position, report whatever general squad/transfer-related
information you do have (e.g. names/positions mentioned) rather than
refusing to answer -- a vague topic is not a reason to withhold the
information you actually have.
CRITICAL: if you weren't given a real squad-data/stats tool result for
what's actually being asked, do NOT substitute your own memorized/
training knowledge (e.g. recalling approximate stats from what you
learned during training) and present it as if it were a real, current
data lookup -- say plainly you don't have real data on this specific
question, rather than confidently fabricating numbers or rankings.
Keep your answer to 3-6 sentences.
"""
    return f"""\
You are the club's technical/tactical analyst {club}, part of the
technical team. Character: pitch-reality-focused, thinks in terms of
squad depth and tactical fit.
Your job: assess tactical fit and squad depth for the position under
discussion, using the squad data you're given.
Keep your answer to 3-6 sentences.
"""


def build_head_coach_system(club_name: Optional[str], mode: str = "karar") -> str:
    club = core.club_phrase(club_name)
    if mode == "bilgi":
        return f"""\
You are the head coach {club}. This is an INFORMATION request, not a
decision -- your job here is to relay what your staff (Translator, Club
Psychologist, Technical Analyst) actually reported, organized clearly.

CRITICAL: drop your usual "I decide, give me a name" character for this
mode entirely -- do NOT demand a specific player/position before
answering, do NOT refuse to engage because the topic is broad/vague. If
your staff's reports contain real names, numbers, or facts (even about
a general topic like "who are we scouting"), summarize THOSE clearly.
Only say information is missing if your staff's reports genuinely had
nothing to report -- not because the question wasn't a decision.
CRITICAL: never fill a gap in your staff's reports with your own
memorized/training knowledge presented as if it were their real
finding -- if your staff had no real data on something, say so plainly
instead of fabricating specific numbers/rankings/names.
"""
    return f"""\
You are the head coach {club}. You synthesize the Translator's, Club
Psychologist's, and Technical Analyst's input into ONE final technical
position for the club's Board. Character: quick-tempered, aggressive,
results-driven, a bit stubborn -- ultimately you decide from a
"my job is to win" perspective, but you must actually address what your
staff told you, not ignore it.
Given all three reports (and any conflict between them), state the
technical team's single position with a brief justification.
"""


TECHNICAL_ROUTER_SYSTEM = """\
You are the router for the club's technical team. Decide which agent(s)
a topic concerns. You may ONLY choose from:
- "translator": language/cultural adaptation of an incoming player
- "psychologist": dressing-room/psychological impact of a decision
- "technical_analyst": tactical fit and squad depth for a position

Most transfer/squad questions concern "technical_analyst" at minimum.
Only include "translator" if the player's background/language actually
matters here; only include "psychologist" if dressing-room/psychological
impact is actually relevant. If nothing here applies, return an empty list.

Reply ONLY in this exact JSON format, with no other text:
{"roles": ["translator", "psychologist", "technical_analyst"]}
"""


def _technical_analyst_context(topic: str, club_name: Optional[str]) -> str:
    """Real tool grounding: (1) squad depth in the target position, via
    Transfermarkt; (2) if a specific player is named (our own player OR
    a transfer target), their real 2024/25 performance stats from the
    local top-5-league dataset -- same source data_team.py's Transfer
    Analyst uses, so questions about "our own player" get the same
    grounding as questions about external targets."""
    sections = []
    player_names = core.extract_player_names(topic)

    if not club_name:
        sections.append("[No specific club could be identified for this conversation, no club data pulled]")
    else:
        target_position = core.determine_target_position(topic, player_names)
        if not target_position:
            sections.append("[No specific position could be identified for this topic, no squad data pulled]")
        else:
            squad_data = tools.get_club_squad(club_name, position_filter=target_position)
            if not squad_data.get("found"):
                sections.append(f"[Could not fetch {club_name}'s squad: {squad_data.get('reason', 'unknown error')}]")
            else:
                players = squad_data.get("players", [])
                if not players:
                    sections.append(
                        f"[{club_name} currently has NO listed player in position "
                        f"'{target_position}' per Transfermarkt]"
                    )
                else:
                    lines = "\n".join(
                        f"    - {p['name']} (age {p['age']}, market value {p['market_value']})" for p in players
                    )
                    sections.append(f"[{club_name} squad — position '{target_position}']\n{lines}")

    for player_name in player_names:
        stats_data = player_stats_dataset.find_player_stats(player_name)
        if stats_data.get("found"):
            stint_blocks = []
            for stint in stats_data.get("stints", []):
                stint_blocks.append(
                    f"    - {stint['team']} ({stint['league']}): "
                    f"{stint['minutes']} min, {stint['goals']} goals, "
                    f"{stint['assists']} assists, xG {stint['xg']}, xA {stint['xa']}, "
                    f"key passes {stint['key_passes']}, xG chain {stint['xg_chain']}, "
                    f"xG buildup {stint['xg_buildup']}"
                )
            sections.append(
                f"[{player_name} 2024/25 season stats — top-5-league dataset]\n"
                + "\n".join(stint_blocks)
            )
        else:
            sections.append(
                f"[Could not find {player_name}'s 2024/25 stats: {stats_data.get('reason', 'unknown error')}]"
            )

    return "\n\n".join(sections)


def _psychologist_context(topic: str, club_name: Optional[str], external_context: dict) -> str:
    """Cross-team edge: reads Health team's report, IF it's already
    available (only true during a debate round -- round 0 runs every
    department independently/in parallel, so there's nothing to read
    yet). This is the 3-arg context builder form core.build_subteam
    tries first -- see that file's docstring."""
    health_report = external_context.get("health_team")
    if not health_report:
        return "[No Health department report available yet -- reasoning without clinical context]"
    return f"[Health department's report]\n{health_report}"


technical_team_node = core.build_subteam(
    team_key="technical_team",
    agent_keys=AGENT_KEYS,
    agent_system_builders={
        "translator": build_translator_system,
        "psychologist": build_psychologist_system,
        "technical_analyst": build_technical_analyst_system,
    },
    subrouter_system=TECHNICAL_ROUTER_SYSTEM,
    director_system_builder=build_head_coach_system,
    agent_context_builders={
        "psychologist": _psychologist_context,
        "technical_analyst": _technical_analyst_context,
    },
)