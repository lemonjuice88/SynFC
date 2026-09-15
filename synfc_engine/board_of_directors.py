"""
board_of_directors.py — SynFC's final decision stage
==========================================================
Replaces the old single-persona president_node. Deliberately does NOT
use core.build_subteam() -- its purpose isn't to synthesize N agents'
reports into one position (every other team already does that), it's
to DEBIAS the president's final call by forcing an explicit two-sided
argument first: a Good Cop states the positive case across everything
the teams reported, a Bad Cop states the risk/negative case, and only
then does the President render the final decision -- having to
genuinely engage with both sides instead of drifting toward whichever
reads more persuasive on its own.

These three nodes are wired directly into the MAIN engine's graph
(engine.py), operating on the main SynFCState (not a private sub-state
like the other teams) since their whole job is to look at what's
already in role_opinions.
"""

import core

GOOD_COP_SYSTEM = f"""\
You are the "Good Cop" on the club's Board of Directors. You'll be
given the opinions of whichever teams were consulted on a topic. Your
ONLY job: list the strongest POSITIVE angles across everything you were
given -- opportunities, upside, reasons this could go well. Do NOT
mention risks or downsides; that's not your role here, someone else
covers that. Be genuinely persuasive, not half-hearted.
Keep your answer to 3-6 sentences.
"""

BAD_COP_SYSTEM = f"""\
You are the "Bad Cop" on the club's Board of Directors. You'll be given
the opinions of whichever teams were consulted on a topic. Your ONLY
job: list the strongest NEGATIVE/RISK angles across everything you were
given -- what could go wrong, what's being underweighted, reasons for
caution. Do NOT mention positives or upside; that's not your role here,
someone else covers that. Be genuinely persuasive, not half-hearted.
Keep your answer to 3-6 sentences.
"""


def build_president_system(club_name) -> str:
    club = core.club_phrase(club_name)
    return f"""\
You are the president {club}. You make the final call. You've been
given: (1) the consulted teams' opinions, (2) the Good Cop's positive
case, and (3) the Bad Cop's risk case. Character: balanced, politically
savvy, weighing the club's long-term image alongside short-term
results.
You must genuinely engage with BOTH the Good Cop's and Bad Cop's
arguments -- don't just default to whichever team talked the most --
and announce a final, actionable decision with a brief justification.
If no team was consulted (the topic was addressed directly to you, and
there's nothing for Good Cop/Bad Cop to react to), decide on your own.
"""


def good_cop_node(state: core.SynFCState) -> dict:
    opinions = state.get("role_opinions", {})
    joined = "\n".join(f"{r}: {o}" for r, o in opinions.items()) if opinions else "(no teams were consulted)"
    user_prompt = f"Topic: {state['topic']}\n\nTeams' opinions:\n{joined}\n\nList the positive case."
    text = core.call_llm(GOOD_COP_SYSTEM, user_prompt)
    return {"board_notes": {"good_cop": text}}


def bad_cop_node(state: core.SynFCState) -> dict:
    opinions = state.get("role_opinions", {})
    joined = "\n".join(f"{r}: {o}" for r, o in opinions.items()) if opinions else "(no teams were consulted)"
    user_prompt = f"Topic: {state['topic']}\n\nTeams' opinions:\n{joined}\n\nList the risk/negative case."
    text = core.call_llm(BAD_COP_SYSTEM, user_prompt)
    return {"board_notes": {"bad_cop": text}}


def president_node(state: core.SynFCState) -> dict:
    opinions = state.get("role_opinions", {})
    board_notes = state.get("board_notes", {})

    joined = "\n".join(f"{r}: {o}" for r, o in opinions.items()) if opinions else "(no teams were consulted)"
    good_cop_text = board_notes.get("good_cop", "(not available -- no teams were consulted)")
    bad_cop_text = board_notes.get("bad_cop", "(not available -- no teams were consulted)")

    user_prompt = (
        f"Topic: {state['topic']}\n\n"
        f"Teams' opinions:\n{joined}\n\n"
        f"Good Cop's positive case:\n{good_cop_text}\n\n"
        f"Bad Cop's risk case:\n{bad_cop_text}\n\n"
        f"Announce your final decision along with a brief justification."
    )
    decision = core.call_llm(build_president_system(state.get("club_name")), user_prompt)
    return {"final_decision": decision}