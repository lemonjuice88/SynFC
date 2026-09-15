"""
legal_team.py — SynFC's Legal department (sub-graph)
===========================================================
Built with core.build_subteam(). A single Legal Agent (covering both
procedural/disciplinary AND FFP/financial-regulatory matters -- the
user deliberately merged what were originally two separate agents into
one) reports to a Legal Director.

Real tool grounding:
- legal_documents.py: searches actual FIFA regulation text (RSTP PDF)
  so answers are grounded in real rule text, not general knowledge.
- mail_drafter.py: drafts (never sends) formal correspondence to
  governing bodies like FIFA/UEFA -- see that file's docstring for why
  sending is deliberately out of scope. The draft text is spliced into
  the final output IN CODE (core.py's agent_output_prefixers), not by
  asking the LLM to reproduce it -- LLMs are unreliable at verbatim
  transcription of long text even when explicitly instructed to do so
  ("copy this exactly" kept getting summarized/paraphrased away in
  testing). The LLM's only job re: the letter is to comment on it, not
  to reproduce it.
"""

from functools import lru_cache
from typing import Optional

import core
import legal_documents
import mail_drafter

AGENT_KEYS = ["legal_agent"]


def build_legal_agent_system(club_name: Optional[str]) -> str:
    club = core.club_phrase(club_name)
    return f"""\
You are the club's legal agent {club}, covering BOTH procedural/
disciplinary matters (registration deadlines, disciplinary proceedings,
governing-body correspondence) AND financial-regulatory matters (FFP,
wage caps, transfer-related compliance). Character: procedural,
precise, risk-aware, thinks in terms of rules, deadlines, and
regulatory exposure.
Your job: assess the legal/regulatory aspects of the topic under
discussion. When you're given actual regulation text (search results)
or a draft letter, ground your answer in it and reference it directly.
When you aren't given relevant regulation text, reason from general
football-regulatory knowledge and say plainly when you're missing a
concrete rule reference -- don't invent specific article numbers.

If a draft letter is included in your context, it will ALREADY be shown
to the user separately in full -- you do NOT need to reproduce it
yourself. Just give your own short risk assessment / recommendation on
it (3-6 sentences), the way you would for any other topic.

Keep your answer to 3-6 sentences.
"""


def build_legal_director_system(club_name: Optional[str]) -> str:
    club = core.club_phrase(club_name)
    return f"""\
You are the Legal Director {club}. You give the Board a short,
board-level take on the legal agent's finding. Character: risk-averse,
precise, prioritizes keeping the club compliant over speed.
Keep your answer to 3-6 sentences -- if a draft letter was discussed,
you do NOT need to reproduce it (it's shown to the user separately),
just give your own recommendation on it.
"""


LEGAL_ROUTER_SYSTEM = """\
You are the router for the club's legal department. Decide whether this
topic concerns the legal agent at all. You may ONLY choose from:
- "legal_agent": any procedural, disciplinary, registration, governing-
  body correspondence, FFP, wage-cap, or financial-regulatory matter

If the topic clearly has no legal/regulatory angle, return an empty list.

Reply ONLY in this exact JSON format, with no other text:
{"roles": ["legal_agent"]}
"""


def _wants_mail_draft(topic: str) -> bool:
    """Simple keyword trigger for "write a mail" intent -- deliberately
    simple (not LLM-based) for now so it's fast, free, and predictable.
    Revisit with an LLM-based intent check if this proves too narrow."""
    topic_lower = topic.lower()
    trigger_phrases = [
        "mail yaz", "mail gönder", "mektup yaz", "e-posta yaz", "eposta yaz",
        "write a letter", "write an email", "draft a letter", "draft an email",
    ]
    return any(phrase in topic_lower for phrase in trigger_phrases)


def _detect_mail_recipient(topic: str) -> str:
    topic_lower = topic.lower()
    if "uefa" in topic_lower:
        return "UEFA"
    if "fifa" in topic_lower:
        return "FIFA"
    return "the relevant governing body"


@lru_cache(maxsize=32)
def _cached_draft(topic: str, recipient: str, club_name: str) -> str:
    """Shared cache so _legal_agent_context (informational, for the LLM)
    and _legal_agent_output_prefix (verbatim, spliced in code) don't
    each trigger their own separate LLM call for the identical letter."""
    return mail_drafter.draft_formal_letter(topic, recipient, club_name)


def _legal_agent_context(topic: str, club_name: Optional[str]) -> str:
    """Real tool grounding: (1) if the topic expresses a "write a mail"
    intent, includes the draft letter as CONTEXT for the LLM to comment
    on (the verbatim copy the user sees is handled separately, in code
    -- see _legal_agent_output_prefix below); (2) OTHERWISE searches the
    actual FIFA regulation text for terms mentioned in the topic --
    SEMANTIC search first (catches differently-worded-but-same-meaning
    questions), falling back to keyword search if semantic search finds
    nothing (e.g. the embedding model isn't installed/available, or the
    query is too vague to embed meaningfully). Both stay available
    deliberately -- keyword search is still useful as a safety net.
    """
    if _wants_mail_draft(topic):
        recipient = _detect_mail_recipient(topic)
        draft = _cached_draft(topic, recipient, club_name or "the club")
        return (
            f"[DRAFT LETTER to {recipient} — this will be shown to the user "
            f"separately in full, you just need to comment on it]\n{draft}"
        )

    try:
        semantic_result = legal_documents.semantic_search_regulation(topic)
    except Exception:
        semantic_result = {"found": False}

    if semantic_result.get("found"):
        lines = []
        for match in semantic_result["results"]:
            lines.append(
                f"    [Page {match['page']}, similarity {match['similarity']}]\n{match['text'][:600]}"
            )
        return f"[FIFA RSTP — semantic search results]\n" + "\n\n".join(lines)

    # Fallback: keyword search
    query_terms = [w.strip(".,?!") for w in topic.split() if len(w) > 3]
    reg_result = legal_documents.search_regulation(query_terms)
    if not reg_result.get("found"):
        return f"[No matching regulation text found: {reg_result.get('reason', 'unknown')}]"

    lines = []
    for match in reg_result["sample"]:
        lines.append(f"    [Page {match['page']}]\n{match['text'][:600]}")
    return (
        f"[FIFA RSTP — keyword search, {reg_result['total_matches']} matching page(s), "
        f"showing {len(reg_result['sample'])}]\n" + "\n\n".join(lines)
    )


def _legal_agent_output_prefix(topic: str, club_name: Optional[str]) -> str:
    """Splices the EXACT draft letter text onto the front of the agent's
    final answer, in code -- guarantees the user sees the real letter
    regardless of what the LLM does with it in its own commentary."""
    if not _wants_mail_draft(topic):
        return ""
    recipient = _detect_mail_recipient(topic)
    draft = _cached_draft(topic, recipient, club_name or "the club")
    return f"**[TASLAK MEKTUP — Alıcı: {recipient} — HENÜZ GÖNDERİLMEDİ]**\n\n{draft}"


legal_team_node = core.build_subteam(
    team_key="legal_team",
    agent_keys=AGENT_KEYS,
    agent_system_builders={
        "legal_agent": build_legal_agent_system,
    },
    subrouter_system=LEGAL_ROUTER_SYSTEM,
    director_system_builder=build_legal_director_system,
    agent_context_builders={
        "legal_agent": _legal_agent_context,
    },
    agent_output_prefixers={
        "legal_agent": _legal_agent_output_prefix,
    },
    director_output_prefixer=_legal_agent_output_prefix,
)