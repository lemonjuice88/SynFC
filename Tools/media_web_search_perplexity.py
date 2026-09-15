"""
media_web_search_perplexity.py — live media/fan sentiment via Perplexity
=============================================================================
ONE unified search tool -- searches broadly (news, press, forums,
reddit.com, x.com) and returns a single synthesized report. This same
report is fed to BOTH the Media team's `fans` and `reporter` agents
(see media_team.py) -- since each already has its own distinct persona
(via system prompt), they naturally emphasize the parts of the report
relevant to their own role rather than us pre-splitting the search
itself into narrower, separately-scoped queries.

Cost: Perplexity bakes web search into the per-token price (no separate
$-per-search fee). Its cheapest tier (Sonar Small Online, ~$0.20 per 1M
tokens as of testing) means a typical short query costs a small
fraction of a cent.

Setup:
    pip install openai   # already a project dependency
    Get an API key at https://www.perplexity.ai/settings/api
    Add to .env: PERPLEXITY_API_KEY=pplx-...
"""

import os

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))

from openai import OpenAI

PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "")
PERPLEXITY_BASE_URL = "https://api.perplexity.ai"
PERPLEXITY_MODEL = os.environ.get("PERPLEXITY_MODEL", "sonar")

_client = None


def _get_client():
    global _client
    if _client is None:
        if not PERPLEXITY_API_KEY:
            raise RuntimeError(
                "PERPLEXITY_API_KEY not set. See media_web_search_perplexity.py's "
                "module docstring for setup steps."
            )
        _client = OpenAI(api_key=PERPLEXITY_API_KEY, base_url=PERPLEXITY_BASE_URL)
    return _client


SEARCH_SYSTEM_PROMPT = """\
You are a media research assistant for a football club's Media team.
Given a topic, search the web for how fans and the press are currently
reacting to it. Prioritize checking reddit.com and x.com specifically
if relevant results exist there, alongside football news sites/forums.

Summarize honestly: the general sentiment (positive/negative/mixed/
genuinely unclear), any recurring themes you actually found -- both
supporter reaction AND press/journalist framing, since this report is
read by both the fans and reporter perspectives on the Media team --
and say plainly if you couldn't find much real discussion rather than
inventing a confident-sounding summary. Keep it to 4-8 sentences.
Don't fabricate quotes.
"""


def search_fan_sentiment(topic: str) -> dict:
    """Searches the web for current fan/press sentiment on `topic` via
    Perplexity's Sonar API and returns ONE synthesized report, meant to
    be shared as tool input for both the `fans` and `reporter` agents.

    Example return:
        {"found": True, "topic": "...", "summary": "...", "citations": [...]}
    """
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=PERPLEXITY_MODEL,
            messages=[
                {"role": "system", "content": SEARCH_SYSTEM_PROMPT},
                {"role": "user", "content": f"Topic: {topic}"},
            ],
        )
        summary = response.choices[0].message.content
        citations = getattr(response, "citations", None) or []
        return {"found": True, "topic": topic, "summary": summary, "citations": citations}
    except Exception as e:
        return {"found": False, "reason": f"Error during Perplexity search: {e}"}


GENERAL_INFO_SYSTEM_PROMPT = """\
You answer a factual question by searching the web -- NOT media/fan
sentiment specifically, just a general, neutral, accurate answer to
whatever is asked. This is used as a fallback when a football club's
internal departments weren't confident enough in their own knowledge to
answer directly.

Answer directly and factually based on what you find. If you can't find
a clear answer, say so plainly rather than guessing. Keep it to 4-8
sentences. Don't fabricate facts or quotes.
"""


def answer_general_question(question: str) -> dict:
    """General-purpose factual web search -- used by engine.py's
    confidence_check_node as a fallback when the department reports for
    an INFORMATION-mode query (not a decision) read as uncertain/hedged
    rather than confidently informative. Deliberately separate from
    search_fan_sentiment() above: no fan/press framing here, just a
    neutral factual-answer prompt.

    Example return:
        {"found": True, "question": "...", "summary": "...", "citations": [...]}
    """
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=PERPLEXITY_MODEL,
            messages=[
                {"role": "system", "content": GENERAL_INFO_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
        )
        summary = response.choices[0].message.content
        citations = getattr(response, "citations", None) or []
        return {"found": True, "question": question, "summary": summary, "citations": citations}
    except Exception as e:
        return {"found": False, "reason": f"Error during Perplexity search: {e}"}


if __name__ == "__main__":
    # Quick manual test:
    #   python perplexity.py "Besiktas Fans to Salah transfer"
    #   python perplexity.py general "Who is Galatasaray's current captain"
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "general":
        question = " ".join(sys.argv[2:]) or "Who is Galatasaray's current captain"
        result = answer_general_question(question)
    else:
        topic = " ".join(sys.argv[1:]) or "Liverpool transfer news reaction"
        result = search_fan_sentiment(topic)

    if result.get("found"):
        print(result["summary"])
        if result["citations"]:
            print("\nKaynaklar:")
            for c in result["citations"]:
                print(" -", c)
    else:
        print("ERROR:", result.get("reason"))