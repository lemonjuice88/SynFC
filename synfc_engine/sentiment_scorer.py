"""
sentiment_scorer.py — SynFC background opinion-STANCE scoring
===================================================================
Scores each role's opinion on a 1-10 scale of how strongly it commits
to a for/against position -- used by collect_node to quietly filter
out genuinely neutral/undecided opinions from further debate rounds
before conflict detection runs. Nothing here is ever shown to the user.

IMPORTANT LESSON FROM v1 OF THIS FILE:
The first version used nlptown/bert-base-multilingual-uncased-sentiment,
a product-review STAR-RATING classifier, to estimate the 1-10 score.
In real testing this was unreliable: that model measures emotional
TONE ("does this text sound positive or negative, like an Amazon
review"), not OPINION STANCE STRENGTH ("how committed is this speaker
to a for/against position"). A genuinely hedgy "I can't form a clear
opinion, we'll just have to see" statement can still contain
positive-sounding words ("young, fit, promising") and get scored as
positive-toned by a review classifier, even though it's actually
neutral/undecided in intent. Tone and stance-commitment are different
axes -- we needed the second one, not the first.

This version instead asks a small, cheap LLM call (reusing the same
DeepSeek model already configured for the rest of SynFC) directly for
a STANCE score. This understands football-domain language, hedging,
and idiom far better than a general-purpose review classifier, at the
cost of one extra small API call per opinion instead of a free local
model running entirely offline.

The old review-sentiment implementation is kept below as
score_opinion_hf() for anyone who wants to experiment with a fully
local, zero-API-cost scorer (e.g. the "pipeline into an open-source
model" angle) -- just know its numbers track emotional tone, not
opinion stance strength, so it is NOT a drop-in equivalent.

Setup: nothing extra beyond what engine.py already needs (this reuses
the same DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL env vars).
"""

import os
from typing import Optional

from openai import OpenAI

# The "neutral zone": scores in this (inclusive) range are treated as
# "no real position either way" and get filtered out of further debate
# rounds by engine.py's collect_node.
NEUTRAL_LOW = 4.0
NEUTRAL_HIGH = 6.0

_DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
_DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
_DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=_DEEPSEEK_API_KEY, base_url=_DEEPSEEK_BASE_URL)
    return _client


_STANCE_SYSTEM = """\
You will be given a single opinion from a football club decision-making
board. Rate how STRONGLY this opinion commits to a for/against position
on whatever is being discussed -- NOT how emotionally positive or
negative its tone is.

Reply with a number from 1 to 10:
- 1 = strongly AGAINST
- 10 = strongly FOR
- 5 or 6 = genuinely undecided/neutral -- hedges, says "we'll have to
  see", refuses to commit, presents both sides without concluding, or
  explicitly states it lacks the information to form a clear view

A hedgy statement stays near 5-6 even if it uses positive-sounding
words (e.g. "young, promising") while explicitly refusing to commit to
a conclusion -- judge the SPEAKER'S COMMITMENT to a position, not the
words' surface tone.

Reply ONLY with a single number (one decimal allowed, e.g. "7.5"),
nothing else -- no explanation, no units.
"""


def score_opinion(text: str) -> float:
    """Returns a 1.0-10.0 float: how strongly the opinion commits to a
    for/against stance (NOT emotional tone -- see module docstring)."""
    if not text or not text.strip():
        return 5.5  # empty input -> dead center, treated as neutral

    client = _get_client()
    response = client.chat.completions.create(
        model=_DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": _STANCE_SYSTEM},
            {"role": "user", "content": text[:1500]},
        ],
        temperature=0.0,
    )
    raw = response.choices[0].message.content.strip()

    try:
        score = float(raw.split()[0])
    except (ValueError, IndexError):
        return 5.5  # couldn't parse the model's reply -> fail safe, treat as neutral

    return max(1.0, min(10.0, score))


def is_neutral(score: float, low: Optional[float] = None, high: Optional[float] = None) -> bool:
    """Convenience wrapper so callers don't have to import the two
    threshold constants separately."""
    low = NEUTRAL_LOW if low is None else low
    high = NEUTRAL_HIGH if high is None else high
    return low <= score <= high


# ---------------------------------------------------------------------
# ALTERNATE: fully local, zero-API-cost scorer using an open-source
# review-sentiment model. Kept for reference/experimentation only --
# NOT used by default (score_opinion() above is what collect_node
# calls). Its scores track emotional TONE, not opinion STANCE
# STRENGTH -- see the module docstring for why that mattered here.
# ---------------------------------------------------------------------
_HF_MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
_hf_pipeline = None


def score_opinion_hf(text: str) -> float:
    global _hf_pipeline
    if not text or not text.strip():
        return 5.5
    if _hf_pipeline is None:
        from transformers import pipeline
        _hf_pipeline = pipeline("sentiment-analysis", model=_HF_MODEL_NAME, top_k=None)
    result = _hf_pipeline(text[:1000])
    scores = result[0] if isinstance(result[0], list) else result
    expected_stars = sum(int(item["label"][0]) * item["score"] for item in scores)
    return round(1 + (expected_stars - 1) * (9 / 4), 2)


if __name__ == "__main__":
    # Quick manual test:
    #   python sentiment_scorer.py "Net bir kanaat belirtemem, izleyip göreceğiz."
    import sys

    text = " ".join(sys.argv[1:]) or "It could go either way, hard to say honestly."
    score = score_opinion(text)
    print(f"Text: {text!r}")
    print(f"Score: {score}/10  (neutral: {is_neutral(score)})")