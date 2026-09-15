# SynFC

**Syn**thetic + **Syn**ergy + **Syn**apse — a LangGraph-based multi-agent
system that simulates how a football club actually makes decisions.

Ask it something — *"Would Ernest Poku suit Beşiktaş?"*, *"What's
Victor Osimhen's injury history?"*, *"Who is our club interested in?"*
— and it routes the question to the relevant departments. Each one
forms a real, data-grounded opinion. If they disagree, they debate. A
Board of Directors renders the final call. The whole thing is built to
look like an actual club's decision-making process, not a chatbot that
happens to know football trivia.

## Why this exists

Most "AI + football" demos are really just an LLM's memorized
knowledge wearing a football costume — ask it anything specific and
current, and it either hallucinates a confident-sounding answer or
quietly falls back on stale training data. SynFC was built around a
different constraint from day one: **every department has to prove it
has real data before it's allowed to have an opinion.** If a
department doesn't have real data on a specific question, its
instruction is to say so plainly — not to fill the gap with something
that sounds plausible.

That constraint shaped almost every architectural decision below.

## The goal isn't "the most logical decision"

It's the decision a real club would plausibly make: sometimes fully
rational, sometimes softened by fan pressure or short-term ambition,
but never random and never purely emotional either. A Data department
that says "we have no evidence this transfer helps" and a Finance
department that says "the cost is manageable" aren't wrong to disagree
— that tension is the point. The Board's job is to sit with that
tension and decide anyway, the way a real boardroom does.

## What makes this different from "an LLM that talks about football"

- **Real data, not recall.** Every department is grounded in an actual
  external source — live Transfermarkt scraping, real wage data, a
  real SQL database, retrieval-augmented search over FIFA's actual
  regulations — never the model's memorized football knowledge alone.
- **Two modes, not one pipeline for everything.** A factual question
  ("who is our club interested in?") and a real decision ("should we
  sign this player?") are fundamentally different asks. SynFC
  classifies which one it's looking at *before* routing. Factual
  questions skip the entire debate/Board machinery entirely — cheaper,
  faster, and the answer reads like information instead of a boardroom
  verdict that nobody asked for.
- **Departments actually disagree.** Conflict detection is real: if
  Finance says yes and Data says no, that's surfaced and debated for
  up to a few rounds before going to the Board, instead of being
  smoothed over into a fake, tidy consensus.
- **Cross-team data flows, but only where it's actually useful.**
  Health's real injury-risk findings reach the Technical team's
  psychologist (dressing-room impact) and Finance's investor
  (investment risk) — and *only* those two, because those are the two
  agents actually positioned to interpret physical risk data through
  their own lens. It isn't broadcast to every department just because
  it exists.

## Architecture

```
question
  → mode classifier (factual vs. decision)
  → department selector (a different prompt for each mode)
  → the relevant departments run in parallel, each grounded in real data
  → mode == "information" → a clean, neutral answer, done
  → mode == "decision" → conflict? → debate rounds → Good Cop / Bad Cop
                                                     → President's final call
```

Each department (Technical, Finance, Media, Data, Scout, Legal, Health)
is itself a small graph, not a single LLM call: a sub-router picks
which of that department's own specialists are relevant, they form
opinions in parallel, and a Director synthesizes one final position —
all built from a single shared factory (`core.py`'s `build_subteam()`)
so every department follows the same pattern instead of duplicating a
few hundred lines of routing/debate/synthesis boilerplate seven times
over.

The **mode split** deserves its own note: it was originally one
combined router call, but that one prompt reliably mis-routed casual
factual questions (like "who's our club interested in?") away from the
right department, because the criteria for "which department matters"
genuinely differ between a factual lookup and a judgment call. Splitting
it into two focused calls — classify first, then select with a
mode-specific prompt — fixed it completely and turned out to be worth
the extra API call.

## Real tool grounding, department by department

| Department | Real data source |
|---|---|
| Legal | Retrieval-augmented search (sentence-transformers + FAISS) over FIFA's actual regulation text, with keyword search as an automatic fallback |
| Data | Text-to-SQL against a real SQLite database of top-5-league player stats — an LLM writes the query, it actually runs against real data |
| Finance | Live Transfermarkt market value/contract data, real wage data (SalaryLeaks), and a real transfer-fee amortization calculation — no guessed defaults when data is missing, the calculation is skipped and that's stated plainly instead |
| Technical | Live squad-depth and performance-stat lookups |
| Media | Live transfer-rumor scraping and live web search, deliberately kept as separate sources depending on whether the question names one specific player or asks for a general list — mixing them caused stale/off-target results |
| Health | Real injury-history records plus research-grounded reference profiles for that injury type, with a genuine effort to never confuse "no data available" with "confirmed clean medical record" — the two look similar but mean opposite things |
| Scout | Market-value and performance data (regional differentiation is a known, documented gap — see Known limitations) |

## A worked example

Ask *"What do you think about the transfer of Bitshiabu to
Galatasaray?"* and here's roughly what happens under the hood:

1. **Mode classifier** reads this as a decision, not a factual lookup.
2. **Department selector** picks Technical, Finance, Data, Media, and
   Health — a full transfer evaluation always includes a medical/
   injury-risk check by default, the same way it always includes cost,
   whether or not the question explicitly mentions either.
3. Each department forms an opinion **grounded in real numbers**: the
   real €18m market value, the real contract running only to 2027 (which
   Finance flags as forcing the entire fee into one accounting year — a
   real amortization problem, not an invented one), the real 1,130
   minutes played last season, the real prior hamstring/knee injury
   history.
4. **No hard conflict** is detected — every department supports the
   move on some condition (loan structure, longer contract, a clean
   medical) rather than flatly opposing it, so it goes straight to the
   Board.
5. **Good Cop and Bad Cop** argue from the *same* real numbers toward
   opposite conclusions — upside vs. risk — and the **President**
   renders a structured final call: pursue a loan with a
   performance-based purchase option, not a permanent transfer, with
   explicit financial and medical conditions attached.

Nothing in that chain is invented. Every number the departments cite
traces back to something a tool actually fetched.

## Engineering lessons, documented as they happened

`DEBUG_NOTES.md` in this repo is a running log of the sneaky bugs this
project actually hit and how they were diagnosed — a date-format
ambiguity that silently misparsed contract dates, a bug where a value's
*origin* (real data vs. a fallback default) was being guessed from the
value itself instead of tracked explicitly, a department's own internal
router still using an outdated rule even after the main router was
fixed, and more. It's kept less as a changelog and more as a "don't
re-derive this the hard way twice" reference — worth reading before
touching the router or any of the tool-grounding code.

`vision.md` is the longer-term architecture spec — what's built, what's
partially built, and what's still just a documented idea (club-identity
injection, MCP exposure, a real action-taking pattern beyond Legal's
mail drafts).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# fill in your API keys
```

```bash
cd synfc_engine
python engine.py
```

### Running the website against the live engine

The website (`website/`) talks to the engine through a small FastAPI/
SSE wrapper (`server.py`), which streams each step of a run to the
browser live — the same step-by-step reveal the CLI prints to a
terminal, just rendered as a real UI.

```bash
# add SYNFC_ACCESS_KEY=... to .env first (see .env.example)
cd synfc_engine
python server.py
```

Then open `website/index.html`. The access key is never committed to
the repo — share it with whoever you want testing the demo out of
band, not by putting it in code.

## Known limitations

- Scout's five regional agents currently see the same non-region-
  filtered data — real regional differentiation isn't built yet.
- The local player-stats database is a season snapshot, not a live
  feed — it needs to be refreshed periodically as seasons progress.
- Club-identity injection (teaching departments a specific club's real
  historical decision patterns, e.g. "this club tends to sign players
  21-26, rarely spends above X in defense") is designed — the data
  source is sourced and the approach is sketched in `vision.md` — but
  not yet wired into any department.

## Tech stack

Python, LangGraph, DeepSeek (OpenAI-compatible API), curl_cffi,
sentence-transformers, FAISS, SQLite, pandas, FastAPI.

## Project layout

```
synfc_engine/    core engine: shared state/LLM wrapper, the router, the graph, the CLI
Tools/           external data sources: scraping, RAG, SQL, live search
Sub_Teams/       the 7 departments
website/         static frontend, talks to server.py over SSE
data/            local datasets
ML_playground/   independent ML experiments, not wired into the engine
DEBUG_NOTES.md   real bugs hit during development, kept as a reference
vision.md        longer-term architecture spec and roadmap
```