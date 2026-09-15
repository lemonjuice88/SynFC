# SynFC — Critical Bug Notes

This file is kept so that if a similar bug shows up again, we can
quickly recall "we already hit this, here's why" instead of re-deriving
it from scratch. Any new critical/sneaky bug found should be added here.

---

## 1) Date-format ambiguity — `_estimate_contract_years` (finance_team.py)

**What happened**: Contract end dates coming from Transfermarkt
sometimes arrive as `30.06.2029` (dots), sometimes as `30/06/2029`
(slashes). The first version of the code only tried the dot format
plus the US-style `%m/%d/%Y` — it never tried the actual European
format, `%d/%m/%Y`. As a result, some queries couldn't parse the date
at all and reported "no data," even though the data genuinely existed.

**Root cause**: The list of formats to try was incomplete, AND
`%m/%d/%Y` / `%d/%m/%Y` can both be **simultaneously valid** for some
dates (e.g. `05/06/2029`), while producing different results (June 5th
vs. May 6th). Whichever one gets tried first can SILENTLY misinterpret
an ambiguous date.

**Fix**: `%d/%m/%Y` (day/month) is now tried BEFORE `%m/%d/%Y`, because
Transfermarkt is Europe-centric and day/month is by far the more likely
real format. If a misread date ever shows up again, check this
ordering first.

**General lesson**: Never assume "one format is enough" for date
parsing — especially for data from international sources, which can
arrive in more than one format, AND the order in which formats are
tried can SILENTLY change the result in some cases (no error is
raised, just a wrong answer — which is more dangerous).

---

## 2) "Guessing the path from the result" bug — `_estimate_contract_years`

**What happened**: The code tried to guess whether a number
(`contract_years`) came from real data or from a fallback default by
looking at the RESULT ITSELF (checking whether it `== 3`). But a real
contract can coincidentally also run for exactly 3 years — in that
case the code mislabeled real data as a "guess."

**General lesson**: Never infer HOW a value was produced (real vs.
estimated) by looking at the VALUE it produced — the two are
independent. The source function needs to report this separately and
explicitly (via a flag, or by returning `None`).

**Final decision**: The fallback estimate was removed entirely — if
there's no data, nothing is guessed anymore, it just says "no data"
directly (per the user's explicit request, on the principle of "don't
just decide on your own").

---

## 3) The router's loose department mapping — `DEPARTMENT_SELECTOR_BILGI_SYSTEM` (engine.py)

**What happened**: There was a rule that routed "a player's contract/
wage/market value" questions to both `finance_team` AND
`technical_team`. But a PURELY financial question like "how much is
the financial burden" has nothing to do with the Technical team's
squad-depth data — it was being called unnecessarily and producing a
pointless "I don't have data on this" answer.

**General lesson**: When setting up loose mappings in router prompts
("topic X -> department A and/or B"), make sure BOTH departments can
actually contribute something useful to that question — "seems
related" isn't enough; the real question is "does it actually have
data for this."

---

## 4) Text written early gets contradicted later — `accounting_node` (finance_team.py)

**What happened**: The amortization note's fixed text was written under
the assumption "no wage data available" (`"no wage data available
yet"`) — but this note was written in the code flow **before** wage
data had even been looked up. When wage data was actually found a
moment later (in the same function), this early-written note ended up
**contradicted** — the report said both "total €43.8M (wages included)"
and "wages not included" at the same time.

**General lesson**: If a piece of text makes an assumption about
information that comes AFTER it ("not available yet," "unknown"),
that assumption can be CONTRADICTED later in the code flow. Fix: have
the text state its own SCOPE ("this figure covers only X") instead of
making assumptions about another section's STATUS ("no Y data" instead
of "see the Y section, if present").

---

## 5) The same bug repeated one layer down — `HEALTH_ROUTER_SYSTEM` (health_team.py)

**What happened**: We made a fix to the main engine's Router so that
Health is included by default in a full transfer decision (the
opposite direction from item 3 — this time a "loosening" fix). But the
Health department's OWN internal router (`HEALTH_ROUTER_SYSTEM`) was
still running on the old, narrow criterion — "don't select the Physio
unless the word injury/fitness explicitly appears in the topic text."
Result: the main engine correctly called Health, but the Physio agent
INSIDE Health never ran, so real injury data (`Thigh problems`, `Hip
injury`) never entered the picture — the Club Doctor said "I have no
report," even though the data genuinely existed and was being fetched
correctly at the tool level.

**General lesson**: When you fix a "default behavior" rule at one
layer (the main Router), check the other layers where the same logic
applies (like departments' own internal routers). A higher layer
making the right call doesn't mean the lower layers automatically
behave correctly too — every layer has its own, independent decision
logic; fixing one doesn't fix the others on its own.

**Diagnostic tip**: A Director/synthesizer agent saying "I don't have
an X report" doesn't necessarily mean "the tool failed." Sometimes it
means "that agent was never run at all" (its internal router didn't
select it). To tell these apart, first test the tool itself in
isolation, then check whether the internal router actually selected
that agent.

---

## General principle (what all of these have in common)

All three of today's bugs trace back to the same root: **moving
forward on "this is probably correct" while writing prompts/code,
without testing against real data.** The line-by-line read-through of
the system messages (ROUTER_SYSTEM, department personas) exists
specifically to catch this kind of sneaky bug ahead of time.

---

## PRIORITY TASK — Kaggle dataset staleness (2024/25 → 2025/26)

**Problem**: `player_stats.sqlite` and the related CSVs (downloaded
from Kaggle) are currently fixed to **2024/25 season** data. The SQL
queries (Data Scientist), squad/stat contexts (Technical Analyst,
Transfer Analyst, Scout) — all of them depend on this fixed data. As
the season progresses, this data **will go stale.**

**What needs to happen**: Either (A) periodically download the 2025/26
update of the Kaggle dataset and rebuild `data/player_stats.sqlite`
(remember: the file is checked via `if not os.path.exists`, so the old
`.sqlite` file needs to be DELETED to pick up fresh data), or (B) move
to a live data source (harder, but stays current automatically).

**Priority**: The user specifically flagged not to take this lightly —
needs to be addressed before release (September 20).

---

## 6) Perplexity's "live" search presenting stale news as current

**What happened**: In the Ernest Poku/Beşiktaş test scenario, the
Reporter agent said "this looks like a profile Solskjaer would want" —
but Solskjaer was **fired from Beşiktaş in August 2025** (after the
Conference League qualifier), replaced by Sergen Yalçın, and according
to one source Yalçın himself was **fired in May 2026**. So the "live
search" presented news that was nearly **a year stale**, with no
freshness warning at all, as if it were current.

**Root cause (not fully confirmed, but most likely)**: Perplexity's
search results probably include older articles written back when
Solskjaer was still in the job, and `SEARCH_SYSTEM_PROMPT`
(`media_web_search_perplexity.py`) doesn't give the model **today's
date**, and never asks it to **verify the currency** of personnel/role
claims (like who the manager or president currently is).

**What needs to happen**: Inject today's date into
`SEARCH_SYSTEM_PROMPT`, plus add an instruction to "verify that
personnel/role claims are still current, and say so explicitly if the
information comes from an older article." This hasn't been implemented
yet — needs to be picked up in a future session.

**Priority**: Needs to be addressed before release (September 20) —
this directly affects the Media department's credibility.
