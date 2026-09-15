# SynFC — Long-Term Vision & Architecture (v0.2)

> **Status update**: this document originally described a target
> architecture that wasn't built yet. As of this update, most of the
> core structure below is **actually implemented** — the three-layer
> hierarchy exists for all 7 departments, cross-team edges are real,
> and a whole new layer (mode classification) was added that this
> document didn't originally anticipate. Sections are marked ✅ Built /
> 🚧 Partial / 📋 Still open below, so this stays a living spec instead
> of a stale one.

---

## 1) The architecture in one sentence

✅ **Built.** SynFC is a three-layer hierarchical organization:

```
Main Router (now itself two steps -- see section 1.1)
  └─ 7 Teams (each with its own SubRouter)
       └─ Specialist agents within each team (each with its own tools)
            └─ Team Director (synthesizes the agents' reports into ONE opinion)
  └─ Board of Directors (Good Cop + Bad Cop + President -- final, unbiased decision)
```

### 1.1 New layer this vision didn't anticipate: mode classification

✅ **Built.** Before team selection even happens, the Router now
classifies the message as either a **decision** ("karar" -- needs a
judgment call, the full debate/Board pipeline) or **information**
("bilgi" -- a factual question, no debate/Board needed at all). This
was added because plenty of real questions ("who is our club
interested in?") don't need a boardroom verdict, just a clean factual
answer -- routing them through the full pipeline was wasted cost and
latency, and made the answer read like a decision when it wasn't one.

This turned out to require **two separate LLM calls** (mode first,
then department selection with a mode-specific prompt) rather than one
combined call -- a single prompt trying to do both reliably
mis-routed factual questions away from the right department. Full
reasoning is in `DEBUG_NOTES.md`.

---

## 2) The seven teams, one by one

### 2.1 Finance

| Agent | Role | Tools |
|---|---|---|
| Accounting | Reports the club's current financial standing | ✅ Transfermarkt (market value + contract data), ✅ Calculator (amortization math), ✅ **Wage data — resolved via SalaryLeaks** (was an open question in v0.1) |
| Investor | Models the potential return/risk of a transfer | 🚧 Reasons qualitatively; ✅ now also reads the Health department's real injury-risk report (a cross-team edge this document didn't originally spec) when one is available; 📋 a real revenue-projection/opportunity-cost model is still not built |
| Finance Director | Synthesizes Accounting + Investor into a single decision | ✅ Built |
| SubRouter | Decides which agent(s) within Finance a topic concerns | ✅ Built |

### 2.2 Technical Team

| Agent | Role | Tools |
|---|---|---|
| Translator | Analyzes language/cultural fit | 📋 Reasoning-only skeleton — no dedicated fit-percentage model exists |
| Club Psychologist | Projects the psychological/dressing-room impact | ✅ Cross-team edge to Health is real and working — reads Health's actual report during debate rounds |
| Technical Analyst | Assesses tactical fit | ✅ Real squad-depth + performance-stat lookups |
| Head Coach | Synthesizes the team's position | ✅ Built |

### 2.3 Media

| Agent | Role | Tools |
|---|---|---|
| Fans | Measures supporter sentiment | ✅ Local fan-comment data + live web search (Perplexity) |
| Reporter | Tracks press/rumors | ✅ Real Transfermarkt rumor-mill scraping (was a stub in v0.1) + live search, deliberately kept separate depending on whether the question names a specific player or asks for a general list |
| Media Director | Combines both into a report | ✅ Built |

**Change from v0.1**: the originally planned Reddit scraping tool was
dropped in favor of a general live web search (Perplexity) — simpler,
avoided a known bot-protection risk, and covers more sources than
Reddit alone would have.

### 2.4 Data

| Agent | Role | Tools |
|---|---|---|
| Data Scientist | Deep statistical expert | ✅ **Real text-to-SQL** against a SQLite database of top-5-league stats — an LLM writes the actual query |
| Transfer Analyst | Translates stats into transfer-relevant terms | ✅ Real squad-depth + performance data |
| Match Analyst | Match-context analysis | 📋 Still a reasoning-only skeleton — no match/opponent-data tool built |
| Data Director | Synthesizes | ✅ Built |

### 2.5 Scout

| Agent | Role | Tools |
|---|---|---|
| 5 regional agents (Asia, Europe, Africa, South America, North America) | Track players in their own region | 🚧 Real market-value/performance data, but **not region-filtered** — every regional agent currently sees the same data regardless of the player's actual region. Real regional differentiation is the single biggest known gap in the whole system. |
| Scout Director | Aggregates | ✅ Built |

### 2.6 Legal

| Agent | Role | Tools |
|---|---|---|
| Legal Agent (covers both procedural and financial-legal matters — merged into one agent rather than the two originally planned) | Regulatory/compliance reasoning, drafts documents | ✅ **Real RAG** — semantic search (sentence-transformers + FAISS) over FIFA's actual regulation text, with keyword search as a fallback; ✅ mail/letter drafting, verbatim-guaranteed via code-level output splicing (an LLM asked to reproduce a long document exactly tends to paraphrase it even when told not to — this is enforced in code instead, not by asking nicely) |
| Legal Director | Synthesizes | ✅ Built |

### 2.7 Health

| Agent | Role | Tools |
|---|---|---|
| Physio | Physical/injury-risk reporting | ✅ Real injury-history records + research-grounded reference profiles by injury type (clearly distinguishing an actual recorded injury from general population-level research context) |
| Psychologist | Psychological/mental-health reporting | 📋 Still a reasoning-only skeleton — no real clinical data source |
| Club Doctor | Synthesizes | ✅ Built |

### 2.8 Board of Directors

✅ **Built as originally specced** — Good Cop / Bad Cop / President,
deliberately not following the "director synthesizes" pattern since
its job is debiasing through two-sided advocacy, not synthesis.

### 2.9 Main Router

✅ **Built**, and evolved beyond the original spec — see section 1.1's
mode-classification layer, which the original v0.1 document didn't
anticipate at all.

---

## 3) Cross-team dependencies ("edges") — what's actually wired

✅ **Health → Technical Team's Psychologist** — reads Health's real
report, but only during a debate round (round 0 runs every department
independently in parallel, so there's nothing to read yet — this
turned out to be a natural consequence of the graph's own timing, no
special sequencing was needed).

✅ **Health → Finance's Investor** — not in the original v0.1 plan at
all. Added because injury risk is directly relevant to investment
risk, and Investor is the one agent positioned to actually interpret
that connection (as opposed to, say, Media or Legal, which have no
real use for it — cross-team data was deliberately NOT broadcast to
every department, only to the ones that can genuinely interpret it).

📋 **Technical Analyst → Data team's lead** (originally planned) — not
implemented; Technical Analyst currently pulls its own squad/stats
data directly rather than requesting it from Data.

---

## 4) From advice to action — partially realized

The original vision's "draft an email to UEFA" example is **partially
built**: Legal's mail-drafting tool does exactly this pattern — draft
the artifact, show it to the user, never auto-send. The verbatim-
content-guarantee problem (an LLM paraphrasing a document it was told
to reproduce exactly) turned out to be a real, recurring issue, solved
with a code-level splicing mechanism (`agent_output_prefixers` /
`director_output_prefixer` in `core.py`) rather than stronger prompt
wording — prompting alone was tried first and wasn't reliable enough.

📋 Still open: this pattern only exists for Legal's mail drafts. No
other department produces a real artifact yet.

---

## 5) New visions (not in v0.1 at all)

### 5.1 Club-identity injection

Teaching departments (particularly Finance and Technical) a specific
club's **real historical decision patterns** — e.g. "this club tends
to sign players 21-26, rarely spends above X in defense" — derived
from real transfer-history data (a Kaggle dataset covering transfers,
players, clubs, and market valuations has been sourced and its schema
mapped out). Planned approach: a hybrid of direct numeric filtering
(age, fee) plus embedding-based semantic search (reusing the same
RAG pattern already proven for Legal's regulation search) over
qualitative fields (position, league tier, loan-vs-permanent).
**Not yet implemented** — the data is ready, the design direction is
set, but no code has been written.

### 5.2 MCP exposure

Two directions considered, neither built: (a) exposing SynFC's own
tools (`get_transfer_rumors`, `ask_database`, etc.) as MCP tools so
other AI clients could call them directly, or (b) making SynFC itself
an MCP server so "consult SynFC" becomes callable from another AI
tool entirely. Neither changes the existing architecture — both would
be an additional, optional access layer on top of what already exists.

### 5.3 Anti-hallucination discipline as a standing principle

Not a single feature but a pattern applied everywhere now: every
persona is explicitly instructed to distinguish "I have no real data"
from "I have a genuinely positive finding" (e.g. a clean injury
record is good news, not missing information — conflating the two was
a real bug, documented in `DEBUG_NOTES.md`), and never to substitute
memorized/training knowledge for a real tool result. This should be
the default assumption for any new department/agent added going
forward, not something to re-derive each time.

---

## 6) Open questions — updated

1. ~~Wage data source~~ — ✅ **resolved** (SalaryLeaks).
2. **Revenue-projection / opportunity-cost models** — still unresolved,
   still needs real methodology + training data.
3. ~~Synchronizing cross-team edges~~ — ✅ **resolved**, turned out to
   be a natural consequence of round-0-parallel/debate-round-sequential
   graph timing, no explicit synchronization needed.
4. ~~Should "SubRouter + Director synthesis" be a shared factory?~~ —
   ✅ **resolved**, yes: `core.py`'s `build_subteam()` is used by 6 of
   7 departments (Finance is hand-written for historical reasons, could
   be migrated later for consistency, not urgent).
5. **Action-approval UX** — 🚧 partially resolved for Legal's mail
   drafts (draft shown, never auto-sent); no general pattern for other
   departments yet.
6. ~~Reddit scraping~~ — abandoned in favor of general live web search
   (see section 2.3).
7. **Scale/cost** — 🚧 meaningfully improved by the mode-classification
   layer (section 1.1): information-mode questions now skip the
   debate/Board pipeline entirely, which was the single biggest cost
   driver. Decision-mode questions still run the full pipeline as
   originally anticipated.

---

## 7) How this document gets used

Same as before: a specification, not a task list. When picking up a
📋 open item, come back here first, then check `DEBUG_NOTES.md` for
any related lessons already learned the hard way — several of the
"resolved" items above only got resolved after a real bug surfaced the
gap, and that history is worth reading before re-deriving it.