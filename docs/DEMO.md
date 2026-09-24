# Demo script — 3 to 5 minutes

Record the screen. Keep the analyst console (`ui/dashboard.html`) open in one
window and a terminal in another. Timings are a guide, not a script to read out.

---

## 0:00 — 0:25 · The problem

> "Every transaction in this dataset already has a fraud score from the bank's
> model. So I pulled the score distribution out of the bank's own closed
> investigations."

**Show:** the table (slide, or `docs/BLOG.md`).

| | cleared | confirmed fraud |
|---|---|---|
| median risk score | 0.88 | 0.47 |
| below 0.30 | 0 | 1,477 |

> "Not one false alarm scored below 0.30 — and 1,477 confirmed frauds did. The
> score tells you where to look. It doesn't tell you what's true. That's the
> job."

---

## 0:25 — 1:10 · The loop

**Show:** the diagram in `README.md`, then scroll `gsql/03_queries.gsql`.

> "Nine installed GSQL queries are the agent's tools. It anchors on the flagged
> transaction, builds the cardholder's own baseline, looks at the window around
> the alert, then traverses *out* — into the device profile, the billing region —
> and finally retrieves the bank's closed cases as memory.
>
> Every claim it makes carries the query that produced it."

---

## 1:10 — 2:20 · Case HHG-014, the finding

**Show:** run it live.

```bash
python -m agent.run_cases --case HHG-014
```

Then open **HHG-014** in the console.

> "This one arrived as an analyst's hunch: *several cards this month show
> purchases from the same unusual device profile.*
>
> The agent traverses out of the flagged transaction into its device profile and
> back down to that profile's other cards — and finds 28 different cards
> belonging to 28 unrelated customers inside 30 days. All product code C. The
> bank's model scored the whole cluster a median of **0.13**."

**Point at the evidence panel** — the graph call, the 27 connected card ids.

> "Not one of these would ever be flagged. Each payment is unremarkable alone.
> The pattern only exists as a shape in the graph.
>
> It matches none of the five documented typologies, so the agent calls it
> `undocumented`, describes it in its own words, and files under R9."

**Then the payoff** — scroll to `similar_prior_cases`:

> "And it retrieved `CC-2649` from the bank's history: confirmed fraud, pattern
> `undocumented`, on the same device profile. The analysts had seen this before
> and never named it."

---

## 2:20 — 3:00 · Uncertainty and the changing recommendation

**Show:** open a case with an evidence request (**HHG-019** or **HHG-011**).

> "Policy R1 says: on a single signal below 0.70, verify before you block. So the
> agent records both recommendations — what it advised *before* asking the
> cardholder, and what it advises *after*."

**Point at the two columns, and the approval routes.**

> "Approval routing is code, not a prompt. `BLOCK_CARD` under $2,500 goes to a
> team lead; over, to a fraud manager. `FILE_REPORT` is always L2. The agent
> recommends — only the `auto` actions execute."

**Then show a clean close (HHG-003 or HHG-018):**

> "This cardholder disputed a $39.08 charge. The card has already made that exact
> charge 144 times. That's rule R7 — dispute recorded, customer warned, no block.
> Eight of the twenty close clean. Blocking everything scores badly, and it
> should."

---

## 3:00 — 3:40 · Calibration, honestly

**Show:** `agent/weights.json` and the calibration output.

> "The fraud probability isn't a number I picked. It replays all 5,565 closed
> investigations through the same tools the agent runs live and fits a model.
> Holdout AUC 0.915.
>
> My first fit scored 0.983 — because it had learned *high score means
> legitimate*, at a coefficient of minus nine. That's real in the file, but only
> because the bank never opened a false-alarm case on a low-scoring alert. It's
> a selection artifact. Training is restricted to the alert band, and the raw
> score is dropped from the features."

---

## 3:40 — 4:20 · It doesn't wait to be asked

```bash
python -m agent.monitor
```

**Show:** `monitoring/ring_findings.md`.

> "Same detector, no alert, across November and December: **362 rings, 1,330
> cards, half a million dollars of transactions**. One of the 362 had a median
> model score above 0.7.
>
> That's the argument for the graph in one line."

---

## 4:20 — 4:50 · Memory

**Show:** `write_case` / `case_memory` in `gsql/03_queries.gsql`, then
`data/case_memory.jsonl`.

> "Each investigation is written back as an `AgentCase` vertex wired to the
> transactions it examined, the card, the connected cards, the device profile and
> the closed cases it cited. A later alert on any of those retrieves it. The
> closed cases were the bank's memory; these become the agent's."

---

## 4:50 — 5:00 · Close

> "Graph evidence, a calibrated probability, policy as code, and a case record
> that survives the investigation. Everything's in the repo — the 20 case files,
> the console, and the ring scan."

---

### Pre-flight

- [ ] `python -m agent.run_cases` runs clean
- [ ] `ui/dashboard.html` opens and HHG-014 is selected
- [ ] `monitoring/ring_findings.md` generated
- [ ] terminal font large enough to read at 1080p
- [ ] if demoing live TigerGraph: workspace **resumed** (Savanna auto-stops)
