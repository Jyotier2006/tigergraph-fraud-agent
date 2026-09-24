<div align="center">

# Agentic Fraud Investigation on TigerGraph

**An investigation agent that works a fraud alert the way an analyst does —
traverses the graph, weighs the bank's own closed cases as memory, says how
certain it is, asks for more evidence when the policy demands it, and
recommends a defensible action with the approval route attached.**

Built for the TigerGraph × Hacker House Goa challenge on the HHGOA / IEEE-CIS dataset.

[![TigerGraph](https://img.shields.io/badge/TigerGraph-4.2.5-orange)](https://savanna.tgcloud.io)
[![GSQL](https://img.shields.io/badge/GSQL-11%20installed%20queries-orange)](gsql/03_queries.gsql)
[![Holdout AUC](https://img.shields.io/badge/holdout%20AUC-0.915-blue)](agent/weights.json)
[![Answer files](https://img.shields.io/badge/20%2F20%20cases-validated%200%20errors-brightgreen)](cases/)
[![Python](https://img.shields.io/badge/python-3.11-blue)](requirements.txt)

[The finding](#the-finding) · [Results](#results-on-the-20-exam-cases) · [How it works](#how-it-works) · [Run it](#run-it) · [What we got wrong first](#three-things-that-were-harder-than-they-looked)

</div>

---

## The problem with a good fraud model

Every transaction in this dataset carries a risk score from the bank's detection
model. The obvious agent reads the score, looks at the transaction, decides.

That agent fails — and the dataset is built to show you why. Pull the score
distribution out of the bank's own 5,565 closed investigations:

| | cleared (false alarm) | confirmed fraud |
|---|---:|---:|
| count | 900 | 4,665 |
| median risk score | **0.88** | **0.47** |
| scored below 0.30 | **0** | **1,477** |

Not one false alarm scored below 0.30 — and 1,477 confirmed frauds did.

A high score means *somebody should look at this*, and almost nothing more. The
question worth answering isn't "is the score high?" but **what does this
transaction look like next to everything else that touched the same card, the
same device, the same region** — and that is a graph question.

---

## The finding

Case **HHG-014** arrived as an analyst's hunch: *"several cards this month show
purchases from the same unusual device profile."*

One traversal — flagged transaction → device profile → that profile's other
cards:

```
device profile   SM-G935F Build/NRD90M | Android 7.0 | chrome 62.0 for android | 1920x1080

    28 cards     belonging to 28 unrelated customers
    30 days      54% of every card the profile has ever been seen with (52 lifetime)
    product C    every single transaction
    0.13         median risk score assigned by the bank's model
```

**Not one of those payments would ever be flagged.** Each is unremarkable alone.
The pattern exists only as a shape in the graph.

It matches none of the five documented typologies, so the agent classifies it
`undocumented`, describes it in its own words, and files under rule R9.

Then the memory closes the loop: retrieval surfaced `CC-2649` from the bank's
own history — **confirmed fraud, pattern `undocumented`, sitting on the same
device profile**. The analysts had seen this before and never gave it a name.

### And it doesn't wait to be asked

The same detector, run across November and December with no alert to start from:

<div align="center">

| rings | cards | transactions | scored above 0.7 by the model |
|---:|---:|---:|---:|
| **362** | **1,330** | **$531,647** | **1 of 362** |

</div>

→ [`monitoring/ring_findings.md`](monitoring/ring_findings.md)

---

## Results on the 20 exam cases

<div align="center">

| verdict | count |  | |
|---|---:|---|---:|
| fraud | **10** | suspicious activity reports filed | **3** |
| legitimate | **8** | undocumented pattern identified | **4** |
| uncertain | **2** | cases written back to the graph | **20 / 20** |

</div>

Validated against the Answer Format with [`tests/validate_answers.py`](tests/validate_answers.py):
**0 errors, 0 warnings** — structure, enums, cross-field consistency, approval
routing, exposure arithmetic, and that every id quoted exists in the dataset.

> The dataset warns that roughly half the cases are legitimate and that an agent
> which blocks everything scores badly. Eight clean closes and two honest
> `uncertain` verdicts are the point, not a shortfall. The expensive failure in
> fraud isn't a missed alert — it's blocking a real cardholder at a checkout
> because one model said 0.87.

---

## How it works

```
 trigger ─▶ anchor ─▶ baseline ─▶ window ─▶ spread ─▶ memory
    │          │          │          │         │         │
 alert /    flagged     card's    ±72h on   device    closed cases
 report /     txn     own profile   card    profile   linked by card,
 analyst                                    → other   device, customer
                                              cards
                                                        │
                                                        ▼
                                        assess  (calibrated model
                                                 + documented typologies)
                                                        │
                                          ┌─────────────┴─────────────┐
                                  enough evidence?               not yet
                                          │                          │
                                          ▼                          ▼
                              decide + explain              request evidence
                                          │                          │
                                          ▼                  re-assess ─┘
                              write case to graph
```

### 1 · The graph is the evidence, not a lookup table

Eleven installed GSQL queries — nine investigation tools plus `write_case` and
`case_memory` ([`gsql/03_queries.gsql`](gsql/03_queries.gsql)). Every claim in
every case file carries the query that produced it, so an analyst can re-run it:

```json
{
  "claim": "Device profile 'SM-G935F Build/NRD90M | Android 7.0 | chrome 62.0 for
            android | 1920x1080' carries 28 distinct cards across unrelated
            customers inside a 30-day window (52 cards on this profile in the whole
            dataset, 54% of them in this window). Transactions cluster on product
            code C and the bank's model scored them a median of 0.13, so the
            cluster is invisible to the score.",
  "source": "graph",
  "ref": "query:device_neighbors(device_key='SM-G935F Build/NRD90M | ...', days=30)",
  "entity_ids": ["C01289-K1", "C01996-K1", "C02910-K1", "..."]
}
```

### 2 · Rarity is what makes a device link evidence

A device profile here is `DeviceInfo | OS | browser | screen`. Count them:

```
Windows | Windows 10 | chrome 63.0 | 1920x1080          →  842 cards
unknown-device | unknown-os | unknown-browser | ...     → 1011 cards
SM-G935F Build/NRD90M | Android 7.0 | chrome 62.0 ...   →   52 cards   ← a machine
```

The first two are browser-family buckets, not machines. Treating them as links
connects every alert to hundreds of irrelevant cards and irrelevant closed cases.

So `DeviceProfile` carries `n_cards_total` and `is_specific`, and a shared-origin
finding requires the profile to be **rare** (≤ 60 lifetime cards) *and*
**concentrated** (≥ 45% of its cards inside the window). That one distinction is
the difference between finding the ring and drowning in noise.

### 3 · The probability is calibrated on the bank's own closed cases

`fraud_probability` is not a number someone picked.
[`agent/calibrate.py`](agent/calibrate.py) replays all 5,565 closed
investigations through the **same tools and detectors** the agent runs live and
fits a logistic model on confirmed-fraud vs cleared.

<div align="center">

**Holdout AUC 0.915 · Brier 0.109 · n = 3,122**

</div>

What history *cannot* teach — card testing has 16 historical examples, the device
ring is undocumented by definition — is applied as an explicit, cited override
rather than pretended to be learned.

### 4 · Policy is code, not a prompt

[`agent/policy.py`](agent/policy.py) encodes the actions, the approval routing
table and rules R1–R10. The agent cannot invent an action or mis-route an
approval, and every recommendation names the rule that produced it:

```python
def route_for(action, exposure=0.0):
    if action in AUTO:                                 return "auto"
    if action == "DECLINE_TRANSACTION":                return "L1"
    if action == "BLOCK_CARD":                         return "L1" if exposure <= 2500 else "L2"
    if action in ("BLOCK_ALL_CARDS", "FILE_REPORT"):   return "L2"
```

`BLOCK_CARD` at $2,400 goes to a team lead; at $2,600, to a fraud manager.
That is not a thing to leave to sampling temperature.

### 5 · Uncertainty changes the recommendation

Rule R1: on a single signal below 0.70, verify before you block. The agent
records both recommendations:

```
initial   VERIFY_WITH_CUSTOMER (auto) · MONITOR_CARD (auto) · CREATE_CASE (auto)
          ↓ assumed response
final     BLOCK_CARD (L1) · CREATE_CASE (auto) · FILE_REPORT (L2)
          · MONITOR_CONNECTED_CARDS (auto) · ESCALATE_TO_ANALYST (auto)

changed   the assumed response moved fraud probability from 0.46 to 0.75
```

Customer replies aren't provided in this round, so they are simulated **from the
independent graph evidence** — never from the conclusion — and the assumption is
written into `evidence_requests`, as policy section 5 requires.

### 6 · GraphRAG over policy, typologies and analyst notes

[`agent/graphrag.py`](agent/graphrag.py) chunks the fraud policy by rule, the
five typologies, and the 5,565 analyst narratives, then retrieves what each case
actually turns on. Retrieved text is cited as `source: "document"` evidence:

```
"three small online authorisations then a larger purchase"  →  policy:R5  (0.479)
"customer disputes a monthly subscription they already pay" →  policy:R7  (0.304)
"several cards share the same device profile"               →  policy:R6  (0.504)
```

Chunks are also materialised as `PolicyChunk` vertices, so the same text is
retrievable from inside the graph.

### 7 · Cases are written back — that *is* the memory

Each investigation becomes an `AgentCase` vertex wired to the transactions it
examined, its card, the connected cards, the device profile that linked them and
the closed cases it cited. A later alert on any of those entities retrieves it
through `case_memory`.

All 20 are live on the challenge workspace — **149 vertices, 135 edges** —
verified by reading them back:

```
GET /restpp/graph/Fraud_Investigation/vertices/FI_AgentCase
→ 20 · HHG-001:fraud:0.95 · HHG-003:legitimate:0.08 · HHG-008:uncertain:0.33 …
```

`written_to_graph` is set from the writer's return value, never optimistically.

---

## Three things that were harder than they looked

<details>
<summary><b>1 · The labelled history has a selection bias, and it will happily teach it to you</b></summary>

<br>

First calibration fit: **AUC 0.983**. Suspiciously good — and the dominant
coefficient was `risk_score` at **−9.5**. The model had learned
*high score ⇒ legitimate*.

That is true in the closed-case file, but only because the bank never opened a
false-alarm case on a low-scoring alert (zero cleared cases below 0.30). It is an
artifact of *which investigations got opened*, not a property of fraud. An agent
trained on it would clear every high-scoring alert.

**Fix:** restrict training to the alert band (score ≥ 0.5), where the real prior
is 57% fraud / 43% cleared, and drop the raw score from the feature set.
Result: **AUC 0.915** — believable, and it doesn't invert.

</details>

<details>
<summary><b>2 · A learned feature can flip meaning between regimes</b></summary>

<br>

In the refit, the strongest *exonerating* feature was "identity record marks the
device New" at **−3.8**. That is genuinely what the history says: most false
alarms are somebody buying a phone. The dataset README even warns you —
*"people buy new phones."*

But those are all **model-triggered** alerts. When the trigger is the cardholder
phoning to say *"I never made this purchase,"* a device new to the account stops
being the innocent explanation and becomes corroboration.

Same feature, opposite sign, depending on who raised the alert. So the learned
coefficients are applied only in the regime they were learned in:

```python
SUPPRESS_FOR_DISPUTE = {"device_status_new", "hist_thin", "night_hour",
                        "amount_ratio_log", "amount_over_max"}
```

</details>

<details>
<summary><b>3 · A dispute is not automatically fraud</b></summary>

<br>

Seven of the twenty cases are customers saying *"I never made this purchase."*
The naive agent blocks all seven.

Policy R7 exists for a reason. The dataset has no merchant field, so
"same merchant, same amount" is approximated by same amount under the same
product code — and then:

```
HHG-018   $39.08   this card has already made that exact charge  144 times
HHG-003   $49.00   …61 times over 151 days
HHG-008   $55.68   …14 times
```

Those are charges the cardholder makes routinely. R7: record the dispute, warn
the customer, **do not block**.

</details>

---

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env          # add TG_HOST and TG_SECRET

python -m agent.graph_build             # emit the vertex/edge CSVs
python -m agent.load_tigergraph --all   # schema + queries + load
python -m agent.calibrate               # refit on the closed cases
python -m agent.run_cases --backend tigergraph --write-graph
python ui/build_dashboard.py            # → ui/dashboard.html
```

Without a database, everything still runs:

```bash
python -m agent.run_cases               # LocalBackend, identical traversal semantics
python tests/validate_answers.py        # → all checks passed
python -m agent.monitor                 # the autonomous ring scan
```

`LocalBackend` and `TigerGraphBackend` satisfy the same interface and return the
same shapes, so `--backend` is a switch, not a second implementation.

> **Type-name prefix.** Global type names are shared workspace-wide. If your
> workspace already defines `Card` or `Transaction` (the Savanna starter
> solution does), pass `--prefix FI_`. A clean workspace needs no prefix.
>
> **Savanna auto-stops idle workspaces** — the REST endpoint then answers
> *"Auto start is not enabled for this workspace"*. Resume it from
> Workgroup → Workspace → ⋯ → **Resume** before loading.

---

## Repository

```
agent/
  core.py              dataset → graph-shaped entities and indices
  tools.py             the agent's graph tools (LocalBackend)
  tg_backend.py        the same tools over installed GSQL queries
  tg_client.py         REST++ / GSQL client
  tg_writer.py         case write-back (AgentCase + evidence + edges)
  detectors.py         typology detectors, each returning a quotable Signal
  features.py          one feature extractor, shared by calibration and inference
  calibrate.py         fits the model on closed_cases_history.csv
  weights.json         the fitted coefficients (AUC 0.915)
  policy.py            actions, routing, R1–R10, SAR test, stop conditions
  graphrag.py          policy / typology / analyst-note retrieval
  investigate.py       the investigation loop
  narrate.py           case summary + FinCEN-shaped SAR narrative
  run_cases.py         runs the case pack → cases/*.json
  monitor.py           autonomous ring scan of the exam period
  graph_build.py       emits the vertex/edge CSVs
  load_tigergraph.py   one-command schema + queries + load
gsql/
  01_schema.gsql       vertices, edges, reverse edges
  02_load.gsql         loading jobs
  03_queries.gsql      nine tool queries + write_case + case_memory
mcp/
  tigergraph_mcp.json  TigerGraph MCP server config
  fraud_mcp.py         the investigation tools exposed over MCP
ui/
  dashboard.html       self-contained analyst console — open it in a browser
cases/                 the 20 answer files
monitoring/            the autonomous ring findings
tests/                 answer-format validator
docs/                  technical write-up, demo script
```

### The analyst console

`ui/dashboard.html` is a single self-contained file — no server, no network. The
case queue, the case record, evidence with its graph references, the uncertainty
step, the before/after next-best-action, and the SAR narrative.

---

## Honest limits

- The `K` suffix in published card ids (`C01234-K1`) is an **issuer label that is
  not recoverable from the transaction columns** — 21 customers carry both `K1`
  and `K2` over an identical `card1`. Cards are keyed on `(customer_id, card1)`,
  which is 100% pure against the closed-case transaction sets, and the published
  label is carried alongside rather than guessed.
- `V1–V339`, `C1–C14`, `D1–D15` and the numeric `id_*` columns are **unnamed
  model features**. Where they are used, the evidence says so instead of
  inventing a meaning.
- The dataset has **no merchant field**, so R7's "same merchant, same amount" is
  approximated by same amount under the same product code, and the evidence
  states the proxy.
- Simulated customer responses are derived from the independent graph evidence
  and recorded as assumptions — they are not ground truth.

---

## Dataset

IEEE-CIS Fraud Detection, Vesta Corporation, via the IEEE Computational
Intelligence Society. Customers, calendar, channel, risk scores, closed cases and
the case pack were added by TigerGraph for Hacker House Goa 2026.

<div align="center">
<br>
<sub><b>Graph evidence · a calibrated probability · policy as code · a case record that survives the investigation.</b></sub>
</div>
