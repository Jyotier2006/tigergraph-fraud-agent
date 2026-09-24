# Agentic Fraud Investigation on TigerGraph

An investigation agent for the TigerGraph × Hacker House Goa challenge. It takes
a fraud alert, works the graph, weighs the bank's own closed cases as memory,
decides how certain it is, asks for more evidence when the policy says it must,
and recommends a defensible action with the approval route attached.

Built on the HHGOA / IEEE-CIS dataset: 590,742 transactions, 144,432 identity
records, 5,565 closed investigations, 20 exam cases.

---

## What it found

Three of the twenty cases sit on a pattern that is in none of the five
documented typologies, and that the bank's own model scores near zero:

> A **rare, specific device profile** — `SM-G935F Build/NRD90M | Android 7.0 |
> chrome 62.0 for android | 1920x1080` — carries transactions for **28 different
> cards belonging to unrelated customers** inside a 30-day window. That is 54% of
> every card the profile has ever been seen with. All of it is product code `C`,
> and the fraud model scored the cluster a **median of 0.13**.

No single transaction looks wrong. The cluster is only visible by traversing
*out of the alert* into the device profile and back down to its other cards —
which is the entire argument for doing this in a graph. The agent classifies it
as `undocumented`, describes it in its own words, and files under R9.

It also validates against history: the closed-case file contains `CC-2649`,
confirmed fraud, pattern `undocumented`, sitting on the same device profile.

## Results on the 20 exam cases

| | count |
|---|---|
| fraud | 10 |
| legitimate | 8 |
| uncertain | 2 |
| suspicious activity reports filed | 3 |
| undocumented pattern identified | 4 |

The dataset README warns that roughly half the cases are legitimate and that an
agent which blocks everything scores badly. Eight clean closes and two honest
`uncertain` verdicts are the point, not a shortfall.

---

## How it works

```
 trigger ─▶ anchor ─▶ baseline ─▶ window ─▶ spread ─▶ memory
                                                        │
                                                        ▼
                                          assess (calibrated + typologies)
                                                        │
                                            ┌───────────┴───────────┐
                                    enough evidence?            not yet
                                            │                       │
                                            ▼                       ▼
                                    decide + explain        request evidence
                                            │                       │
                                            ▼                       │
                                  write case to graph ◀─── re-assess ┘
```

### 1. The graph is the evidence, not a lookup table

Nine installed GSQL queries are the agent's tools (`gsql/03_queries.gsql`).
Every claim in every case file carries the query that produced it:

```json
{
  "claim": "Device profile '...' carries 28 distinct cards across unrelated customers
            inside a 30-day window (52 cards on this profile in the whole dataset,
            54% of them in this window) ... the bank's model scored them a median
            of 0.13, so the cluster is invisible to the score.",
  "source": "graph",
  "ref": "query:device_neighbors(device_key='SM-G935F Build/NRD90M | ...', days=30)",
  "entity_ids": ["C01289-K1", "C01996-K1", "..."]
}
```

### 2. Rarity is what makes a device link evidence

A "device profile" here is `DeviceInfo | OS | browser | screen`. The profile
`Windows | Windows 10 | chrome 63.0 | 1920x1080` is shared by **842 unrelated
cards** — it is a browser-family bucket, not a machine. Treating it as a link
connects every alert to hundreds of irrelevant cases.

So `DeviceProfile` carries `n_cards_total` and `is_specific`, and a shared-origin
finding requires the profile to be rare (≤ 60 lifetime cards) *and* concentrated
(≥ 45% of its cards inside the window). That single distinction is the difference
between finding the ring and drowning in noise.

### 3. The probability is calibrated on the bank's own closed cases

`fraud_probability` is not a number someone picked. `agent/calibrate.py` replays
all 5,565 closed investigations through the *same* tools and detectors the agent
runs live, and fits a logistic model on confirmed-fraud vs cleared.

**Holdout AUC 0.915, Brier 0.109.**

Two things had to be handled honestly:

- **Selection bias.** The closed-case file contains *no* cleared investigation
  below a score of ~0.7 — the bank only ever opened false-alarm cases on
  high-scoring alerts. Training on the raw file learns "high score ⇒ legitimate",
  which is an artifact of case selection, not a property of fraud. Training is
  therefore restricted to the alert band (score ≥ 0.5), where the real prior is
  57% fraud / 43% cleared.
- **Regime transfer.** "Device marked New" is the strongest *exonerating* feature
  among model-triggered alerts — the cardholder bought a phone. It is not
  exonerating when the cardholder is the one saying the charge isn't theirs, so
  it is suppressed for customer disputes (`SUPPRESS_FOR_DISPUTE`).

What history *cannot* teach — card testing (16 examples), the device ring
(undocumented by definition) — is applied as an explicit, cited override rather
than pretended to be learned.

### 4. Policy is code, not a prompt

`agent/policy.py` encodes the actions, the approval routing table and rules
R1–R10. The agent cannot invent an action, cannot mis-route an approval, and
every recommendation names the rule it came from. `BLOCK_CARD` at $2,400 routes
`L1`; at $2,600 it routes `L2`. `FILE_REPORT` is always `L2`.

### 5. Uncertainty changes the recommendation

Policy R1 says: on a single signal below 0.70, verify before you block. The agent
records what it recommended **before** asking and **after** the response:

```
initial:  VERIFY_WITH_CUSTOMER (auto)  ·  MONITOR_CARD (auto)
final:    BLOCK_CARD (L1)  ·  CREATE_CASE (auto)  ·  FILE_REPORT (L2)
changed:  the assumed response moved probability from 0.46 to 0.75
```

Customer replies are not provided, so they are simulated *from the independent
graph evidence* and the assumption is stated in `evidence_requests`, as policy
section 5 requires.

### 6. GraphRAG over policy, typologies and analyst notes

`agent/graphrag.py` chunks the fraud policy by rule, the five typologies, and
the 5,565 analyst narratives, and retrieves what each case actually turns on.
Retrieved text is cited as `source: "document"` evidence
(`policy:R6 (R6. Shared origin.)`). The chunks are also materialised as
`PolicyChunk` vertices so the same text is retrievable inside the graph.

### 7. Cases are written back — that is the memory

Each investigation becomes an `AgentCase` vertex wired to the transactions it
examined, the card, the connected cards, the device profile that linked them and
the closed cases it cited (`write_case`, `case_memory` in `gsql/03_queries.gsql`).
A later alert on any of those entities retrieves it.

---

## Repository

```
agent/
  core.py            dataset → graph-shaped entities and indices
  tools.py           the agent's graph tools (LocalBackend)
  tg_backend.py      the same tools over installed GSQL queries
  tg_client.py       REST++ / GSQL client
  tg_writer.py       case write-back (AgentCase + evidence + edges)
  detectors.py       the typology detectors, each returning a quotable Signal
  features.py        one feature extractor, shared by calibration and inference
  calibrate.py       fits the model on closed_cases_history.csv
  weights.json       the fitted coefficients (AUC 0.915)
  policy.py          actions, routing, R1–R10, SAR test, stop conditions
  graphrag.py        policy / typology / analyst-note retrieval
  investigate.py     the investigation loop
  narrate.py         case summary + FinCEN-shaped SAR narrative
  run_cases.py       runs the case pack → cases/*.json
  graph_build.py     emits the vertex/edge CSVs
  load_tigergraph.py one-command schema + queries + load
gsql/
  01_schema.gsql     vertices, edges, reverse edges
  02_load.gsql       loading jobs
  03_queries.gsql    the nine tool queries + write_case + case_memory
mcp/
  tigergraph_mcp.json  TigerGraph MCP server config
  fraud_mcp.py         the investigation tools exposed over MCP
ui/
  build_dashboard.py   builds the console
  dashboard.html       self-contained analyst console (open in a browser)
cases/                 the 20 answer files
docs/BLOG.md           technical write-up
docs/DEMO.md           demo script
```

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env          # add TG_HOST and TG_SECRET

python -m agent.graph_build             # build the vertex/edge CSVs
python -m agent.load_tigergraph --all   # schema + queries + load
python -m agent.calibrate               # refit on the closed cases
python -m agent.run_cases --backend tigergraph --write-graph
python ui/build_dashboard.py && open ui/dashboard.html
```

Without a database, everything still runs:

```bash
python -m agent.run_cases      # LocalBackend: identical traversal semantics
```

`LocalBackend` and `TigerGraphBackend` satisfy the same interface and return the
same shapes, so `--backend` is a switch, not a second implementation.

## The live graph

The schema and all 20 agent cases are on the challenge Savanna workspace
(graph `Fraud_Investigation`, TigerGraph 4.2.5). Each case is an `AgentCase`
vertex wired to the transactions it investigated, its card, the connected cards,
the device profile that linked them and the closed cases it cited —
**20 cases, 149 vertices, 135 edges**, verified by reading them back:

```
GET /restpp/graph/Fraud_Investigation/vertices/FI_AgentCase
-> 20 · HHG-001:fraud:0.95 · HHG-003:legitimate:0.08 · HHG-008:uncertain:0.33 ...
```

`written_to_graph` is set from the writer's return value, never optimistically.

**On the type prefix.** That workspace already carried a starter solution
defining global `Card` and `Transaction` types, and global type names are shared
workspace-wide, so the types were created under an `FI_` prefix rather than
dropping another graph's schema:

```bash
python -m agent.load_tigergraph --schema --prefix FI_
```

A clean workspace needs no prefix and uses the names in `gsql/01_schema.gsql`
as written.

**Loading the transaction data.** `python -m agent.load_tigergraph --load`
batches the full dataset over REST++. The machine that produced these answer
files sits behind an egress policy that blocks the TigerGraph host, so the bulk
load has not been run from here — the case write-back above went through a
browser-side bridge to the same REST endpoint. The graph currently holds the
case layer; run `--load` from any machine that can reach the workspace to add
the 590k transactions underneath it.

Savanna auto-stops idle workspaces and the REST endpoint then answers
*"Auto start is not enabled for this workspace"* — resume it from
Workgroup → Workspace → ⋯ → **Resume** before loading.

## Honest limits

- The `K` suffix in published card ids (`C01234-K1`) is an issuer label that is
  **not** recoverable from the transaction columns: 21 customers carry both `K1`
  and `K2` over an identical `card1`. Cards are keyed on `(customer_id, card1)`,
  which is 100% pure against the closed-case transaction sets, and the published
  label is carried alongside rather than guessed.
- `V1–V339`, `C1–C14`, `D1–D15` and the numeric `id_*` columns are unnamed model
  features. Where they are used, the evidence says so instead of inventing a
  meaning.
- The dataset has no merchant field, so R7's "same merchant, same amount" is
  approximated by same amount under the same product code, and the evidence
  states the proxy.

## Dataset

IEEE-CIS Fraud Detection, Vesta Corporation, via the IEEE Computational
Intelligence Society. Customers, calendar, channel, risk scores, closed cases and
the case pack were added by TigerGraph for Hacker House Goa 2026.
