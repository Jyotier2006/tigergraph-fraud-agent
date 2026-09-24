# The fraud the score can't see: building an agentic investigator on TigerGraph

*Built for the TigerGraph × Hacker House Goa challenge, on the IEEE-CIS
dataset — 590,742 transactions, 5,565 closed investigations, 20 exam cases.*

---

## The problem with a good fraud model

Every transaction in this dataset carries a risk score from the bank's detection
model. The obvious agent reads the score, looks at the transaction, and decides.

That agent fails, and the dataset is built to show you why. I pulled the score
distribution out of the bank's own closed investigations:

| | cleared (false alarm) | confirmed fraud |
|---|---|---|
| count | 900 | 4,665 |
| median risk score | **0.88** | **0.47** |
| below 0.30 | **0** | 1,477 |

Read the last row again. Not one false alarm scored below 0.30 — and 1,477
confirmed frauds did. A high score means "somebody should look at this," and
almost nothing more.

So the interesting question isn't "is the score high?" It's **what does this
transaction look like next to everything else that touched the same card, the
same device, the same region** — which is a graph question.

## What I built

An investigation loop, not a classifier:

```
trigger → anchor → baseline → window → spread → memory
                                                  ↓
                                   assess (calibrated + typologies)
                                          ↓
                          enough evidence? ──no──→ request evidence
                                          │                 │
                                         yes ←── re-assess ──┘
                                          ↓
                          decide + explain → write case to graph
```

Nine installed GSQL queries are the agent's tools. Every claim it makes carries
the query that produced it, so an analyst can re-run the evidence:

```json
{
  "claim": "Device profile 'SM-G935F Build/NRD90M | Android 7.0 | chrome 62.0
            for android | 1920x1080' carries 28 distinct cards across unrelated
            customers inside a 30-day window (52 cards on this profile in the
            whole dataset, 54% of them in this window). Transactions cluster on
            product code C and the bank's model scored them a median of 0.13,
            so the cluster is invisible to the score.",
  "source": "graph",
  "ref": "query:device_neighbors(device_key='SM-G935F Build/...', days=30)"
}
```

## The finding

Case HHG-014 arrived as an analyst's hunch: *"several cards this month show
purchases from the same unusual device profile."*

Traversing out of the flagged transaction into its device profile and back down
to that profile's other cards found 28 different cards, belonging to 28
unrelated customers, inside 30 days. Every transaction product code `C`. Median
risk score **0.13**.

None of it would ever be flagged. Each payment is unremarkable on its own. The
pattern exists only as a shape in the graph.

It matches none of the five documented typologies, so the agent classifies it
`undocumented`, describes it in its own words, and files under rule R9. Then I
checked the bank's own history — and found `CC-2649`: confirmed fraud, pattern
`undocumented`, sitting on the same device profile. The analysts had seen this
before and never gave it a name.

Then I let the agent scan November and December on its own, with no alert to
start from:

> **362 rings. 1,330 cards. $531,647 of transactions.**
> **1 of the 362** had a median model score above 0.7.

## Three things that were harder than they looked

### 1. Most "shared devices" are not shared devices

A device profile here is `DeviceInfo | OS | browser | screen`. My first
shared-origin detector fired on almost every case, which felt great until I
counted:

```
Windows | Windows 10 | chrome 63.0 | 1920x1080   →  842 distinct cards
unknown-device | unknown-os | unknown-browser     → 1011 distinct cards
SM-G935F Build/NRD90M | Android 7.0 | ...         →   52 distinct cards
```

The first two aren't machines, they're browser-family buckets. Treating them as
links connects every alert to hundreds of irrelevant cards and irrelevant closed
cases.

So the `DeviceProfile` vertex carries `n_cards_total` and `is_specific`, and a
shared-origin finding requires the profile to be **rare** (≤ 60 lifetime cards)
*and* **concentrated** (≥ 45% of its lifetime cards inside the window). That one
distinction is the entire difference between finding the ring and drowning.

### 2. The labelled history has a selection bias, and it will happily teach you the wrong thing

I calibrated `fraud_probability` by replaying all 5,565 closed investigations
through the same tools and detectors the agent runs live, then fitting a
logistic model.

First fit: **AUC 0.983**. Suspiciously good — and the dominant coefficient was
`risk_score` at **−9.5**. The model had learned *high score ⇒ legitimate*.

That's true in the closed-case file, but only because the bank never opened a
false-alarm case on a low-scoring alert (see the table above: zero cleared cases
under 0.30). It's an artifact of which investigations got opened, not a property
of fraud. An agent trained on it would clear every high-scoring alert.

Fix: restrict training to the alert band (score ≥ 0.5), where the real prior is
57% fraud / 43% cleared, and drop the raw score from the feature set. Result:
**AUC 0.915, Brier 0.109** — believable, and it doesn't invert.

### 3. A learned feature can flip meaning between regimes

In the refit, the strongest *exonerating* feature was "identity record marks the
device New" at −3.8. That is genuinely what the history says: most false alarms
are somebody buying a phone. The dataset README even warns you — *"people buy
new phones."*

But those are all **model-triggered** alerts. When the trigger is the cardholder
phoning to say *"I never made this purchase,"* a device that's new to the account
stops being the innocent explanation and starts being corroboration.

Same feature, opposite sign, depending on who raised the alert. So the learned
coefficients are applied in the regime they were learned in, and a short
suppression list handles disputes:

```python
SUPPRESS_FOR_DISPUTE = {"device_status_new", "hist_thin", "night_hour",
                        "amount_ratio_log", "amount_over_max"}
```

Things history *can't* teach — card testing has 16 examples, the device ring is
undocumented by definition — are applied as explicit, cited overrides rather than
dressed up as learned.

## Policy as code, not as prompt

Actions, the approval routing table and rules R1–R10 live in `policy.py`, not in
a system prompt. The agent cannot invent an action or mis-route an approval, and
every recommendation names the rule that produced it:

```python
def route_for(action, exposure=0.0):
    if action in AUTO:              return "auto"
    if action == "BLOCK_CARD":      return "L1" if exposure <= 2500 else "L2"
    if action in ("BLOCK_ALL_CARDS", "FILE_REPORT"): return "L2"
```

`BLOCK_CARD` at $2,400 routes to a team lead; at $2,600, to a fraud manager.
That's not a thing to leave to sampling temperature.

The LLM's job is what it's good at — reading the retrieved policy text and the
analyst narratives, and writing the case summary and the SAR. The decision
itself is graph evidence plus a calibrated model plus an explicit rule.

## Uncertainty is the product

Rule R1: on a single signal below 0.70, verify before you block. So the agent
records both recommendations:

```
initial:  VERIFY_WITH_CUSTOMER (auto) · MONITOR_CARD (auto)
final:    BLOCK_CARD (L1) · CREATE_CASE (auto) · FILE_REPORT (L2)
changed:  the assumed response moved probability from 0.46 to 0.75
```

Customer replies aren't provided in this round, so they're simulated *from the
independent graph evidence* — never from the conclusion — and the assumption is
written into `evidence_requests`.

Across the 20 exam cases: **10 fraud, 8 legitimate, 2 uncertain**, 3 reports
filed. Eight clean closes is the result I wanted. The dataset warns that half the
cases are legitimate and that an agent which blocks everything scores badly; the
expensive failure in fraud isn't a missed alert, it's blocking a cardholder at a
checkout because one model said 0.87.

## What I'd do with more time

- **Vector search over the analyst narratives.** I used TF-IDF, which is
  dependency-free and works, but the closed-case notes are exactly the kind of
  text embeddings are good at. TigerGraph's vector store would replace it.
- **Learn the ring detector instead of specifying it.** The rarity and
  concentration thresholds (60 cards, 45%) came from looking at the distribution.
  With labelled rings they could be fit.
- **Make the second hop cheaper.** Every investigation re-derives the device
  cohort. Materialising a card↔card "shared rare device" edge at load time would
  turn the expensive traversal into a one-hop lookup.
- **Feed the agent's own cases back in properly.** Each case is written back as
  an `AgentCase` vertex wired to its transactions, cards, device and cited priors,
  and `case_memory` reads them. What I haven't done is run the pack *twice* and
  measure whether the second pass is better — which is the only honest test of
  whether the memory is worth anything.

## What I'd tell someone starting this

Spend the first hour on the data, not the agent. Everything that made this work —
the device-profile rarity, the selection bias in the labels, the feature that
flips sign between triggers — came from counting things before writing the
investigation loop. The agent is the easy part; knowing what makes a link
*evidence* is the whole job.

---

*Code, the 20 case files, the analyst console and the autonomous ring scan are in
the repository.*
