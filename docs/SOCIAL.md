# Social post

Pick one, add the blog/demo link, post on X or LinkedIn. Tag **@TigerGraphDB**.

---

## LinkedIn (recommended)

> I spent the weekend building a fraud investigation agent on TigerGraph for the
> Hacker House Goa challenge, and the most interesting thing I found wasn't in
> the agent. It was in the data.
>
> The dataset gives every transaction a risk score from the bank's fraud model.
> So I pulled the score distribution out of the bank's own 5,565 closed
> investigations:
>
> · cleared false alarms — median score 0.88, and **none below 0.30**
> · confirmed fraud — median score 0.47, with **1,477 cases below 0.30**
>
> A high score means "someone should look at this." It doesn't mean fraud.
>
> What it does mean is that the interesting question is a graph question: what
> does this transaction look like next to everything else that touched the same
> card, the same device, the same region?
>
> One case arrived as an analyst's hunch about "an unusual device profile." One
> traversal out of the flagged transaction into its device profile and back down
> to that profile's other cards: 28 different cards, 28 unrelated customers, 30
> days, all the same product code. The bank's model scored the whole cluster a
> median of 0.13. Not one of those payments would ever have been flagged — the
> pattern only exists as a shape in the graph.
>
> Then I let the detector run across November and December with no alert to start
> from: 362 rings, 1,330 cards, $531K of transactions. One of the 362 had a
> median model score above 0.7.
>
> Two things I'd pass on to anyone building something similar:
>
> 1. **Most "shared devices" aren't.** A device profile of
> `Windows | Windows 10 | chrome 63.0 | 1920x1080` is shared by 842 unrelated
> cards. It's a browser bucket, not a machine. Rarity is what makes a link
> evidence.
>
> 2. **Your labelled history has a selection bias and it will teach it to you.**
> My first calibration hit AUC 0.983 — because it had learned "high score ⇒
> legitimate" at a coefficient of −9.5. True in the file, only because the bank
> never opened a false-alarm case on a low-scoring alert. Restricting training to
> the alert band gave a believable 0.915.
>
> Policy as code rather than prompt, a probability calibrated on the bank's own
> closed cases, and every claim carrying the graph query that produced it.
>
> Write-up and code below. Thanks @TigerGraphDB for a genuinely well-built
> challenge dataset — the traps in it are the lesson.
>
> #TigerGraph #GraphDatabase #FraudDetection #AIAgents #GraphRAG

---

## X / Twitter (thread)

**1/**
> Built a fraud investigation agent on @TigerGraphDB this weekend.
>
> The dataset scores every transaction with the bank's fraud model. I checked
> that score against the bank's own 5,565 closed cases:
>
> cleared false alarms: median 0.88, none below 0.30
> confirmed fraud: median 0.47, 1,477 below 0.30
>
> The score tells you where to look. Not what's true.

**2/**
> So the real question is a graph question.
>
> One case came in as an analyst's hunch about "an unusual device profile."
>
> One traversal — flagged txn → device profile → its other cards:
>
> 28 cards. 28 unrelated customers. 30 days. Same product code.
> Median model score: 0.13.

**3/**
> None of those payments would ever be flagged. Each is unremarkable alone.
> The pattern only exists as a shape in the graph.
>
> Then I ran the detector across Nov–Dec with no alert at all:
> 362 rings · 1,330 cards · $531K
> 1 of 362 scored above 0.7.

**4/**
> Two traps worth knowing:
>
> • Most "shared devices" aren't. `Windows | Windows 10 | chrome 63.0 |
> 1920x1080` is shared by 842 unrelated cards. Rarity is what makes a link
> evidence.
>
> • My first calibration hit AUC 0.983 by learning "high score ⇒ legitimate."
> Pure selection bias. Real answer: 0.915.

**5/**
> Policy as code, not prompt. Probability calibrated on the bank's closed cases.
> Every claim carries the GSQL query that produced it.
>
> Code + write-up 👇
> [link]
>
> #GraphRAG #FraudDetection #AIAgents
