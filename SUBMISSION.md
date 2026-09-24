# Submission checklist — TigerGraph Agentic Fraud Investigation

Everything below is in this repository.

| Requirement | Where | Status |
|---|---|---|
| Working agent | `agent/` — run `python -m agent.run_cases` | done |
| GitHub repository | this repo (`git init` done, one commit) | push required |
| Agent output on the 20 cases | `cases/HHG-001.json` … `HHG-020.json` | done, validated |
| Case written to the graph | all 20 `AgentCase` vertices live on Savanna, verified by read-back | done |
| Suspicious activity reports | inside each answer file (`sar`), 3 filed | done |
| Next best action, before + after evidence | `next_best_actions.initial` / `.final` | done |
| TigerGraph Savanna / CE | graph `Fraud_Investigation` created, schema + cases loaded | done |
| GSQL + graph algorithms | `gsql/03_queries.gsql` — 11 queries | done |
| TigerGraph MCP | `mcp/tigergraph_mcp.json` + `mcp/fraud_mcp.py` | done |
| GraphRAG | `agent/graphrag.py` — policy, typologies, analyst notes | done |
| User interface | `ui/dashboard.html` (open in a browser) | done |
| Technical blog post | `docs/BLOG.md` | done — publish and link |
| Demo video 3–5 min | `docs/DEMO.md` is the shot-by-shot script | **you record this** |
| Social post, tag @TigerGraphDB | `docs/SOCIAL.md` — LinkedIn + X drafts | **you post this** |
| Optional: autonomous monitoring | `monitoring/ring_findings.md` | done |

## What you still have to do

1. **Push to GitHub.**
   ```bash
   cd fraud-agent
   gh repo create tigergraph-fraud-agent --public --source=. --push
   # or: git remote add origin <url> && git push -u origin main
   ```
2. **Publish the blog post** (`docs/BLOG.md`) — Medium, Hashnode, dev.to or your own site.
3. **Record the demo** — follow `docs/DEMO.md`. Resume the Savanna workspace first.
4. **Post on X or LinkedIn** — `docs/SOCIAL.md`, tag **@TigerGraphDB**, link the blog.
5. **Submit** at https://forms.gle/yxXzqSULGgZ9VUF56 before 23:59 IST.

## Headline numbers for the form / post

- 20/20 answer files, validated against the Answer Format: **0 errors, 0 warnings**
- All 20 cases written back to TigerGraph as `AgentCase` vertices (149 vertices, 135 edges)
- Verdicts: **10 fraud · 8 legitimate · 2 uncertain**; 3 reports filed
- Probability calibrated on the bank's 5,565 closed cases: **AUC 0.915, Brier 0.109**
- Undocumented pattern found: a rare device profile carrying **28 unrelated cards
  in 30 days**, median model score **0.13**
- Autonomous scan of Nov–Dec: **362 rings · 1,330 cards · $531,647**, of which
  only **1** had a median model score above 0.7
