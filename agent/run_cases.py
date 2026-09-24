"""
Run the agent over the case pack and write the answer files.

    python -m agent.run_cases                     # all 20 -> cases/
    python -m agent.run_cases --case HHG-014      # one case
    python -m agent.run_cases --backend tigergraph

Each answer file follows the Answer Format in the dataset README exactly:
top-level case / evidence_requests / next_best_actions / sar / stop_reason /
tool_calls / tokens / latency_s.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.core import get_dataset                 # noqa: E402
from agent.investigate import investigate          # noqa: E402
from agent.narrate import apply as narrate         # noqa: E402
from agent.tools import get_backend                # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES_DIR = os.path.join(ROOT, "cases")

ORDER = ["case_id", "case", "evidence_requests", "next_best_actions", "sar",
         "stop_reason", "tool_calls", "tokens", "latency_s"]
CASE_ORDER = ["status", "verdict", "fraud_probability", "pattern",
              "pattern_description", "affected_txn_ids", "first_suspicious_txn_id",
              "connected_card_ids", "connected_device_profiles", "exposure_usd",
              "evidence", "similar_prior_cases", "summary", "written_to_graph",
              "graph_case_id"]


def _estimate_tokens(ans: dict) -> int:
    """Characters of prompt-equivalent context the narrator reasoned over."""
    blob = json.dumps({k: v for k, v in ans.items() if k != "_internal"})
    internal = ans.get("_internal", {})
    ctx = json.dumps({k: internal.get(k) for k in ("hist", "dev", "mem")})
    return int((len(blob) + len(ctx)) / 3.6)


def finalize(ans: dict) -> dict:
    ans = narrate(ans)
    ans["tokens"] = _estimate_tokens(ans)
    ans.pop("_internal", None)
    case = {k: ans["case"][k] for k in CASE_ORDER}
    ans["case"] = case
    return {k: ans[k] for k in ORDER}


def run(case_ids=None, backend_kind="local", write_graph=False, verbose=True):
    ds = get_dataset()
    os.makedirs(CASES_DIR, exist_ok=True)
    pack = ds.pack
    if case_ids:
        pack = pack[pack["case_id"].isin(case_ids)]

    writer = None
    if write_graph:
        from agent.tg_writer import CaseWriter
        writer = CaseWriter()

    results = []
    for _, row in pack.iterrows():
        be = get_backend(backend_kind)
        t0 = time.time()
        ans = investigate(row, be)
        ans["_internal"]["trigger_type"] = row["trigger_type"]

        if writer is not None:
            gid = writer.write(ans, row)
            if gid:
                ans["case"]["written_to_graph"] = True
                ans["case"]["graph_case_id"] = gid

        out = finalize(ans)
        out["latency_s"] = round(time.time() - t0, 2)
        path = os.path.join(CASES_DIR, f"{out['case_id']}.json")
        with open(path, "w") as fh:
            json.dump(out, fh, indent=2)
        results.append(out)
        if verbose:
            c = out["case"]
            print(f"{out['case_id']}  {c['verdict']:<10} p={c['fraud_probability']:.2f}  "
                  f"{c['pattern']:<28} exposure=${c['exposure_usd']:>9,.2f}  "
                  f"sar={'Y' if out['sar']['file'] else 'n'}  "
                  f"tools={out['tool_calls']:>2}  -> {os.path.basename(path)}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", action="append", dest="cases")
    ap.add_argument("--backend", default="local", choices=["local", "tigergraph"])
    ap.add_argument("--write-graph", action="store_true",
                    help="also write each case into TigerGraph as an AgentCase vertex")
    a = ap.parse_args()
    res = run(a.cases, a.backend, a.write_graph)
    v = {}
    for r in res:
        v[r["case"]["verdict"]] = v.get(r["case"]["verdict"], 0) + 1
    print("\nverdicts:", v)
    print("reports filed:", sum(1 for r in res if r["sar"]["file"]))
