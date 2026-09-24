"""
Case write-back: the agent's memory.

Every investigation becomes an `AgentCase` vertex wired to the transactions it
examined, the card, the connected cards, the device profile that linked them
and the closed cases it cited.  A later alert on any of those entities
retrieves this case through `case_memory` / `similar_prior_cases`, which is
what turns a batch of one-shot investigations into a memory that improves.

If TigerGraph is unreachable the writer degrades to a local JSONL journal
(`data/case_memory.jsonl`) in exactly the same shape, so the pipeline never
silently claims a write that did not happen: `written_to_graph` is set from
the writer's return value.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOURNAL = os.path.join(ROOT, "data", "case_memory.jsonl")


def _vertex_payload(ans: dict, row, gid: str) -> dict:
    c = ans["case"]
    it = ans.get("_internal", {})
    nba = ans["next_best_actions"]
    return {
        "graph_case_id": gid,
        "alert_case_id": ans["case_id"],
        "opened_at": str(row["opened_at"]),
        "status": c["status"],
        "verdict": c["verdict"],
        "fraud_probability": float(c["fraud_probability"]),
        "pattern": c["pattern"],
        "pattern_description": c["pattern_description"],
        "exposure_usd": float(c["exposure_usd"]),
        "summary": c["summary"],
        "stop_reason": ans["stop_reason"],
        "sar_filed": bool(ans["sar"]["file"]),
        "initial_actions": "|".join(a["action"] for a in nba["initial"]),
        "final_actions": "|".join(a["action"] for a in nba["final"]),
        "trigger_type": str(row["trigger_type"]),
        "tool_calls": int(ans["tool_calls"]),
    }


class CaseWriter:
    """Writes AgentCase vertices + their edges into TigerGraph."""

    def __init__(self, client=None):
        self.client = client
        self.live = False
        if client is None:
            try:
                from .tg_client import from_env
                c = from_env()
                if c.ping():
                    self.client = c
                    self.live = True
            except Exception:
                self.client = None
        else:
            self.live = True

    # ------------------------------------------------------------------
    def write(self, ans: dict, row) -> str:
        c = ans["case"]
        gid = f"CASE-2016-{ans['case_id'].split('-')[-1]}"
        payload = _vertex_payload(ans, row, gid)
        it = ans.get("_internal", {})

        vertices = {"AgentCase": {gid: {k: {"value": v} for k, v in payload.items()}}}
        edges = {
            "AgentCase": {
                gid: {
                    "CASE_INVESTIGATES": {"Transaction": {t: {} for t in c["affected_txn_ids"]}},
                    "CASE_ON_CARD": {"Card": {it.get("official_card_id", ""): {}}},
                    "CASE_CONNECTED_CARD": {"Card": {x: {} for x in c["connected_card_ids"]}},
                    "CASE_CITES_PRIOR": {"ClosedCase": {x: {} for x in c["similar_prior_cases"]}},
                    "CASE_LINKS_DEVICE": {"DeviceProfile": {x: {} for x in c["connected_device_profiles"]}},
                }
            }
        }
        # drop empty edge buckets
        for k in list(edges["AgentCase"][gid]):
            inner = edges["AgentCase"][gid][k]
            tgt = next(iter(inner))
            if not inner[tgt] or "" in inner[tgt]:
                inner[tgt].pop("", None)
            if not inner[tgt]:
                edges["AgentCase"][gid].pop(k)

        self._journal(payload, c)

        if not self.live:
            return ""
        try:
            self.client.upsert(vertices=vertices, edges=edges)
            self._write_evidence(gid, c)
            return gid
        except Exception as exc:                                  # pragma: no cover
            print(f"  ! case write-back failed for {ans['case_id']}: {exc}")
            return ""

    # ------------------------------------------------------------------
    def _write_evidence(self, gid: str, c: dict):
        ev_v, ev_e = {}, {}
        for i, ev in enumerate(c["evidence"], 1):
            eid = f"{gid}-E{i}"
            ev_v[eid] = {"claim": {"value": ev["claim"][:2000]},
                         "source": {"value": ev["source"]},
                         "ref": {"value": ev["ref"][:500]},
                         "seq": {"value": i}}
            ev_e[eid] = {}
        if not ev_v:
            return
        self.client.upsert(
            vertices={"CaseEvidence": ev_v},
            edges={"AgentCase": {gid: {"CASE_HAS_EVIDENCE": {"CaseEvidence": ev_e}}}})

    def _journal(self, payload: dict, c: dict):
        os.makedirs(os.path.dirname(JOURNAL), exist_ok=True)
        rec = dict(payload)
        rec["affected_txn_ids"] = c["affected_txn_ids"]
        rec["connected_card_ids"] = c["connected_card_ids"]
        rec["similar_prior_cases"] = c["similar_prior_cases"]
        rec["written_at"] = datetime.utcnow().isoformat(timespec="seconds")
        with open(JOURNAL, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
