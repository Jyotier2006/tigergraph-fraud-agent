"""
GraphRAG: ground the agent in the policy, the typologies and the closed-case
narratives instead of handing an LLM raw rows.

Two retrievers:

  * `PolicyIndex`   - the fraud policy, the five documented typologies and the
    regulatory references, chunked by section and rule number.  Retrieval is
    TF-IDF over the chunk text; each chunk keeps its rule id so a retrieved
    chunk can be cited as `policy:R5` in a case's evidence.
  * `CaseNarrativeIndex` - the 5,565 closed-case analyst notes.  This is the
    text half of case memory: the graph gives *which* prior cases connect,
    this gives *what the analyst wrote* about cases that read the same way.

Both indices are also materialised as `PolicyChunk` vertices so the same text
is retrievable from inside TigerGraph (see gsql/01_schema.gsql).
"""
from __future__ import annotations

import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY_MD = os.path.join(ROOT, "data", "raw", "README.md")


def _split_sections(text: str) -> list[dict]:
    """Chunk the policy document by heading and by rule."""
    chunks: list[dict] = []
    # rules R1..R10 are the unit an investigation cites
    for m in re.finditer(r"\*\*(R\d+)\.\s*(.+?)\*\*\s*(.*?)(?=\n\*\*R\d+\.|\n###|\n---|\Z)",
                         text, flags=re.S):
        rule, title, body = m.group(1), m.group(2).strip(), m.group(3).strip()
        chunks.append({"chunk_id": f"policy-{rule}", "doc": "Fraud Policy",
                       "section": f"{rule}. {title}", "rule_id": rule,
                       "text": f"{rule}. {title} {body}"})
    # the five documented typologies
    for m in re.finditer(r"\*\*(\d)\.\s*(.+?)\.\*\*\s*(.*?)(?=\n\*\*\d\.|\n##|\Z)",
                         text, flags=re.S):
        n, title, body = m.group(1), m.group(2).strip(), m.group(3).strip()
        if len(body) < 40:
            continue
        chunks.append({"chunk_id": f"typology-{n}", "doc": "Known fraud patterns",
                       "section": f"Pattern {n}: {title}", "rule_id": "",
                       "text": f"Pattern {n} {title}. {body}"})
    # the remaining policy sections (actions, routing, 3a, stopping, exposure)
    for m in re.finditer(r"\n### (\d+[a-z]?\.?\s*.+?)\n(.*?)(?=\n### |\n---|\Z)",
                         text, flags=re.S):
        title, body = m.group(1).strip(), m.group(2).strip()
        cid = "policy-" + re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40]
        chunks.append({"chunk_id": cid, "doc": "Fraud Policy", "section": title,
                       "rule_id": "", "text": f"{title} {body}"})
    return chunks


class _TfIdf:
    """Small dependency-free TF-IDF; keeps the agent runnable anywhere."""

    def __init__(self, docs: list[str]):
        self.docs = docs
        self.vocab: dict[str, int] = {}
        rows = []
        for d in docs:
            tf: dict[int, float] = {}
            for w in self._tok(d):
                i = self.vocab.setdefault(w, len(self.vocab))
                tf[i] = tf.get(i, 0.0) + 1.0
            rows.append(tf)
        n = len(docs)
        df = np.zeros(len(self.vocab))
        for tf in rows:
            for i in tf:
                df[i] += 1
        self.idf = np.log((1 + n) / (1 + df)) + 1.0
        self.M = np.zeros((n, len(self.vocab)), dtype=np.float32)
        for r, tf in enumerate(rows):
            for i, v in tf.items():
                self.M[r, i] = v * self.idf[i]
        norms = np.linalg.norm(self.M, axis=1, keepdims=True)
        self.M /= np.where(norms == 0, 1, norms)

    @staticmethod
    def _tok(s: str) -> list[str]:
        return re.findall(r"[a-z][a-z_]{2,}", s.lower())

    def query(self, q: str, k: int = 4) -> list[tuple[int, float]]:
        v = np.zeros(len(self.vocab), dtype=np.float32)
        for w in self._tok(q):
            i = self.vocab.get(w)
            if i is not None:
                v[i] += self.idf[i]
        nv = np.linalg.norm(v)
        if nv == 0:
            return []
        sims = self.M @ (v / nv)
        idx = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in idx if sims[i] > 0]


class PolicyIndex:
    def __init__(self, path: str = POLICY_MD):
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        self.chunks = _split_sections(text)
        self.index = _TfIdf([c["text"] for c in self.chunks])

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        out = []
        for i, score in self.index.query(query, k):
            c = dict(self.chunks[i])
            c["score"] = round(score, 3)
            out.append(c)
        return out

    def by_rule(self, rule_id: str) -> dict | None:
        for c in self.chunks:
            if c.get("rule_id") == rule_id:
                return c
        return None

    def as_vertices(self) -> dict:
        return {c["chunk_id"]: {"doc": {"value": c["doc"]},
                                "section": {"value": c["section"]},
                                "text": {"value": c["text"][:8000]},
                                "rule_id": {"value": c["rule_id"]}}
                for c in self.chunks}


class CaseNarrativeIndex:
    """TF-IDF over the closed-case analyst notes -- the text half of memory."""

    def __init__(self, closed_df):
        self.ids = closed_df["case_id"].tolist()
        self.meta = closed_df.set_index("case_id")
        notes = closed_df["analyst_notes"].fillna("").tolist()
        self.index = _TfIdf(notes)

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        out = []
        for i, score in self.index.query(query, k):
            cid = self.ids[i]
            r = self.meta.loc[cid]
            out.append({"case_id": cid, "score": round(score, 3),
                        "outcome": r["outcome"], "pattern": r["pattern"],
                        "analyst_notes": str(r["analyst_notes"])[:400]})
        return out


_POLICY: PolicyIndex | None = None


def policy_index() -> PolicyIndex:
    global _POLICY
    if _POLICY is None:
        _POLICY = PolicyIndex()
    return _POLICY


def cite(rule_id: str) -> dict | None:
    """Return an evidence-shaped citation for a policy rule."""
    c = policy_index().by_rule(rule_id)
    if not c:
        return None
    return {"claim": c["text"][:400].replace("\n", " "),
            "source": "document",
            "ref": f"policy:{rule_id} ({c['section']})",
            "entity_ids": []}
