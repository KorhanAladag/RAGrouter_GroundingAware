"""
Feature Extraction for the grounding router.

Turns each oracle-scored condition into a feature vector + label, producing the
router's training data. Three feature groups:

  COMPLEXITY (identical across a base's 3 conditions -- question text is the same):
    n_hops          number of reasoning hops (from musique_base)
    query_length    tokens in the question
    entity_count    capitalized/quoted entity mentions in the question

  PROBE (one cheap vector pass over the corpus; differs across conditions):
    top1_score      best query-chunk cosine
    mean_topk_score average over top-k chunks
    score_gap       top1 minus k-th
    dispersion      spread of top-k chunk embeddings

  GROUNDING (from the REAL LightRAG graph; differs across conditions):
    graph_connectivity  fraction of query entities sharing one component
    subgraph_size       size of the k-hop neighborhood of query entities
    n_query_nodes       how many query entities are in the graph

ALL features are answer-INDEPENDENT (inference-safe): at query time the router
does not know the answer. We deliberately do NOT use answer_reachable as a
feature -- that needs the gold answer and would be leakage. It stays in
validation only.

LABEL = oracle winner (the empirically best method on that condition).

Output: results/grounding_features.jsonl  (one row per condition)

USAGE:
    python extract_grounding_features.py     # after run_grounding_oracle.py
"""

import re
import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import requests

from graph_features import grounding_features
from lightrag_core import OLLAMA_HOST, EMBED_MODEL

ORACLE_PATH = Path("./results/oracle_raw.jsonl")
PAIRS_PATH = Path("./data/grounding_pairs.jsonl")
BASE_PATH = Path("./data/musique_base.jsonl")
OUT_PATH = Path("./results/grounding_features.jsonl")

COMPUTE_PROBE = True
PROBE_K = 5
METHODS = ["vector", "graph", "long_context"]
CONDITIONS = ["grounded", "ablated", "disconnect"]


# ---------------------------------------------------------------------------
# Complexity features
# ---------------------------------------------------------------------------
def candidate_query_terms(question):
    terms = set()
    for m in re.findall(r"[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*", question):
        terms.add(m)
    for m in re.findall(r'"([^"]+)"', question):
        terms.add(m)
    return {t for t in terms if len(t) > 2}


def complexity_features(question, n_hops):
    return {
        "n_hops": int(n_hops),
        "query_length": len(question.split()),
        "entity_count": len(candidate_query_terms(question)),
    }


# ---------------------------------------------------------------------------
# Probe features (cheap vector pass)
# ---------------------------------------------------------------------------
def ollama_embed_texts(texts):
    r = requests.post(
        f"{OLLAMA_HOST}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=180,
    )
    r.raise_for_status()
    return np.array(r.json()["embeddings"], dtype=float)


def probe_features(question, source_docs):
    default = {
        "top1_score": 0.0,
        "mean_topk_score": 0.0,
        "score_gap": 0.0,
        "dispersion": 0.0,
    }
    chunks = [d for d in source_docs if d.strip()]
    if not chunks:
        return default
    try:
        embs = ollama_embed_texts([question] + chunks)
    except Exception as e:
        print(f"    probe embed error: {type(e).__name__}: {e}")
        return default
    q, c = embs[0], embs[1:]
    q = q / (np.linalg.norm(q) + 1e-9)
    c = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-9)
    sims = c @ q
    order = np.argsort(-sims)
    k = min(PROBE_K, len(sims))
    top = sims[order[:k]]
    if k > 1:
        te = c[order[:k]]
        pair = te @ te.T
        iu = np.triu_indices(k, k=1)
        dispersion = float(1 - np.mean(pair[iu]))
        score_gap = float(top[0] - top[-1])
    else:
        dispersion = 0.0
        score_gap = 0.0
    return {
        "top1_score": float(top[0]),
        "mean_topk_score": float(np.mean(top)),
        "score_gap": score_gap,
        "dispersion": dispersion,
    }


# ---------------------------------------------------------------------------
# Load + join
# ---------------------------------------------------------------------------
def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    if not ORACLE_PATH.exists():
        raise FileNotFoundError("Run run_grounding_oracle.py first.")

    oracle = {r["id"]: r for r in load_jsonl(ORACLE_PATH)}
    pairs = {r["id"]: r for r in load_jsonl(PAIRS_PATH)}
    base_hops = {r["base_id"]: r["n_hops"] for r in load_jsonl(BASE_PATH)}

    rows = []
    for i, (qid, orc) in enumerate(oracle.items(), 1):
        pair = pairs.get(qid)
        if pair is None:
            continue
        question = pair["question"]
        source_docs = pair["source_docs"]
        n_hops = base_hops.get(pair["base_id"], 2)

        feats = {}
        feats.update(complexity_features(question, n_hops))
        if COMPUTE_PROBE:
            feats.update(probe_features(question, source_docs))
        try:
            gf = grounding_features(source_docs, question)
        except Exception as e:
            print(f"    grounding error {qid}: {type(e).__name__}: {e}")
            gf = {"graph_connectivity": 0.0, "subgraph_size": 0, "n_query_nodes": 0}
        feats.update(gf)

        rows.append(
            {
                "id": qid,
                "base_id": pair["base_id"],
                "condition": pair["condition"],
                "split": pair.get("split"),
                "label": orc["winner"],  # what the router predicts
                "expected_graph_useful": pair.get("expected_graph_useful"),
                "features": feats,
                "scores": orc["scores"],  # kept for analysis
            }
        )
        if i % 20 == 0:
            print(f"  {i}/{len(oracle)} conditions")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(rows)} feature rows -> {OUT_PATH}")
    report(rows)


def report(rows):
    feat_names = list(rows[0]["features"].keys()) if rows else []
    # mean of each grounding-ish feature by condition (does it separate?)
    print("\nMean features by condition (watch grounding features separate):")
    header = "  " + f"{'feature':18s}" + "".join(f"{c:>13s}" for c in CONDITIONS)
    print(header)
    for fn in feat_names:
        by_c = {c: [] for c in CONDITIONS}
        for r in rows:
            if r["condition"] in by_c:
                by_c[r["condition"]].append(r["features"][fn])
        line = f"  {fn:18s}"
        for c in CONDITIONS:
            vals = by_c[c]
            line += f"{(sum(vals)/len(vals) if vals else 0):13.3f}"
        print(line)

    print("\nLabel (oracle winner) distribution by condition:")
    for c in CONDITIONS:
        cnt = Counter(r["label"] for r in rows if r["condition"] == c)
        tot = sum(cnt.values()) or 1
        print(
            f"  {c:11s} "
            + ", ".join(f"{m}:{cnt[m]}({cnt[m]/tot:.0%})" for m in METHODS)
        )

    print("\nNote: complexity features are identical across a base's 3 conditions;")
    print("grounding features should differ (esp. subgraph_size / n_query_nodes).")
    print("That difference is the signal the grounding router uses and the")
    print("complexity-only router cannot see.")


if __name__ == "__main__":
    main()
