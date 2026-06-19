"""
Router Comparison — the headline experiment (with cost-adjusted scoring).

Trains routers that pick a retrieval method per query, then measures how good
AND how cheap the resulting answers are.

  complexity_only  : question-only features (n_hops, length, entities) — IDENTICAL
                     across a base's 3 conditions, so it cannot tell them apart.
  complexity+probe : adds cheap vector-probe signals.
  grounding_aware  : adds graph-grounding signals (connectivity, subgraph_size,
                     n_query_nodes) computed on the real graph — these DIFFER
                     across conditions and let it route to/away from graph.

METRICS (reported separately, nothing hidden in one number):
  quality   = realized answer score (follow the pick, look up that method's
              actual oracle score on that condition; graph-when-broken = 0).
  cost      = paragraphs of context fed to the generator (what you pay at
              inference). vector~top5, graph~8 (chunks+entities+relations),
              long_context = whole corpus (~20). ASSUMPTION — tune COST_* below.
  cost_adj  = quality - COST_LAMBDA * cost. Penalizes "read everything".

Why cost matters: long-context wins on raw quality only because MuSiQue corpora
are tiny (~20 paragraphs fit in context). That does not scale to real corpora,
so we credit methods for being cheap. We report quality and cost separately too,
so the reader sees the tradeoff rather than trusting one combined score.

Tiny pilot data -> GroupKFold by base_id (out-of-fold predictions), so all 3
conditions of a base stay on the same side of every split (no leakage).

Output: results/router_comparison.json + printed report.

USAGE:
    pip install scikit-learn
    python compare_routers.py
"""

import json
import random
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

FEATURES_PATH = Path("./results/grounding_features.jsonl")
PAIRS_PATH = Path("./data/grounding_pairs.jsonl")  # for per-condition n_paragraphs
OUT_PATH = Path("./results/router_comparison.json")

SEED = 42
N_SPLITS = 5
METHODS = ["vector", "graph", "long_context"]
CONDITIONS = ["grounded", "ablated", "disconnect"]

# --- cost model (paragraphs of context fed to the generator) ---
COST_VECTOR_TOPK = 5  # vector retrieves ~5 chunks
COST_GRAPH_EQUIV = 8  # graph context ~ chunks+entities+relations (from logs)
DEFAULT_NPARA = 20  # fallback corpus size if pairs file missing
COST_LAMBDA = 0.01  # score penalty per paragraph of context

COMPLEXITY = ["n_hops", "query_length", "entity_count"]
PROBE = ["top1_score", "mean_topk_score", "score_gap", "dispersion"]
GROUNDING = ["graph_connectivity", "subgraph_size", "n_query_nodes"]

ROUTERS = {
    "complexity_only": COMPLEXITY,
    "complexity+probe": COMPLEXITY + PROBE,
    "grounding_aware": COMPLEXITY + PROBE + GROUNDING,
}

random.seed(SEED)
np.random.seed(SEED)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_npara():
    """Map condition id -> number of paragraphs (corpus size)."""
    if not PAIRS_PATH.exists():
        return {}
    return {
        r["id"]: r.get("n_paragraphs", DEFAULT_NPARA) for r in load_jsonl(PAIRS_PATH)
    }


def method_cost(method, npara):
    if method == "vector":
        return min(COST_VECTOR_TOPK, npara)
    if method == "graph":
        return min(COST_GRAPH_EQUIV, npara)
    return npara  # long_context reads the whole corpus


def matrix(rows, feat_names):
    return np.array([[r["features"][f] for f in feat_names] for r in rows], dtype=float)


def oof_predictions(X, y, groups):
    preds = np.empty(len(y), dtype=object)
    n_splits = min(N_SPLITS, len(set(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups):
        if len(set(y[tr])) < 2:
            preds[te] = Counter(y[tr]).most_common(1)[0][0]
            continue
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=3000, class_weight="balanced")
        clf.fit(scaler.transform(X[tr]), y[tr])
        preds[te] = clf.predict(scaler.transform(X[te]))
    return preds


def metrics(rows, preds, npara, conds=None):
    idx = [i for i, r in enumerate(rows) if conds is None or r["condition"] in conds]
    if not idx:
        return {"quality": 0.0, "cost": 0.0, "cost_adj": 0.0, "accuracy": 0.0}
    quality = np.mean([rows[i]["scores"][preds[i]] for i in idx])
    cost = np.mean(
        [method_cost(preds[i], npara.get(rows[i]["id"], DEFAULT_NPARA)) for i in idx]
    )
    acc = np.mean([preds[i] == rows[i]["label"] for i in idx])
    return {
        "quality": float(quality),
        "cost": float(cost),
        "cost_adj": float(quality - COST_LAMBDA * cost),
        "accuracy": float(acc),
    }


def evaluate(rows):
    y = np.array([r["label"] for r in rows], dtype=object)
    groups = np.array([r["base_id"] for r in rows])
    npara = load_npara()

    def pack(preds):
        m = metrics(rows, preds, npara)
        m["realized_score"] = m["quality"]  # back-compat alias
        m["routing_accuracy"] = m["accuracy"]
        m["ablated"] = metrics(rows, preds, npara, ["ablated"])
        m["broken"] = metrics(rows, preds, npara, ["ablated", "disconnect"])
        m["grounded"] = metrics(rows, preds, npara, ["grounded"])
        m["prediction_dist"] = dict(Counter(preds.tolist()))
        return m

    results = {}
    for name, feats in ROUTERS.items():
        preds = oof_predictions(matrix(rows, feats), y, groups)
        results[name] = pack(preds)
        results[name]["type"] = "learned"
    for mth in METHODS:
        results[f"always_{mth}"] = pack(np.array([mth] * len(rows), dtype=object))
        results[f"always_{mth}"]["type"] = "baseline"
    results["oracle"] = pack(
        np.array([max(r["scores"], key=r["scores"].get) for r in rows], dtype=object)
    )
    results["oracle"]["type"] = "ceiling"
    rnd = random.Random(SEED)
    results["random"] = pack(
        np.array([rnd.choice(METHODS) for _ in rows], dtype=object)
    )
    results["random"]["type"] = "floor"
    return {"n_conditions": len(rows), "cost_lambda": COST_LAMBDA, "routers": results}


def report(rep):
    order = [
        "oracle",
        "grounding_aware",
        "complexity+probe",
        "complexity_only",
        "always_vector",
        "always_graph",
        "always_long_context",
        "random",
    ]
    R = rep["routers"]
    L = [
        "=" * 80,
        "ROUTER COMPARISON  (cost = paragraphs of context; lambda="
        f"{rep['cost_lambda']})",
        "=" * 80,
        f"Conditions: {rep['n_conditions']}\n",
        f"  {'router':22s}{'acc':>6s}{'quality':>9s}{'cost':>7s}{'cost_adj':>10s}"
        f"{'abl_q':>8s}{'abl_adj':>9s}",
    ]
    L.append("  " + "-" * 69)
    for name in order:
        if name not in R:
            continue
        m = R[name]
        L.append(
            f"  {name:22s}{m['accuracy']:6.2f}{m['quality']:9.3f}{m['cost']:7.1f}"
            f"{m['cost_adj']:10.3f}{m['ablated']['quality']:8.3f}"
            f"{m['ablated']['cost_adj']:9.3f}"
        )
    co, ga = R["complexity_only"], R["grounding_aware"]
    L += [
        "\n" + "-" * 80,
        "HEADLINE (grounding_aware vs complexity_only):",
        f"  cost-adjusted overall : {ga['cost_adj']:.3f} vs {co['cost_adj']:.3f}  "
        f"({ga['cost_adj']-co['cost_adj']:+.3f})",
        f"  cost-adjusted ablated : {ga['ablated']['cost_adj']:.3f} vs {co['ablated']['cost_adj']:.3f}  "
        f"({ga['ablated']['cost_adj']-co['ablated']['cost_adj']:+.3f})",
        f"  quality on broken     : {ga['broken']['quality']:.3f} vs {co['broken']['quality']:.3f}  "
        f"({ga['broken']['quality']-co['broken']['quality']:+.3f})",
        f"  prediction mix  complexity_only={co['prediction_dist']}",
        f"                  grounding_aware={ga['prediction_dist']}",
        "=" * 80,
    ]
    text = "\n".join(L)
    print("\n" + text)
    return text


def main():
    rows = load_jsonl(FEATURES_PATH)
    rep = evaluate(rows)
    rep["report_text"] = report(rep)
    OUT_PATH.write_text(json.dumps(rep, indent=2))
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
