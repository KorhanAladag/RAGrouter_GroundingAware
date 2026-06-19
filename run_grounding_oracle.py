"""
Grounding Oracle Runner — the make-or-break premise check.

For a sample of base questions, run all THREE retrieval methods on each of the
3 conditions (grounded / ablated / disconnect), score against gold, and report
whether the premise holds:

    Graph should WIN on grounded but LOSE on ablated/disconnect,
    even though the question text is identical across conditions.

Fairness: all three methods use the SAME generator (LLM_MODEL via Ollama) and
the vector arm uses the SAME embedder (EMBED_MODEL) as LightRAG. Only the
RETRIEVAL differs, so any score gap is attributable to retrieval, not generation.

Cost warning: the graph arm builds a LightRAG graph per corpus (~20 paragraphs
each). Start with MAX_BASES small (10-30). Graphs are cached by corpus hash, so
re-runs are cheap. Results are written incrementally and completed ids are
skipped on restart, so you can stop/resume.

USAGE:
    python run_grounding_oracle.py
    # in : data/grounding_pairs.jsonl
    # out: results/oracle_raw.jsonl + results/oracle_report.txt
"""

import re
import json
import time
import string
import random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import requests

from lightrag_core import query_async, OLLAMA_HOST, LLM_MODEL, EMBED_MODEL

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
MAX_BASES = 120  # base questions to evaluate (x3 conditions each)
SPLIT_FILTER = None  # None = any split; or "train"/"val"/"test"
TOP_K = 5  # chunks for the vector arm
LONG_CTX_CHAR_BUDGET = 24000  # ~6k tokens cap for long-context
GEN_MAX_TOKENS = 256

GROUNDING_PATH = Path("./data/grounding_pairs.jsonl")
OUT_DIR = Path("./results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_PATH = OUT_DIR / "oracle_raw.jsonl"
REPORT_PATH = OUT_DIR / "oracle_report.txt"

METHODS = ["vector", "graph", "long_context"]
COST_RANK = {"vector": 0, "graph": 1, "long_context": 2}
CONDITIONS = ["grounded", "ablated", "disconnect"]

random.seed(SEED)

ANSWER_PROMPT = (
    "Answer the question using ONLY the context below. "
    "Give just the short answer, no explanation.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)


# ---------------------------------------------------------------------------
# Ollama helpers (shared generator + embedder)
# ---------------------------------------------------------------------------
def ollama_generate(prompt, max_tokens=GEN_MAX_TOKENS):
    r = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.0},
        },
        timeout=300,
    )
    r.raise_for_status()
    return r.json().get("response", "").strip()


def ollama_embed_texts(texts):
    r = requests.post(
        f"{OLLAMA_HOST}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=180,
    )
    r.raise_for_status()
    return np.array(r.json()["embeddings"], dtype=float)


# ---------------------------------------------------------------------------
# Retrieval arms — all feed the SAME generator
# ---------------------------------------------------------------------------
def vector_answer(question, source_docs):
    chunks = [d for d in source_docs if d.strip()]
    if not chunks:
        return ""
    embs = ollama_embed_texts([question] + chunks)
    q, c = embs[0], embs[1:]
    qn = q / (np.linalg.norm(q) + 1e-9)
    cn = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-9)
    sims = cn @ qn
    idx = np.argsort(-sims)[:TOP_K]
    ctx = "\n\n".join(chunks[i] for i in idx)
    return ollama_generate(ANSWER_PROMPT.format(context=ctx, question=question))


def long_context_answer(question, source_docs):
    ctx = "\n\n".join(source_docs)[:LONG_CTX_CHAR_BUDGET]
    return ollama_generate(ANSWER_PROMPT.format(context=ctx, question=question))


async def graph_answer(question, source_docs):
    try:
        return (await query_async(source_docs, question, mode="local")) or ""
    except Exception as e:
        print(f"    graph error: {type(e).__name__}: {e}")
        return ""


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def normalize_answer(s):
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(pred, gold):
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def score_answer(pred, gold):
    golds = gold if isinstance(gold, list) else [gold]
    # MuSiQue answers are short; F1 handles them well. Also reward substring hits.
    best = 0.0
    for g in golds:
        g = str(g)
        f1 = f1_score(pred, g)
        contains = (
            1.0
            if normalize_answer(g) and normalize_answer(g) in normalize_answer(pred)
            else 0.0
        )
        best = max(best, f1, contains)
    return best


def pick_winner(scores):
    best = max(scores.values())
    tied = [m for m, s in scores.items() if abs(s - best) < 1e-9]
    return min(tied, key=lambda m: COST_RANK[m])


# ---------------------------------------------------------------------------
# Data / resume
# ---------------------------------------------------------------------------
def load_conditions():
    rows = []
    with open(GROUNDING_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if SPLIT_FILTER and r.get("split") != SPLIT_FILTER:
                continue
            rows.append(r)
    by_base = defaultdict(dict)
    for r in rows:
        by_base[r["base_id"]][r["condition"]] = r
    # keep only bases with all 3 conditions
    complete = {b: c for b, c in by_base.items() if set(CONDITIONS) <= set(c)}
    base_ids = sorted(complete.keys())
    random.shuffle(base_ids)
    base_ids = base_ids[:MAX_BASES]
    return [complete[b] for b in base_ids]


def already_done():
    done = set()
    if RAW_PATH.exists():
        with open(RAW_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    return done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def run():
    bases = load_conditions()
    done = already_done()
    print(
        f"Evaluating {len(bases)} base questions "
        f"({len(bases) * 3} conditions). Already done: {len(done)}"
    )

    fout = open(RAW_PATH, "a", encoding="utf-8")
    t_start = time.time()
    for bi, conds in enumerate(bases, 1):
        for cond in CONDITIONS:
            row = conds[cond]
            if row["id"] in done:
                continue
            q, gold, docs = row["question"], row["answer"], row["source_docs"]
            scores, times = {}, {}

            t = time.time()
            a_vec = vector_answer(q, docs)
            scores["vector"] = score_answer(a_vec, gold)
            times["vector"] = time.time() - t

            t = time.time()
            a_g = await graph_answer(q, docs)

            scores["graph"] = score_answer(a_g, gold)
            times["graph"] = time.time() - t

            t = time.time()
            a_lc = long_context_answer(q, docs)
            scores["long_context"] = score_answer(a_lc, gold)
            times["long_context"] = time.time() - t

            rec = {
                "id": row["id"],
                "base_id": row["base_id"],
                "condition": cond,
                "split": row.get("split"),
                "bridge_entity": row.get("bridge_entity"),
                "scores": scores,
                "winner": pick_winner(scores),
                "latency": times,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
        if bi % 5 == 0:
            elapsed = time.time() - t_start
            print(f"  {bi}/{len(bases)} bases  ({elapsed:.0f}s elapsed)")
    fout.close()
    analyze()


def analyze():
    results = [json.loads(l) for l in open(RAW_PATH, encoding="utf-8")]
    if not results:
        print("No results.")
        return

    mean = {c: {m: [] for m in METHODS} for c in CONDITIONS}
    wins = {c: Counter() for c in CONDITIONS}
    for r in results:
        c = r["condition"]
        if c not in CONDITIONS:
            continue
        for m in METHODS:
            mean[c][m].append(r["scores"][m])
        wins[c][r["winner"]] += 1
    means = {
        c: {m: (sum(v) / len(v) if v else 0.0) for m, v in mean[c].items()}
        for c in CONDITIONS
    }

    lines = [
        "=" * 64,
        "GROUNDING ORACLE REPORT",
        "=" * 64,
        f"Conditions scored: {len(results)}\n",
    ]
    lines.append("Mean answer score by condition x method:")
    lines.append(f"  {'condition':12s}" + "".join(f"{m:>14s}" for m in METHODS))
    for c in CONDITIONS:
        lines.append(f"  {c:12s}" + "".join(f"{means[c][m]:14.3f}" for m in METHODS))
    lines.append("\nWinner distribution by condition:")
    for c in CONDITIONS:
        tot = sum(wins[c].values()) or 1
        lines.append(
            f"  {c:12s} "
            + ", ".join(f"{m}:{wins[c][m]}({wins[c][m]/tot:.0%})" for m in METHODS)
        )

    g, a, d = (
        means["grounded"]["graph"],
        means["ablated"]["graph"],
        means["disconnect"]["graph"],
    )
    # graph vs best-non-graph on grounded
    best_non_graph_grounded = max(
        means["grounded"]["vector"], means["grounded"]["long_context"]
    )
    lines.append("\n" + "-" * 64)
    lines.append("PREMISE CHECK (the thesis):")
    lines.append(f"  graph mean: grounded={g:.3f}  ablated={a:.3f}  disconnect={d:.3f}")
    lines.append(f"  graph drop grounded->ablated   : {g - a:+.3f}")
    lines.append(f"  graph drop grounded->disconnect: {g - d:+.3f}")
    lines.append(
        f"  graph vs best-other on grounded: {g - best_non_graph_grounded:+.3f}"
    )
    holds = (g - a >= 0.05) and (g - d >= 0.05)
    if holds and g >= best_non_graph_grounded:
        lines.append(
            "  VERDICT: PREMISE HOLDS — graph helps when grounded, not when broken."
        )
        lines.append("           Proceed to feature extraction + router comparison.")
    elif holds:
        lines.append(
            "  VERDICT: PARTIAL — graph drops when broken (good), but doesn't lead"
        )
        lines.append(
            "           on grounded. Check generator/retrieval quality; may still work."
        )
    else:
        lines.append(
            "  VERDICT: WEAK/FAILS — graph doesn't clearly drop when the chain breaks."
        )
        lines.append(
            "           Investigate: graph too small? answers leaking via distractors?"
        )
        lines.append(
            "           Run validate_grounding_lightrag.py to check reachability first."
        )
    lines.append("=" * 64)

    report = "\n".join(lines)
    print("\n" + report)
    REPORT_PATH.write_text(report)


if __name__ == "__main__":
    import asyncio

    asyncio.run(run())
