"""
Validate grounding pairs on LightRAG's REAL knowledge graph (Stage 1.5).

Same purpose as before, but reachability is checked on the graph LightRAG
actually builds and traverses — not a spaCy co-occurrence approximation.

For each base question we build the LightRAG graph for all three conditions,
then check answer-reachability:
  grounded   -> should be True  (chain intact)
  ablated    -> should be broken (False/None)
  disconnect -> should be broken (False/None)
Base questions that don't show this pattern are dropped (and counted), giving a
clean, defensible dataset.

WARNING: this builds a LightRAG graph per condition via Ollama. For N base
questions that is 3N graph builds — slow. Start with MAX_BASES small (e.g. 50).

USAGE:
    python validate_grounding_lightrag.py
    # in:  data/grounding_pairs.jsonl
    # out: data/grounding_pairs_clean.jsonl + data/validation_report.txt
"""

import json
import asyncio
from collections import defaultdict, Counter
from pathlib import Path

from lightrag_core import build_graph_async
from graph_features import answer_reachable

IN_PATH = Path("./data/grounding_pairs.jsonl")
OUT_PATH = Path("./data/grounding_pairs_clean.jsonl")
REPORT_PATH = Path("./data/validation_report.txt")
MAX_BASES = 50          # cap for a first run; raise once it works


def is_broken(r):
    return r is False or r is None


async def status_for(row):
    # build the graph first (async), then read reachability (sync, from GraphML)
    await build_graph_async(row["source_docs"])
    r = answer_reachable(row["source_docs"], row["question"], row["answer"])
    cond = row["condition"]
    ok = (r is True) if cond == "grounded" else is_broken(r)
    return r, ok


async def main():
    if not IN_PATH.exists():
        raise FileNotFoundError("Run build_grounding_pairs.py first.")

    by_base = defaultdict(dict)
    with open(IN_PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            by_base[row["base_id"]][row["condition"]] = row

    kept_rows = []
    drop_reasons = Counter()
    reach_log = defaultdict(Counter)
    n_bases = kept_bases = 0

    for base_id, conds in by_base.items():
        if kept_bases >= MAX_BASES and n_bases >= MAX_BASES:
            break
        n_bases += 1
        if not {"grounded", "ablated", "disconnect"} <= set(conds):
            drop_reasons["missing_condition"] += 1
            continue

        statuses, ok_all = {}, True
        for cond in ("grounded", "ablated", "disconnect"):
            r, ok = await status_for(conds[cond])
            statuses[cond] = (r, ok)
            reach_log[cond][str(r)] += 1
            ok_all = ok_all and ok

        if not ok_all:
            if not statuses["grounded"][1]:
                drop_reasons["grounded_not_reachable"] += 1
            else:
                drop_reasons["ablation_leaked"] += 1
            continue

        kept_bases += 1
        for cond in ("grounded", "ablated", "disconnect"):
            row = conds[cond]
            row["reachable"] = statuses[cond][0]
            kept_rows.append(row)

        if n_bases % 10 == 0:
            print(f"  processed {n_bases} bases, kept {kept_bases}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for row in kept_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "=" * 60, "GROUNDING VALIDATION REPORT (LightRAG graph)", "=" * 60,
        f"Base questions processed : {n_bases}",
        f"Base questions kept      : {kept_bases} ({kept_bases / max(1, n_bases):.1%})",
        f"Clean rows written       : {len(kept_rows)}  -> {OUT_PATH}",
        "\nDrop reasons:",
    ]
    for k, v in drop_reasons.most_common():
        lines.append(f"  {k:24s}: {v}")
    lines.append("\nReachability by condition (True=intact, False/None=broken):")
    for cond in ("grounded", "ablated", "disconnect"):
        c = reach_log[cond]
        lines.append(f"  {cond:11s}: " +
                     ", ".join(f"{k}={c[k]}" for k in ("True", "False", "None")))
    lines.append("=" * 60)
    report = "\n".join(lines)
    print(report)
    REPORT_PATH.write_text(report)


if __name__ == "__main__":
    asyncio.run(main())
