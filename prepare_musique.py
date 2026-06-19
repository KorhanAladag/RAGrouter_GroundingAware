"""
Prepare the MuSiQue base set for the grounding project.

This REPLACES the old multi-dataset mix_datasets.py. The grounding experiment
uses ONLY MuSiQue, because it ships:
  - question_decomposition : the individual hops (sub-question + answer)
  - per-hop supporting paragraph titles
which is exactly what we need to surgically break graph connections later.

What this does:
  1. Load MuSiQue (answerable / validation split).
  2. Keep only genuine multi-hop questions (>= MIN_HOPS) that have paragraphs.
  3. Normalize each question to a clean schema (paragraphs get explicit idx).
  4. Assign a train/val/test split AT THE BASE-QUESTION LEVEL, stratified by
     hop count. (Splitting per base_id is critical: all 3 grounding conditions
     of one question must land in the SAME split, or you leak.)

Output: data/musique_base.jsonl  (one row per multi-hop question)

USAGE:
    pip install datasets
    python prepare_musique.py
"""

import json
import random
from pathlib import Path
from collections import Counter, defaultdict

from datasets import load_dataset

SEED = 42
MIN_HOPS = 2
MAX_QUESTIONS = 1500  # cap; raise later if you want more
SPLIT = (0.70, 0.15, 0.15)  # train / val / test
OUT = Path("./data/musique_base.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)
random.seed(SEED)


def normalize_decomp(decomp):
    out = []
    for hop in decomp:
        out.append(
            {
                "question": hop.get("question", ""),
                "answer": hop.get("answer", ""),
                "paragraph_support_idx": hop.get("paragraph_support_idx"),
            }
        )
    return out


def normalize_paragraphs(paragraphs):
    out = []
    for idx, p in enumerate(paragraphs):
        out.append(
            {
                "idx": idx,
                "title": p.get("title", ""),
                "paragraph_text": p.get("paragraph_text", p.get("text", "")),
                "is_supporting": p.get("is_supporting", False),
            }
        )
    return out


def assign_splits(rows):
    """Stratified by hop count, assigned at the base level."""
    by_hops = defaultdict(list)
    for r in rows:
        by_hops[r["n_hops"]].append(r)
    train, val, test = [], [], []
    for hops, items in by_hops.items():
        random.shuffle(items)
        n = len(items)
        n_tr = int(n * SPLIT[0])
        n_va = int(n * SPLIT[1])
        for r in items[:n_tr]:
            r["split"] = "train"
            train.append(r)
        for r in items[n_tr : n_tr + n_va]:
            r["split"] = "val"
            val.append(r)
        for r in items[n_tr + n_va :]:
            r["split"] = "test"
            test.append(r)
    return train, val, test


def main():
    print("Loading MuSiQue (validation split)...")
    ds = load_dataset("dgslibisey/MuSiQue", split="validation")

    rows = []
    for row in ds:
        decomp = row.get("question_decomposition") or []
        paragraphs = row.get("paragraphs") or []
        if len(decomp) < MIN_HOPS or not paragraphs:
            continue
        rows.append(
            {
                "base_id": row.get("id"),
                "question": row["question"],
                "answer": row["answer"],
                "paragraphs": normalize_paragraphs(paragraphs),
                "decomposition": normalize_decomp(decomp),
                "n_hops": len(decomp),
            }
        )
        if len(rows) >= MAX_QUESTIONS:
            break

    train, val, test = assign_splits(rows)
    allrows = train + val + test
    random.shuffle(allrows)

    with open(OUT, "w", encoding="utf-8") as f:
        for r in allrows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(allrows)} multi-hop questions -> {OUT}")
    print("Split sizes :", dict(Counter(r["split"] for r in allrows)))
    print("Hop counts  :", dict(Counter(r["n_hops"] for r in allrows)))
    print(
        "Avg paragraphs/question:",
        round(sum(len(r["paragraphs"]) for r in allrows) / max(1, len(allrows)), 1),
    )


if __name__ == "__main__":
    main()
