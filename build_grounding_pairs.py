"""
Build grounding contrast pairs from the prepared MuSiQue base.

This is the connection-BREAKING manipulation. Reads data/musique_base.jsonl
(from prepare_musique.py) and, for each multi-hop question, emits up to 3
conditions with the SAME question text and the SAME split:

  GROUNDED   : full context. The reasoning chain is intact.
               -> the graph CAN traverse query-entities -> answer. graph useful.

  ABLATED    : remove the BRIDGE hop's supporting paragraph (the paragraph that
               links the first entity to the bridge entity). The first hop can
               no longer be made. -> chain broken. graph NOT useful.

  DISCONNECT : keep the bridge paragraph but remove the OTHER supporting
               paragraph(s) (the later hop's link to the answer). The bridge
               entity is still present as a node, but cannot reach the answer.
               -> graph traversal stalls. graph NOT useful.

The "bridge" entity = an intermediate hop's answer that a LATER sub-question
references (MuSiQue writes references like '#1', '#2' inside sub-questions).
Breaking the chain on EITHER side gives a complex query whose graph is useless,
while the question text stays identical -- that's the controlled conflict that a
complexity-only router cannot see but our grounding features can.

Output: data/grounding_pairs.jsonl

USAGE:
    python build_grounding_pairs.py      # after prepare_musique.py
"""

import re
import json
from pathlib import Path
from collections import defaultdict

IN = Path("./data/musique_base.jsonl")
OUT = Path("./data/grounding_pairs.jsonl")


def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def find_bridge_hop(decomp):
    """Index of the hop whose answer is referenced (#i) by a later sub-question."""
    for i, hop in enumerate(decomp):
        ref = f"#{i + 1}"
        for j, later in enumerate(decomp):
            if j <= i:
                continue
            if ref in (later.get("question") or ""):
                return i
    return None


def paragraph_idx_for_hop(hop, paragraphs):
    pi = hop.get("paragraph_support_idx")
    if pi is None:
        return None
    # confirm the index actually exists in this paragraph list
    valid = {p["idx"] for p in paragraphs}
    return pi if pi in valid else None


def docs_from(paragraphs, drop_idx=()):
    drop = set(drop_idx)
    return [
        f"{p['title']}: {p['paragraph_text']}".strip()
        for p in paragraphs
        if p["idx"] not in drop
    ]


def titles_for(paragraphs, idxs):
    s = set(idxs)
    return [p["title"] for p in paragraphs if p["idx"] in s]


def main():
    if not IN.exists():
        raise FileNotFoundError(
            "Run prepare_musique.py first (need musique_base.jsonl)."
        )

    out = []
    stats = defaultdict(int)
    n_seen = n_kept = 0

    with open(IN, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            n_seen += 1
            decomp = row["decomposition"]
            paragraphs = row["paragraphs"]
            split = row["split"]

            bridge_i = find_bridge_hop(decomp)
            if bridge_i is None:
                stats["no_bridge"] += 1
                continue
            bridge_hop = decomp[bridge_i]
            bridge_entity = bridge_hop.get("answer", "")
            bridge_idx = paragraph_idx_for_hop(bridge_hop, paragraphs)
            if bridge_idx is None:
                stats["bridge_para_not_found"] += 1
                continue

            support = set()
            for hop in decomp:
                pi = paragraph_idx_for_hop(hop, paragraphs)
                if pi is not None:
                    support.add(pi)
            other_support = support - {bridge_idx}
            if not other_support:
                stats["no_other_support"] += 1
                continue

            base_id = row["base_id"]
            q = row["question"]
            a = row["answer"]
            n_para = len(paragraphs)

            def make(condition, drop, expected, removed_idx):
                return {
                    "id": f"{base_id}__{condition}",
                    "base_id": base_id,
                    "condition": condition,
                    "split": split,
                    "question": q,
                    "answer": a,
                    "source_docs": docs_from(paragraphs, drop),
                    "bridge_entity": bridge_entity,
                    "removed_titles": titles_for(paragraphs, removed_idx),
                    "n_paragraphs": n_para - len(removed_idx),
                    "expected_graph_useful": expected,
                }

            out.append(make("grounded", (), True, set()))
            out.append(make("ablated", {bridge_idx}, False, {bridge_idx}))
            out.append(make("disconnect", other_support, False, other_support))
            n_kept += 1

    with open(OUT, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    print(f"Base questions read   : {n_seen}")
    print(f"Base questions kept   : {n_kept}")
    print(f"Conditions written    : {len(out)}  -> {OUT}")
    print(f"Dropped (reasons)     : {dict(stats)}")
    by_cond = defaultdict(int)
    by_split = defaultdict(int)
    for o in out:
        by_cond[o["condition"]] += 1
        by_split[o["split"]] += 1
    print(f"Per condition         : {dict(by_cond)}")
    print(f"Per split             : {dict(by_split)}")
    print()
    print("Each base_id has grounded/ablated/disconnect rows with identical")
    print("question text and the same split -- the controlled conflict is ready.")


if __name__ == "__main__":
    main()
