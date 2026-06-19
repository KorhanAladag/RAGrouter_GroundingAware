"""
Significance check for the router comparison.

Puts a confidence interval on the grounding_aware MINUS complexity_only gap,
using a base-level bootstrap: we resample whole BASE QUESTIONS with replacement
(not individual conditions), because the 3 conditions of a base are not
independent. Router predictions are the fixed out-of-fold predictions from
compare_routers; the bootstrap captures evaluation-sample uncertainty, which is
the dominant uncertainty at this small N.

Reports, for three metrics (ablated cost-adjusted, broken quality, overall
cost-adjusted):
  - observed gap
  - 95% bootstrap CI
  - bootstrap support P(gap > 0)  (fraction of resamples favoring grounding_aware)
  - per-base win/tie/loss count (intuitive paired view)

A gap whose 95% CI excludes 0 is solid; one that straddles 0 means "promising
but needs the larger run". Either way you learn whether to trust +0.133.

USAGE:
    python significance_check.py     # after compare_routers.py
"""

import random
from collections import defaultdict

import numpy as np

from compare_routers import (
    load_jsonl,
    matrix,
    oof_predictions,
    method_cost,
    load_npara,
    COMPLEXITY,
    PROBE,
    GROUNDING,
    COST_LAMBDA,
    DEFAULT_NPARA,
    FEATURES_PATH,
)

B = 5000
SEED = 42


def realized_metric(rows, preds, idxs, npara, kind):
    if not idxs:
        return 0.0
    q = np.mean([rows[i]["scores"][preds[i]] for i in idxs])
    if kind == "quality":
        return float(q)
    cost = np.mean(
        [method_cost(preds[i], npara.get(rows[i]["id"], DEFAULT_NPARA)) for i in idxs]
    )
    return float(q - COST_LAMBDA * cost)


def bootstrap(rows, preds_a, preds_b, npara, conds, kind, b=B, seed=SEED):
    by_base = defaultdict(list)
    for i, r in enumerate(rows):
        if conds is None or r["condition"] in conds:
            by_base[r["base_id"]].append(i)
    bases = list(by_base.keys())
    all_idx = [i for bb in bases for i in by_base[bb]]

    def gap(idxs):
        return realized_metric(rows, preds_a, idxs, npara, kind) - realized_metric(
            rows, preds_b, idxs, npara, kind
        )

    observed = gap(all_idx)
    rng = random.Random(seed)
    diffs = []
    for _ in range(b):
        sampled = [rng.choice(bases) for _ in bases]
        idxs = [i for bb in sampled for i in by_base[bb]]
        diffs.append(gap(idxs))
    diffs.sort()
    lo = diffs[int(0.025 * b)]
    hi = diffs[int(0.975 * b)]
    p_pos = float(np.mean([d > 0 for d in diffs]))

    # per-base paired view (only meaningful for single-condition subsets like ablated)
    wins = ties = losses = 0
    for bb in bases:
        idxs = by_base[bb]
        d = gap(idxs)
        if d > 1e-9:
            wins += 1
        elif d < -1e-9:
            losses += 1
        else:
            ties += 1
    return observed, lo, hi, p_pos, (wins, ties, losses)


def main():
    rows = load_jsonl(FEATURES_PATH)
    y = np.array([r["label"] for r in rows], dtype=object)
    groups = np.array([r["base_id"] for r in rows])
    npara = load_npara()

    preds_ga = oof_predictions(matrix(rows, COMPLEXITY + PROBE + GROUNDING), y, groups)
    preds_co = oof_predictions(matrix(rows, COMPLEXITY), y, groups)

    print("=" * 72)
    print("SIGNIFICANCE: grounding_aware  -  complexity_only")
    print(
        f"base-level bootstrap, B={B}, {len(set(groups))} bases, {len(rows)} conditions"
    )
    print("=" * 72)
    tests = [
        ("ablated cost-adjusted", ["ablated"], "cost_adj"),
        ("broken quality", ["ablated", "disconnect"], "quality"),
        ("overall cost-adjusted", None, "cost_adj"),
    ]
    for name, conds, kind in tests:
        obs, lo, hi, p, (w, t, l) = bootstrap(
            rows, preds_ga, preds_co, npara, conds, kind
        )
        excludes0 = "YES" if (lo > 0 or hi < 0) else "no"
        print(f"\n{name}:")
        print(f"  observed gap : {obs:+.3f}")
        print(f"  95% CI       : [{lo:+.3f}, {hi:+.3f}]   excludes 0? {excludes0}")
        print(f"  P(gap>0)     : {p:.2f}")
        print(f"  per-base     : {w} win / {t} tie / {l} loss")
    print("\n" + "=" * 72)
    print("Read: CI excluding 0 = solid. CI straddling 0 = promising, needs larger N.")
    print("=" * 72)


if __name__ == "__main__":
    main()
