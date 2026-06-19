# RAGrouter_GroundingAware

**Grounding-aware adaptive retrieval routing for Graph RAG.** Query complexity is *necessary but not sufficient* for routing: a complex multi-hop query whose entities are disconnected in the knowledge graph gains nothing from graph traversal. We add query-conditional graph-grounding features that recover the routing decision a complexity-only router is structurally blind to.

---

## TL;DR

A router decides, per query, whether to use vector search, Graph RAG, or long-context. Existing routers escalate *complex* queries to Graph RAG. But Graph RAG only helps when the query's entities are actually **connected** in the graph. We show that:

- Graph RAG quality collapses from **0.46 (chain intact) → 0.00 (chain broken)**.
- Complexity and corpus-probe features are **identical / flat** across intact vs broken; only **grounding features** separate them.
- A grounding-aware router significantly beats a complexity-only one (ablated cost-adjusted gap **+0.101**, 95% bootstrap CI **[+0.032, +0.174]**, P(gap>0)=1.00).

---

## The idea in one picture

For each multi-hop question we build three conditions with **identical question text** but different corpora:

| condition | what we remove | effect on graph | best method |
|---|---|---|---|
| `grounded` | nothing | chain intact | graph often best |
| `ablated` | the bridge paragraph | hop 1 broken (answer often still findable) | vector |
| `disconnect` | the linking/answer paragraph | entity present, path stalls | vector (all low) |

A complexity-only router sees the same features for all three and must make the same choice. The grounding features (`graph_connectivity`, `subgraph_size`, `n_query_nodes`) differ sharply and supply the missing signal.

---

## Pipeline

Run the scripts in order. Each consumes the previous stage's output.

| # | script | output | purpose |
|---|---|---|---|
| 1 | `prepare_musique.py` | `data/musique_base.jsonl` | load MuSiQue, keep multi-hop, base-level train/val/test split |
| 2 | `build_grounding_pairs.py` | `data/grounding_pairs.jsonl` | build grounded / ablated / disconnect conditions |
| 3 | `lightrag_core.py`, `graph_features.py` | `lightrag_workdirs/` | build LightRAG graphs; compute grounding features |
| 4 | `run_grounding_oracle.py` | `results/oracle_raw.jsonl` | score vector / graph / long-context per condition |
| 5 | `extract_grounding_features.py` | `results/grounding_features.jsonl` | complexity + probe + grounding features + oracle label |
| 6 | `compare_routers.py` | `results/router_comparison.json` | complexity-only vs grounding-aware router |
| 7 | `significance_check.py` | (stdout) | base-level bootstrap CI on the gap |

---

## Setup

Requires Python 3.10+ and [Ollama](https://ollama.com) running locally.

```bash
# 1. virtual environment
python -m venv ksm_env
# Windows: ksm_env\Scripts\activate   |   macOS/Linux: source ksm_env/bin/activate

# 2. python deps
pip install -r requirements.txt

# 3. models (via Ollama)
ollama pull qwen2.5:7b          # generation + entity/relation extraction
ollama pull nomic-embed-text    # embeddings (768-dim)
```

Key configuration (env vars or edit `lightrag_core.py`):

| variable | default | notes |
|---|---|---|
| `LIGHTRAG_LLM` | `qwen2.5:7b` | generator + extractor |
| `LIGHTRAG_EMBED` | `nomic-embed-text` | embeddings |
| `LIGHTRAG_EMBED_DIM` | `768` | must match the embedder |
| `LIGHTRAG_NUM_CTX` | `16384` | context window; do not drop too low or graph queries truncate |

---

## Run it

```bash
python prepare_musique.py
python build_grounding_pairs.py

# oracle is the slow part (builds a graph per condition).
# Start small to smoke-test, then scale MAX_BASES in run_grounding_oracle.py.
python run_grounding_oracle.py          # resumable: skips already-scored conditions

python extract_grounding_features.py
python compare_routers.py
python significance_check.py
```

**Performance notes.** The oracle builds one LightRAG graph per condition (~20 paragraphs each), so it dominates runtime (~minutes per base on a local 7B model). Results are written incrementally and completed ids are skipped, so you can stop/resume freely. Ensure the model runs on GPU (`ollama ps` should show GPU); close other VRAM-heavy apps.

---

## Results (120 base questions, 360 conditions)

Mean answer quality by condition × method:

| condition | vector | graph | long-context |
|---|---|---|---|
| grounded | 0.424 | **0.458** | 0.430 |
| ablated | 0.345 | 0.003 | 0.348 |
| disconnect | 0.142 | 0.003 | 0.112 |

Router comparison (`cost_adj` = quality − 0.01·context-cost):

| router | cost_adj | ablated cost_adj |
|---|---|---|
| oracle (ceiling) | 0.386 | 0.414 |
| **grounding_aware** | **0.224** | **0.256** |
| complexity_only | 0.180 | 0.155 |
| always_vector | 0.254 | 0.295 |

Among learned routers, grounding-awareness wins on every cut; the ablated cost-adjusted gap is statistically significant. (`always_vector` remains a strong simple baseline on MuSiQue's small corpora — see the paper's limitations.)

---

## Repository layout

```
.
├── prepare_musique.py
├── build_grounding_pairs.py
├── lightrag_core.py
├── graph_features.py
├── run_grounding_oracle.py
├── extract_grounding_features.py
├── compare_routers.py
├── significance_check.py
├── paper/
│   ├── grounding_aware_routing_paper.md
│   └── figures/
├── data/        # generated (git-ignored)
├── results/     # generated
└── lightrag_workdirs/   # generated graphs (git-ignored)
```

---

## Method notes

- **Shared generator & embedder.** All three retrieval methods use the same LLM and embedder, so any score gap is attributable to *retrieval*, not generation.
- **No leakage.** Routers are evaluated with `GroupKFold` grouped by base question, so all three conditions of a question stay on the same side of every split. Answer-dependent signals (e.g. answer reachability) are used only to validate the manipulation, never as router features.
- **Cost-adjusted scoring.** Long-context wins raw quality only by reading the whole (small) corpus, which does not scale; we report quality and a cost-adjusted score separately.

---

## Limitations

Single benchmark (MuSiQue), local 7B model (bounds absolute quality and graph density), and a simple vector baseline that stays competitive on small corpora. The contribution is the *necessary-but-not-sufficient* finding and the grounding features, not state-of-the-art answer quality. See the paper for full discussion.

---

## Citation

If you use this work, please cite the paper (details in `paper/`). Bibliographic entries for related work and datasets must be verified before formal publication.

## License

MIT (or your chosen license — add a `LICENSE` file).