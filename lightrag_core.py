"""
LightRAG core wrapper — shared by the whole grounding pipeline.

This replaces the spaCy co-occurrence graph with LightRAG's real LLM-extracted
knowledge graph. Crucially, LightRAG persists that graph as a GraphML file
(graph_chunk_entity_relation.graphml) in its working dir, which we load with
networkx. So graph_connectivity / subgraph_size / reachability are computed on
the SAME graph LightRAG actually traverses at query time.

Per-query design:
  Each MuSiQue condition has its own small corpus (its paragraphs). We build a
  separate LightRAG working dir per condition, keyed by a hash of the docs, so
  the graph reflects exactly that condition's corpus (grounded vs ablated vs
  disconnect produce different graphs — that's the whole point).

Verified against LightRAG core API (Ollama backend):
  from lightrag import LightRAG, QueryParam
  from lightrag.llm.ollama import ollama_model_complete, ollama_embed
  from lightrag.utils import EmbeddingFunc
  await rag.initialize_storages(); await initialize_pipeline_status()
  await rag.ainsert(text); await rag.aquery(q, QueryParam(mode="local"))

REQUIREMENTS
------------
    pip install "lightrag-hku" networkx numpy nano-vectordb
    ollama pull qwen2.5:7b          # extraction LLM (needs ~32k ctx ideally)
    ollama pull nomic-embed-text    # embeddings (dim 768)

NOTE: entity extraction quality depends heavily on the LLM. qwen2.5:7b or
llama3.1:8b work; very small models (≤3b) extract few entities and will weaken
your graph features. Use the largest model your machine can run.
"""

import os
import asyncio
import hashlib
from functools import partial
from pathlib import Path

import networkx as nx

# --- verified imports for current LightRAG ---
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LIGHTRAG_LLM", "qwen2.5:7b")
EMBED_MODEL = os.getenv("LIGHTRAG_EMBED", "nomic-embed-text")
EMBED_DIM = int(os.getenv("LIGHTRAG_EMBED_DIM", "768"))
NUM_CTX = int(os.getenv("LIGHTRAG_NUM_CTX", "16384"))

WORK_ROOT = Path("./lightrag_workdirs")
WORK_ROOT.mkdir(parents=True, exist_ok=True)

GRAPH_FILE = "graph_chunk_entity_relation.graphml"


def corpus_key(source_docs):
    """Stable id for a corpus so we can cache/reuse its working dir."""
    h = hashlib.sha1("\n\n".join(source_docs).encode("utf-8")).hexdigest()[:16]
    return h


def _make_rag(working_dir):
    embedding_func = EmbeddingFunc(
        embedding_dim=EMBED_DIM,  # 768
        max_token_size=8192,
        func=partial(
            ollama_embed.func,  # <-- .func = the UNWRAPPED raw function
            embed_model=EMBED_MODEL,
            host=OLLAMA_HOST,
        ),
    )
    return LightRAG(
        working_dir=str(working_dir),
        llm_model_func=ollama_model_complete,
        llm_model_name=LLM_MODEL,
        llm_model_kwargs={"host": OLLAMA_HOST, "options": {"num_ctx": NUM_CTX}},
        embedding_func=embedding_func,
    )


async def _init(rag):
    await rag.initialize_storages()
    await initialize_pipeline_status()


async def build_graph_async(source_docs, force=False):
    """Build (or reuse) a LightRAG graph for one corpus. Returns working_dir."""
    wd = WORK_ROOT / corpus_key(source_docs)
    graph_path = wd / GRAPH_FILE
    if graph_path.exists() and not force:
        return wd  # already indexed
    wd.mkdir(parents=True, exist_ok=True)
    rag = _make_rag(wd)
    await _init(rag)
    # insert each paragraph as its own document so chunk boundaries align
    for i, doc in enumerate(source_docs):
        if doc.strip():
            await rag.ainsert(doc, ids=[f"{corpus_key(source_docs)}-{i}"])
    return wd


async def query_async(source_docs, question, mode="local"):
    """Run a LightRAG graph query over a corpus. Returns the answer string."""
    wd = await build_graph_async(source_docs)
    rag = _make_rag(wd)
    await _init(rag)
    return await rag.aquery(question, param=QueryParam(mode=mode))


# ---------------------------------------------------------------------------
# Graph access for FEATURES (load LightRAG's persisted GraphML with networkx)
# ---------------------------------------------------------------------------
def load_graph(source_docs):
    """Load the LightRAG knowledge graph for a corpus as a networkx graph.

    Must be called after the graph has been built (build_graph_async). Returns
    an undirected nx.Graph. Node ids are entity names (often quoted/upper-cased
    by LightRAG); we normalize a lookup copy for matching.
    """
    wd = WORK_ROOT / corpus_key(source_docs)
    graph_path = wd / GRAPH_FILE
    if not graph_path.exists():
        return nx.Graph()
    g = nx.read_graphml(graph_path)
    return g.to_undirected()


def normalize_entity(name):
    return (name or "").strip().strip('"').strip("'").lower()


def node_lookup(graph):
    """Map normalized entity name -> actual node id."""
    return {normalize_entity(n): n for n in graph.nodes}


# ---------------------------------------------------------------------------
# Sync convenience wrappers
# ---------------------------------------------------------------------------
def build_graph(source_docs, force=False):
    return asyncio.run(build_graph_async(source_docs, force=force))


def query(source_docs, question, mode="local"):
    return asyncio.run(query_async(source_docs, question, mode=mode))


if __name__ == "__main__":
    # smoke test on a tiny corpus
    docs = [
        "Inception is a 2010 film directed by Christopher Nolan.",
        "Christopher Nolan is a filmmaker who was born in London in 1970.",
        "Paris is the capital of France.",
    ]
    print("Building graph (this calls Ollama; may take a minute)...")
    build_graph(docs)
    g = load_graph(docs)
    print(f"Graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    print("Sample nodes:", list(g.nodes)[:10])
    ans = query(docs, "Where was the director of Inception born?", mode="local")
    print("Answer:", ans)
