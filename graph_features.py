"""
Graph grounding features — computed on LightRAG's real knowledge graph.

Replaces the spaCy co-occurrence approach. Every function here loads the
GraphML that LightRAG persisted for a corpus, so the features describe the
exact graph the system traverses.

The grounding signal (what makes our thesis work):
  - graph_connectivity : are the query's entities connected to the answer region?
  - subgraph_size      : how large is the k-hop neighborhood of query entities?
  - answer_reachable   : can we traverse query-entities -> answer? (validation)

Query entities: we ask the LLM-built graph which of its nodes the query mentions
by matching normalized entity strings. (We avoid spaCy entirely so the entity
space matches LightRAG's.)
"""

import re
import networkx as nx

from lightrag_core import (
    load_graph,
    node_lookup,
    normalize_entity,
)

GRAPH_HOPS = 2


def _candidate_query_terms(question):
    """Extract capitalized spans + quoted spans as candidate entity mentions.

    Lightweight, dependency-free. LightRAG node matching does the real work:
    a term only counts if it matches a graph node.
    """
    terms = set()
    # capitalized word sequences (e.g. "Christopher Nolan", "Inception")
    for m in re.findall(r"[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*", question):
        terms.add(m)
    # quoted spans
    for m in re.findall(r'"([^"]+)"', question):
        terms.add(m)
    return {t for t in terms if len(t) > 2}


def query_entity_nodes(graph, question):
    """Graph nodes that the query mentions (matched via normalized strings)."""
    lut = node_lookup(graph)
    hits = set()
    for term in _candidate_query_terms(question):
        nt = normalize_entity(term)
        if nt in lut:
            hits.add(lut[nt])
            continue
        # partial containment either direction
        for norm_name, node in lut.items():
            if nt and (nt in norm_name or norm_name in nt):
                hits.add(node)
    return hits


def answer_nodes(graph, answer):
    lut = node_lookup(graph)
    na = normalize_entity(answer if isinstance(answer, str) else " ".join(answer))
    if not na:
        return set()
    hits = set()
    if na in lut:
        hits.add(lut[na])
    for norm_name, node in lut.items():
        if na in norm_name or norm_name in na:
            hits.add(node)
    return hits


def answer_reachable(source_docs, question, answer):
    """True (path), False (no path), or None (answer entity absent / unanchorable)."""
    g = load_graph(source_docs)
    if g.number_of_nodes() == 0:
        return None
    q_nodes = query_entity_nodes(g, question)
    a_nodes = answer_nodes(g, answer)
    if not a_nodes or not q_nodes:
        return None
    for s in q_nodes:
        for d in a_nodes:
            if s in g and d in g and nx.has_path(g, s, d):
                return True
    return False


def grounding_features(source_docs, question):
    """Return graph_connectivity, subgraph_size, n_query_nodes."""
    g = load_graph(source_docs)
    if g.number_of_nodes() == 0:
        return {"graph_connectivity": 0.0, "subgraph_size": 0, "n_query_nodes": 0}

    seeds = query_entity_nodes(g, question)
    if not seeds:
        return {"graph_connectivity": 0.0, "subgraph_size": 0, "n_query_nodes": 0}

    # k-hop neighborhood size
    neighborhood = set(seeds)
    frontier = set(seeds)
    for _ in range(GRAPH_HOPS):
        nxt = set()
        for node in frontier:
            nxt |= set(g.neighbors(node))
        neighborhood |= nxt
        frontier = nxt
    subgraph_size = len(neighborhood)

    # connectivity: fraction of seed entities sharing the largest component
    if len(seeds) >= 2:
        comp_id = {
            n: i for i, comp in enumerate(nx.connected_components(g)) for n in comp
        }
        from collections import Counter

        groups = Counter(comp_id[s] for s in seeds if s in comp_id)
        graph_connectivity = (max(groups.values()) / len(seeds)) if groups else 0.0
    else:
        graph_connectivity = 1.0

    return {
        "graph_connectivity": graph_connectivity,
        "subgraph_size": subgraph_size,
        "n_query_nodes": len(seeds),
    }
