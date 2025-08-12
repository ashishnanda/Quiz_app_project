"""
offline_node2vec_pipeline.py

End-to-end OFFLINE pipeline:

1) Load nodes & edges from SQL (any column casing) and normalize column names to lowercase.
2) Build an undirected, weighted NetworkX graph.
3) Train Node2Vec embeddings (via the `node2vec` package).
4) Prepare a fast cosine KNN search (pure NumPy) over **Clients only**.
5) Annotate a DataFrame of input ids with:
   - best_client_id
   - similarity
   - path_to_best_client (compact vertex/edge payload, JSON by default)

Assumptions (adjust as needed)
------------------------------
- NODES table has: id, label (optional), entity_type (e.g., 'Client', 'Prospect', ...)
- EDGES table has: source, target, weight (optional; default 1.0)
- Graph is **undirected** (random source/target order is fine).

Dependencies
------------
pip install sqlalchemy pandas networkx node2vec numpy

Quick Start
-----------
from sqlalchemy import create_engine
from offline_node2vec_pipeline import (
    SQLSource, load_nodes, load_edges, build_graph_from_frames,
    train_node2vec, prepare_client_matrix, annotate_best_fit_clients_offline
)

engine = create_engine("postgresql+psycopg2://user:pass@host/db")

nodes_src = SQLSource("nodes", "graph", engine)
edges_src = SQLSource("edges", "graph", engine)

nodes_df = load_nodes(nodes_src)     # columns -> ['id','label','entity_type']
edges_df = load_edges(edges_src)     # columns -> ['source','target','weight']

G = build_graph_from_frames(nodes_df, edges_df, use_weights=True, directed=False)

embeddings = train_node2vec(G, dimensions=128, walk_length=30, num_walks=10)

# Select client ids (entity_type == 'client', case-insensitive)
client_ids = nodes_df.loc[nodes_df["entity_type"].str.lower().eq("client"), "id"].tolist()

# Prep cosine search over client vectors (fast pure-NumPy)
client_id_list, client_matrix = prepare_client_matrix(embeddings, client_ids)

# Annotate a DataFrame of inputs
import pandas as pd
df = pd.DataFrame({"input_id": ["X123", "Y456"]})
out = annotate_best_fit_clients_offline(
    df=df,
    input_col="input_id",
    embeddings=embeddings,
    client_id_list=client_id_list,
    client_matrix=client_matrix,
    G=G,
    weighted_path=False,   # set True to use Dijkstra on 'weight'; else BFS with cutoff
    max_hops=6,
    path_as_json=True
)
print(out.head())

Notes
-----
- For **very large** client catalogs, consider swapping the pure-NumPy KNN for FAISS/HNSW.
- Path computation uses NetworkX:
  • weighted_path=True  -> Dijkstra shortest path by edge weight (no cutoff)
  • weighted_path=False -> Unweighted BFS with cutoff=max_hops (fast, length-bounded)
"""

from __future__ import annotations

from typing import Optional, Dict, Any, Iterable, List, Tuple
import json
import numpy as np
import pandas as pd
import networkx as nx
from sqlalchemy.engine import Engine

# node2vec package: https://pypi.org/project/node2vec/
try:
    from node2vec import Node2Vec
except ImportError as e:
    raise ImportError(
        "Missing dependency 'node2vec'. Install it with:\n\n"
        "    pip install node2vec\n"
    ) from e


# =============================================================================
# SQL loader
# =============================================================================

class SQLSource:
    """
    Minimal wrapper around a SQL table. Automatically normalizes column names
    to lowercase so downstream code can rely on stable names.

    Parameters
    ----------
    table_name : str
        Table name (e.g., 'edges').
    schema_name : str
        Schema name (e.g., 'public', 'graph').
    engine : sqlalchemy.engine.Engine
        SQLAlchemy Engine connected to your database.

    Methods
    -------
    load(columns: Optional[list] = None) -> pd.DataFrame
        SELECT the given columns (or '*') and return a DataFrame with lowercase
        column names.
    """
    def __init__(self, table_name: str, schema_name: str, engine: Engine):
        self.table_name = table_name
        self.schema_name = schema_name
        self.engine = engine

    def load(self, columns: Optional[list] = None) -> pd.DataFrame:
        cols_sql = "*" if columns is None else ", ".join([f'"{c}"' for c in columns])
        query = f'SELECT {cols_sql} FROM "{self.schema_name}"."{self.table_name}"'
        df = pd.read_sql(query, self.engine)
        # Normalize column names to lowercase so casing in SQL doesn't matter
        df.columns = [c.lower() for c in df.columns]
        return df


def load_nodes(nodes_src: SQLSource) -> pd.DataFrame:
    """
    Load nodes with flexible casing from SQL; ensure lowercase standard columns.

    Returns
    -------
    pd.DataFrame with columns ['id','label','entity_type']
    """
    df = nodes_src.load(columns=["id", "label", "entity_type"])
    if "id" not in df.columns:
        raise ValueError("Nodes table must include an 'id' column (case-insensitive).")
    if "label" not in df.columns:
        df["label"] = ""
    if "entity_type" not in df.columns:
        df["entity_type"] = ""
    df["id"] = df["id"].astype(str)
    return df[["id", "label", "entity_type"]]


def load_edges(edges_src: SQLSource) -> pd.DataFrame:
    """
    Load edges with flexible casing from SQL; ensure lowercase standard columns.

    Returns
    -------
    pd.DataFrame with columns ['source','target','weight']
    """
    df = edges_src.load(columns=["source", "target", "weight"])
    for col in ("source", "target"):
        if col not in df.columns:
            raise ValueError("Edges table must include 'source' and 'target' (case-insensitive).")
    df["source"] = df["source"].astype(str)
    df["target"] = df["target"].astype(str)

    # Coerce weight -> float, default to 1.0
    if "weight" not in df.columns:
        df["weight"] = 1.0
    else:
        def _coerce_w(x):
            try:
                return float(x)
            except Exception:
                return 1.0
        df["weight"] = df["weight"].apply(_coerce_w)

    # Drop self-loops and duplicate rows
    df = df[df["source"] != df["target"]].copy()
    df = df.drop_duplicates(subset=["source", "target", "weight"])
    return df[["source", "target", "weight"]]


# =============================================================================
# Graph construction
# =============================================================================

def build_graph_from_frames(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    use_weights: bool = True,
    directed: bool = False,
) -> nx.Graph:
    """
    Build a weighted (or unweighted) NetworkX graph from DataFrames.

    Parameters
    ----------
    nodes_df : DataFrame with ['id','label','entity_type']
    edges_df : DataFrame with ['source','target','weight']
    use_weights : if False, all edge weights are set to 1.0
    directed : if True, builds nx.DiGraph; else nx.Graph (undirected)

    Returns
    -------
    nx.Graph (or nx.DiGraph)
    """
    G = nx.DiGraph() if directed else nx.Graph()

    # Add nodes with attributes
    for _, r in nodes_df.iterrows():
        G.add_node(
            r["id"],
            label=r.get("label", ""),
            entity_type=r.get("entity_type", "")
        )

    # Add edges; if duplicate pair appears, sum their weights (accumulate signal)
    for _, r in edges_df.iterrows():
        u, v = r["source"], r["target"]
        w = float(r.get("weight", 1.0)) if use_weights else 1.0
        if G.has_edge(u, v):
            G[u][v]["weight"] = float(G[u][v].get("weight", 1.0)) + w
        else:
            G.add_edge(u, v, weight=w)

    return G


# =============================================================================
# Node2Vec training
# =============================================================================

def train_node2vec(
    G: nx.Graph,
    *,
    dimensions: int = 128,
    walk_length: int = 30,
    num_walks: int = 10,
    p: float = 1.0,
    q: float = 1.0,
    window: int = 5,
    min_count: int = 1,
    epochs: int = 3,
    workers: int = 4,
    seed: int = 42,
    use_weights: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Train Node2Vec embeddings using the 'node2vec' package.

    Returns
    -------
    dict: {node_id: np.ndarray of length `dimensions`}
    """
    np.random.seed(seed)

    n2v = Node2Vec(
        graph=G,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        weight_key=("weight" if use_weights else None),
        workers=workers,
        seed=seed,
        quiet=True,
    )
    model = n2v.fit(
        window=window,
        min_count=min_count,
        batch_words=1024,
        epochs=epochs,
    )

    embeddings: Dict[str, np.ndarray] = {}
    for node in G.nodes():
        if node in model.wv:
            embeddings[node] = np.asarray(model.wv[node])
    return embeddings


# =============================================================================
# Cosine KNN over Clients (pure NumPy, fast)
# =============================================================================

def _l2_normalize_matrix(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return mat / norms


def _l2_normalize_vector(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    return vec if n == 0.0 else (vec / n)


def prepare_client_matrix(
    embeddings: Dict[str, np.ndarray],
    client_ids: Iterable[str],
) -> Tuple[List[str], np.ndarray]:
    """
    Precompute a Clients-only normalized matrix for fast cosine search.

    Returns
    -------
    client_id_list : list[str]
        Ordered ids aligned to rows of the matrix.
    client_matrix  : np.ndarray, shape (n_clients, d)
        L2-normalized embedding matrix (cosine == dot with normalized vectors).
    """
    ids: List[str] = []
    vecs: List[np.ndarray] = []
    for cid in client_ids:
        v = embeddings.get(cid)
        if v is not None:
            ids.append(cid)
            vecs.append(v)
    if not vecs:
        return [], np.zeros((0, 1), dtype=float)
    M = np.vstack(vecs).astype(float)
    M = _l2_normalize_matrix(M)
    return ids, M


def search_clients_top1_numpy(
    vec: np.ndarray,
    client_id_list: List[str],
    client_matrix: np.ndarray,
) -> Tuple[str, float]:
    """
    Return (best_client_id, cosine_similarity) using a precomputed Clients matrix.
    """
    if client_matrix.size == 0:
        return "", 0.0
    v = _l2_normalize_vector(vec.astype(float))
    sims = client_matrix.dot(v)  # cosine because both sides are L2-normalized
    idx = int(np.argmax(sims))
    return client_id_list[idx], float(sims[idx])


# =============================================================================
# Path extraction (NetworkX)
# =============================================================================

def find_path_nodes(
    G: nx.Graph,
    src: str,
    dst: str,
    *,
    weighted: bool = False,
    max_hops: int = 6,
) -> List[str]:
    """
    Return a node sequence [src,...,dst] if reachable, else [].

    - weighted=False: uses BFS with cutoff=max_hops (fast, bounded length).
    - weighted=True : uses Dijkstra by 'weight' (no cutoff).
    """
    try:
        if weighted:
            # Weighted shortest path by edge 'weight' (can be longer than max_hops)
            return nx.dijkstra_path(G, src, dst, weight="weight")
        else:
            # Unweighted, bounded by max_hops (edges). Fast for large graphs.
            paths_dict = nx.single_source_shortest_path(G, src, cutoff=max_hops)
            return paths_dict.get(dst, [])
    except Exception:
        return []


def path_nodes_to_objects(G: nx.Graph, node_path: List[str]) -> List[Dict[str, Any]]:
    """
    Convert a node sequence to a compact alternating V,E,V,... payload:
        Vertex: {'id','label','type'}
        Edge  : {'label','weight'}
    """
    if not node_path:
        return []
    out: List[Dict[str, Any]] = []
    # First vertex
    v0 = node_path[0]
    out.append({
        "id": v0,
        "label": G.nodes[v0].get("label", ""),
        "type": G.nodes[v0].get("entity_type", ""),
    })
    # Edges and subsequent vertices
    for u, v in zip(node_path[:-1], node_path[1:]):
        edata = G.get_edge_data(u, v, default={})
        out.append({
            "label": edata.get("label", ""),           # label only if you set it when adding edges
            "weight": float(edata.get("weight", 1.0)),
        })
        out.append({
            "id": v,
            "label": G.nodes[v].get("label", ""),
            "type": G.nodes[v].get("entity_type", ""),
        })
    return out


# =============================================================================
# Final: annotate a DataFrame with best client + path (OFFLINE)
# =============================================================================

def annotate_best_fit_clients_offline(
    df: pd.DataFrame,
    input_col: str,
    embeddings: Dict[str, np.ndarray],
    client_id_list: List[str],
    client_matrix: np.ndarray,
    G: nx.Graph,
    *,
    weighted_path: bool = False,
    max_hops: int = 6,
    path_as_json: bool = True,
) -> pd.DataFrame:
    """
    For each row in `df`, use Node2Vec embeddings to pick the most similar Client
    (cosine over a precomputed Clients-only matrix), then compute a short path
    from the input node to that Client inside the offline NetworkX graph.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain column `input_col` with graph node ids (strings).
    input_col : str
        Column in df that holds the input node id.
    embeddings : Dict[str, np.ndarray]
        Node id -> embedding vector.
    client_id_list : List[str]
        Ordered list of client ids aligned to `client_matrix` rows.
    client_matrix : np.ndarray
        L2-normalized matrix of client vectors (shape: n_clients x d).
    G : nx.Graph
        Offline graph for path extraction.
    weighted_path : bool, default False
        If True, use Dijkstra shortest path by 'weight' (no cutoff).
        If False, use unweighted BFS with cutoff=max_hops (fast, bounded length).
    max_hops : int, default 6
        BFS path cutoff when `weighted_path=False`.
    path_as_json : bool, default True
        If True, serialize compact path payload as JSON; else keep Python list.

    Returns
    -------
    pd.DataFrame
        Copy of df with three new columns:
        - 'best_client_id' (str)
        - 'similarity'     (float, cosine)
        - 'path_to_best_client' (JSON string or list of alternating V/E/V dicts)
    """
    if input_col not in df.columns:
        raise KeyError(f"Input column '{input_col}' not in DataFrame.")

    best_ids: List[str] = []
    sims: List[float] = []
    paths: List[Any] = []

    for _, row in df.iterrows():
        nid = str(row[input_col])

        v = embeddings.get(nid)
        if v is None:
            # No embedding for this node — fill safe defaults
            best_ids.append("")
            sims.append(0.0)
            paths.append("[]" if path_as_json else [])
            continue

        # Cosine top-1 over clients (fast matrix dot)
        best_client_id, sim = search_clients_top1_numpy(v, client_id_list, client_matrix)

        # Path extraction inside offline graph
        compact_payload = []
        if best_client_id:
            node_path = find_path_nodes(
                G, nid, best_client_id, weighted=weighted_path, max_hops=max_hops
            )
            compact_payload = path_nodes_to_objects(G, node_path)

        best_ids.append(best_client_id)
        sims.append(sim)
        paths.append(json.dumps(compact_payload) if path_as_json else compact_payload)

    out = df.copy()
    out["best_client_id"] = best_ids
    out["similarity"] = sims
    out["path_to_best_client"] = paths
    return out


# =============================================================================
# Example runnable (adjust connection string + names)
# =============================================================================

if __name__ == "__main__":
    from sqlalchemy import create_engine

    # 1) Connect to SQL
    engine = create_engine("postgresql+psycopg2://user:pass@host/dbname")

    # 2) Read tables (any column casing is accepted; normalized to lowercase)
    nodes_src = SQLSource(table_name="nodes",  schema_name="graph", engine=engine)
    edges_src = SQLSource(table_name="edges",  schema_name="graph", engine=engine)
    nodes_df = load_nodes(nodes_src)
    edges_df = load_edges(edges_src)
    print(f"Loaded {len(nodes_df)} nodes, {len(edges_df)} edges.")

    # 3) Build graph
    G = build_graph_from_frames(nodes_df, edges_df, use_weights=True, directed=False)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    # 4) Train Node2Vec
    embeddings = train_node2vec(
        G,
        dimensions=128,
        walk_length=30,
        num_walks=10,
        p=1.0, q=1.0,
        window=5, min_count=1, epochs=3,
        workers=4, seed=42,
        use_weights=True
    )
    print(f"Embeddings learned for {len(embeddings)} nodes.")

    # 5) Prepare Clients-only cosine search
    client_ids = nodes_df.loc[nodes_df["entity_type"].str.lower().eq("client"), "id"].tolist()
    client_id_list, client_matrix = prepare_client_matrix(embeddings, client_ids)
    print(f"Prepared client matrix: {len(client_id_list)} clients.")

    # 6) Annotate a sample DataFrame
    sample_df = pd.DataFrame({"input_id": nodes_df["id"].head(5).tolist()})
    result_df = annotate_best_fit_clients_offline(
        df=sample_df,
        input_col="input_id",
        embeddings=embeddings,
        client_id_list=client_id_list,
        client_matrix=client_matrix,
        G=G,
        weighted_path=False,  # set True to use Dijkstra by edge weight
        max_hops=6,
        path_as_json=True
    )
    print(result_df)