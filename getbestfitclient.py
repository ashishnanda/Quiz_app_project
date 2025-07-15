from typing import List, Dict
import pandas as pd
import time
from functools import reduce
import operator

def find_best_fit_clients(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    max_hops: int = 4,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Identify the best-fit Client for each Prospect via multi-hop traversal,
    using a single undirected-edges merge per hop for maximum speed.

    Parameters
    ----------
    nodes_df : pd.DataFrame
        DataFrame of all nodes with columns:
          - 'ID' (str): node identifier, e.g. 'p1', 'c42', 'a7'
          - 'entity_type' (str): one of 'Prospect', 'Client', 'UBS Financial Advisor', etc.
    edges_df : pd.DataFrame
        DataFrame of all edges with columns:
          - 'source' (str), 'target' (str): node IDs
          - 'weight' (float): in (0,1]
          - 'edge_detail' (str): description of the relationship
    max_hops : int, default=4
        Maximum number of hops to traverse from each Prospect
    verbose : bool, default=True
        If True, print timing and status updates at each major step

    Returns
    -------
    pd.DataFrame
        Columns:
          - 'prospect_id' (str)
          - 'client_id'   (str)
          - 'total_score' (float): sum of product-of-weights over all paths
          - 'path_details_dict' (Dict[str,str]):
                mapping each full-path string "n1→n2→…→nK" →
                concatenated edge_detail string "d12|…|d(K−1)K"

    Examples
    --------
    >>> nodes = pd.DataFrame([
    ...     {"ID":"p1","entity_type":"Prospect"},
    ...     {"ID":"a1","entity_type":"Other"},
    ...     {"ID":"c1","entity_type":"Client"}])
    >>> edges = pd.DataFrame([
    ...     {"source":"p1","target":"a1","weight":0.5,"edge_detail":"pd1"},
    ...     {"source":"a1","target":"c1","weight":0.8,"edge_detail":"ac1"}])
    >>> result = find_best_fit_clients(nodes, edges, max_hops=2, verbose=False)
    >>> result.to_dict(orient="records")
    [{'prospect_id':'p1','client_id':'c1',
      'total_score':0.4,
      'path_details_dict':{'p1→a1→c1':'pd1|ac1'}}]

    Performance Tuning
    ------------------
    - Cast the ID columns to `category` dtype to speed up merges.
    - If cycle-checking on Python lists is slow, rewrite that part in numba or C.
    - For very large graphs, lower `max_hops` or pre-prune edges by minimum weight.

    """
    t0 = time.perf_counter()

    # 1) Identify Prospect and Client sets
    prospect_ids = set(nodes_df.loc[nodes_df["entity_type"]=="Prospect", "ID"])
    client_ids   = set(nodes_df.loc[nodes_df["entity_type"]=="Client",  "ID"])
    if verbose:
        print(f"[{time.perf_counter()-t0:.2f}s] Found "
              f"{len(prospect_ids)} prospects and {len(client_ids)} clients.")

    # 2) Build an undirected edge table once
    edges_rev = edges_df.rename(columns={"source":"target","target":"source"})
    edges_ud  = pd.concat([edges_df, edges_rev], ignore_index=True)
    # Drop self‐loops
    edges_ud = edges_ud.loc[edges_ud["source"] != edges_ud["target"]].reset_index(drop=True)
    if verbose:
        print(f"[{time.perf_counter()-t0:.2f}s] Built undirected edges "
              f"({len(edges_ud)} rows).")

    # 3) Hop 0: seed frontier with any edge touching a Prospect
    df0 = edges_ud[
        edges_ud["source"].str.startswith("p") |
        edges_ud["target"].str.startswith("p")
    ].copy()
    # Orient so Prospect always in 'source'
    mask_flip = ~df0["source"].str.startswith("p")
    df0.loc[mask_flip, ["source","target"]] = df0.loc[mask_flip, ["target","source"]].values

    # Track path info as lists
    df0 = df0.assign(
        prospect_id = df0["source"],
        last_node   = df0["target"],
        path_nodes  = df0.apply(lambda r: [r["source"], r["target"]], axis=1),
        path_weights= df0["weight"].apply(lambda w: [w]),
        path_details= df0["edge_detail"].apply(lambda d: [d])
    )
    frontier_df = df0[["prospect_id","last_node","path_nodes","path_weights","path_details"]]
    hits_df     = pd.DataFrame(columns=[
        "prospect_id","client_id","path_nodes","path_weights","path_details"
    ])
    if verbose:
        print(f"[{time.perf_counter()-t0:.2f}s] Hop 0 frontier: {len(frontier_df)} rows.")

    # 4) Multi-hop expansion
    for hop in range(1, max_hops+1):
        if frontier_df.empty:
            break
        t_hop = time.perf_counter()

        # Single undirected merge to get *all* neighbors
        merged = frontier_df.merge(
            edges_ud,
            left_on="last_node",
            right_on="source",
            how="inner"
        ).assign(intermediate=lambda d: d["target"])

        # Extend path tracking lists
        merged["path_nodes"]   = merged.apply(lambda r: r["path_nodes"]   + [r["intermediate"]], axis=1)
        merged["path_weights"] = merged.apply(lambda r: r["path_weights"] + [r["weight"]],      axis=1)
        merged["path_details"] = merged.apply(lambda r: r["path_details"] + [r["edge_detail"]],  axis=1)

        # Prune cycles: drop if intermediate already in path (excluding last slot)
        merged = merged[
            ~merged.apply(lambda r: r["intermediate"] in r["path_nodes"][:-1], axis=1)
        ]

        # Split hits vs next frontier
        is_client = merged["intermediate"].isin(client_ids)
        new_hits   = merged.loc[is_client, ["prospect_id"]].assign(
            client_id   = merged.loc[is_client, "intermediate"],
            path_nodes  = merged.loc[is_client, "path_nodes"],
            path_weights= merged.loc[is_client, "path_weights"],
            path_details= merged.loc[is_client, "path_details"]
        )
        frontier_df = merged.loc[~is_client, [
            "prospect_id","intermediate","path_nodes","path_weights","path_details"
        ]].rename(columns={"intermediate":"last_node"})

        hits_df = pd.concat([hits_df, new_hits], ignore_index=True)

        if verbose:
            print(f"[{time.perf_counter()-t0:.2f}s] Hop {hop}: "
                  f"frontier={len(frontier_df)}, new_hits={len(new_hits)} "
                  f"(cumulative_hits={len(hits_df)}) – "
                  f"{time.perf_counter()-t_hop:.2f}s elapsed for this hop")

    # 5) Aggregate all hits
    t_agg = time.perf_counter()
    if hits_df.empty:
        return pd.DataFrame(columns=[
            "prospect_id","client_id","total_score","path_details_dict"
        ])

    # Compute row_score and path dicts
    hits_df = hits_df.assign(
        row_score = hits_df["path_weights"].apply(lambda w: reduce(operator.mul, w, 1.0)),
        path_key  = hits_df["path_nodes"].apply(lambda p: "→".join(p)),
        details_str = hits_df["path_details"].apply(lambda d: "|".join(d))
    )
    hits_df["path_details_dict"] = hits_df.apply(
        lambda r: {r["path_key"]: r["details_str"]}, axis=1
    )

    # Group & aggregate per prospect–client
    agg = hits_df.groupby(["prospect_id","client_id"]).agg({
        "row_score": "sum",
        "path_details_dict": lambda dicts: {
            k:v for d in dicts for k,v in d.items()
        }
    }).reset_index().rename(columns={
        "row_score":"total_score"
    })

    if verbose:
        print(f"[{time.perf_counter()-t0:.2f}s] Aggregation took "
              f"{time.perf_counter()-t_agg:.2f}s; "
              f"resolved {agg['prospect_id'].nunique()} prospects.")

    return agg