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
    Identify the best-fit Client for each Prospect via multi-hop traversal.

    This function assumes you have already bulk-loaded:
      - `nodes_df`: a DataFrame with columns
          - 'ID' (str): node identifier, e.g. 'p1', 'c42', 'a7'
          - 'entity_type' (str): one of 'Prospect', 'Client', 'UBS Financial Advisor', or others
      - `edges_df`: a DataFrame with columns
          - 'source' (str), 'target' (str): node IDs
          - 'weight' (float): in (0,1]
          - 'edge_detail' (str): description of the relationship

    The algorithm:
      1. **Hop 0**: filter edges touching any Prospect and orient so Prospect is `source`.
      2. **Initialize frontier** with these edges, tracking:
         - `prospect_id`, `last_node`
         - `path_nodes` (list of visited nodes)
         - `path_weights` (list of weights)
         - `path_details` (list of edge_detail)
      3. **Iterate hops** 1..max_hops:
         - **Vectorized merge** frontier ⇄ edges to find next‐hop neighbors.
         - **Orient** so `intermediate1 → intermediate2`.
         - **Extend** path lists.
         - **Prune cycles** (drop rows where intermediate2 in path_nodes).
         - **Split** into:
             - **Hits**: intermediate2 is a Client (collect these rows).
             - **New frontier**: intermediate2 is not a Client.
      4. **Aggregate hits**:
         - Per row, compute `row_score = product(path_weights)`.
         - Build `path_key` as "n1→n2→…→nK".
         - Build `details_str` as "d12|d23|…".
         - Group by `(prospect_id, client_id)`:
           - Sum all `row_score` → `total_score`.
           - Merge all per-row dicts into one `path_details_dict`.
      5. **Return** DataFrame with columns:
         - `prospect_id`, `client_id`, `total_score`, `path_details_dict`.

    **Performance tuning tips**:
      - Convert `nodes_df['ID']`, `edges_df['source']`, `edges_df['target']`
        to `category` dtype to speed up merges.
      - If cycle‐check becomes a bottleneck, consider `numba` or storing
        `path_nodes` as tuples and using set‐based lookups.
      - For very large graphs, restrict `max_hops` or pre-prune by weight.

    **Example**:
    ```python
    import pandas as pd

    nodes = pd.DataFrame([
        {"ID": "p1", "entity_type": "Prospect"},
        {"ID": "a1", "entity_type": "Other"},
        {"ID": "c1", "entity_type": "Client"}
    ])

    edges = pd.DataFrame([
        {"source": "p1", "target": "a1", "weight": 0.5, "edge_detail": "pd1"},
        {"source": "a1", "target": "c1", "weight": 0.8, "edge_detail": "ac1"}
    ])

    result = find_best_fit_clients(nodes, edges, max_hops=2, verbose=True)
    # result will contain:
    #   prospect_id client_id total_score                           path_details_dict
    # 0         p1        c1         0.40  {'p1→a1→c1': 'pd1|ac1'}
    ```

    :param nodes_df: DataFrame of all nodes with ID and entity_type.
    :param edges_df: DataFrame of all edges with source, target, weight, edge_detail.
    :param max_hops: Maximum number of hops to traverse.
    :param verbose: If True, print timing and status updates.
    :return: DataFrame with columns
             ['prospect_id','client_id','total_score','path_details_dict'].
    """
    t0 = time.perf_counter()

    # 1. Identify Prospect and Client sets
    prospect_ids = set(nodes_df.loc[nodes_df["entity_type"]=="Prospect", "ID"])
    client_ids   = set(nodes_df.loc[nodes_df["entity_type"]=="Client",  "ID"])

    if verbose:
        print(f"[{time.perf_counter()-t0:.2f}s] Step 1: Found"
              f" {len(prospect_ids)} prospects and {len(client_ids)} clients.")

    # 2. Hop 0: filter edges touching a Prospect
    df0 = edges_df[
        edges_df["source"].str.startswith("p") |
        edges_df["target"].str.startswith("p")
    ].copy()
    # orient so Prospect is always in 'source'
    mask_flip = ~df0["source"].str.startswith("p")
    df0.loc[mask_flip, ["source","target"]] = df0.loc[mask_flip, ["target","source"]].values

    # initialize columns for path tracking
    df0 = df0.assign(
        prospect_id = df0["source"],
        last_node   = df0["target"],
        path_nodes  = df0.apply(lambda r: [r["source"], r["target"]], axis=1),
        path_weights= df0["weight"].apply(lambda w: [w]),
        path_details= df0["edge_detail"].apply(lambda d: [d])
    )
    frontier_df = df0[["prospect_id","last_node","path_nodes","path_weights","path_details"]]
    hits_df = pd.DataFrame(columns=[
        "prospect_id","client_id","path_nodes","path_weights","path_details"
    ])

    if verbose:
        print(f"[{time.perf_counter()-t0:.2f}s] Step 2: Hop 0 frontier:"
              f" {len(frontier_df)} rows.")

    # 3. Multi-hop expansion
    for hop in range(1, max_hops+1):
        t_hop_start = time.perf_counter()
        if frontier_df.empty:
            break

        # join frontier to edges on either end
        left = frontier_df.merge(
            edges_df, left_on="last_node", right_on="source", how="inner", suffixes=("","_e")
        )
        right = frontier_df.merge(
            edges_df, left_on="last_node", right_on="target", how="inner", suffixes=("","_e")
        )
        # orient right-joined edges
        right = right.rename(columns={"source_e":"target","target_e":"source"})
        right = right.drop(columns=["source","target"]).rename(
            columns={"source_e":"source","target_e":"target"}
        )
        merged = pd.concat([left, right], ignore_index=True)

        # derive next intermediate node
        merged = merged.assign(
            intermediate = merged["target"],
        )

        # extend path columns
        merged["path_nodes"]   = merged.apply(lambda r: r["path_nodes"]   + [r["intermediate"]], axis=1)
        merged["path_weights"] = merged.apply(lambda r: r["path_weights"] + [r["weight"]], axis=1)
        merged["path_details"] = merged.apply(lambda r: r["path_details"] + [r["edge_detail"]], axis=1)

        # prune cycles
        merged = merged[~merged.apply(lambda r: r["intermediate"] in r["path_nodes"][:-1], axis=1)]

        # split hits vs next frontier
        is_client = merged["intermediate"].isin(client_ids)
        new_hits = merged.loc[is_client, [
            "prospect_id"
        ]].assign(
            client_id  = merged.loc[is_client, "intermediate"],
            path_nodes = merged.loc[is_client, "path_nodes"],
            path_weights = merged.loc[is_client, "path_weights"],
            path_details = merged.loc[is_client, "path_details"]
        )
        frontier_df = merged.loc[~is_client, [
            "prospect_id","intermediate","path_nodes","path_weights","path_details"
        ]].rename(columns={"intermediate":"last_node"})

        # collect hits
        hits_df = pd.concat([hits_df, new_hits], ignore_index=True)

        if verbose:
            print(f"[{time.perf_counter()-t0:.2f}s] Hop {hop}:"
                  f" frontier={len(frontier_df)}, new_hits={len(new_hits)},"
                  f" cumulative_hits={len(hits_df)}, took {time.perf_counter()-t_hop_start:.2f}s")

    # 4. Aggregate all hits
    t_agg = time.perf_counter()
    if hits_df.empty:
        return pd.DataFrame(
            columns=["prospect_id","client_id","total_score","path_details_dict"]
        )

    # compute row_score and path_details_dict
    hits_df = hits_df.assign(
        row_score = hits_df["path_weights"]
            .apply(lambda w: reduce(operator.mul, w, 1.0)),
        path_key = hits_df["path_nodes"]
            .apply(lambda p: "→".join(p)),
        details_str = hits_df["path_details"]
            .apply(lambda d: "|".join(d))
    )
    hits_df["path_dict"] = hits_df.apply(
        lambda r: {r["path_key"]: r["details_str"]}, axis=1
    )

    # group & aggregate
    agg = hits_df.groupby(["prospect_id","client_id"]).agg({
        "row_score": "sum",
        "path_dict": lambda dicts: {k:v for d in dicts for k,v in d.items()}
    }).reset_index().rename(columns={
        "row_score":"total_score", "path_dict":"path_details_dict"
    })

    if verbose:
        print(f"[{time.perf_counter()-t0:.2f}s] Aggregation took {time.perf_counter()-t_agg:.2f}s")
        print(f"[{time.perf_counter()-t0:.2f}s] Total prospects resolved: {agg['prospect_id'].nunique()}")

    return agg