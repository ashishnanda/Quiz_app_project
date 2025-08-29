from collections import defaultdict
import pandas as pd
from typing import Dict, Set, List, Tuple, Any

# ---------- Types ----------
NodeId = Any
Score = float

class MultiHopIndex:
    """
    Holds adjacency and metadata for fast multi-hop scoring.
    """
    def __init__(self):
        self.P: Set[NodeId] = set()
        self.O: Set[NodeId] = set()
        self.C: Set[NodeId] = set()

        self.P_to_O: Dict[NodeId, Set[NodeId]] = defaultdict(set)
        self.O_to_P: Dict[NodeId, Set[NodeId]] = defaultdict(set)
        self.O_to_C: Dict[NodeId, Set[NodeId]] = defaultdict(set)
        self.C_to_O: Dict[NodeId, Set[NodeId]] = defaultdict(set)

        self.W: Dict[NodeId, float] = {}  # org -> weight (constant per org)

        # Optional metadata
        self.id_to_label: Dict[NodeId, str] = {}
        self.edge_info: Dict[frozenset, Dict[str, Any]] = {}  # frozenset({u,v}) -> {weight, details}

    def orgs_with_clients(self) -> Set[NodeId]:
        return {o for o, Cs in self.O_to_C.items() if len(Cs) > 0}

    def orgs_without_clients(self) -> Set[NodeId]:
        return {o for o in self.O if len(self.O_to_C[o]) == 0}

    def is_isolated_prospect(self, p: NodeId) -> bool:
        return len(self.P_to_O[p]) == 0


# ---------- Building the index from nodes_df / edges_df ----------
def build_index(nodes_df: pd.DataFrame, edges_df: pd.DataFrame,
                org_weight_column: str = "weight") -> MultiHopIndex:
    """
    nodes_df columns: id, label, entity_type in {'prospect','org','client'}
    edges_df columns: source, target, edge_details, weight
    *Undirected* edges among P-O and O-C. We assume weight on edges equals the org weight
    for convenience; if org weight lives only on the org node, pass via nodes_df.

    Returns:
        MultiHopIndex with adjacency and metadata populated.
    """
    idx = MultiHopIndex()

    # --- Nodes
    for _, row in nodes_df.iterrows():
        nid = row["id"]
        etype = str(row["entity_type"]).lower()
        idx.id_to_label[nid] = row.get("label", str(nid))

        if etype in {"p", "prospect"}:
            idx.P.add(nid)
        elif etype in {"o", "org", "organisation", "organization"}:
            idx.O.add(nid)
            # If the org's weight is encoded on the node row, prefer that:
            if org_weight_column in row and pd.notna(row[org_weight_column]):
                idx.W[nid] = float(row[org_weight_column])
        elif etype in {"c", "client"}:
            idx.C.add(nid)

    # --- Edges (undirected)
    for _, row in edges_df.iterrows():
        u = row["source"]; v = row["target"]
        w = row.get("weight", None)
        details = row.get("edge_details", None)

        idx.edge_info[frozenset({u, v})] = {"weight": w, "details": details}

        # normalize P-O or O-C
        if u in idx.P and v in idx.O:
            idx.P_to_O[u].add(v)
            idx.O_to_P[v].add(u)
            if w is not None and v in idx.O and v not in idx.W:
                # fallback: if W not set from nodes_df, derive from edge weight
                idx.W[v] = float(w)
        elif v in idx.P and u in idx.O:
            idx.P_to_O[v].add(u)
            idx.O_to_P[u].add(v)
            if w is not None and u in idx.O and u not in idx.W:
                idx.W[u] = float(w)
        elif u in idx.O and v in idx.C:
            idx.O_to_C[u].add(v)
            idx.C_to_O[v].add(u)
            if w is not None and u in idx.O and u not in idx.W:
                idx.W[u] = float(w)
        elif v in idx.O and u in idx.C:
            idx.O_to_C[v].add(u)
            idx.C_to_O[u].add(v)
            if w is not None and v in idx.O and v not in idx.W:
                idx.W[v] = float(w)
        else:
            # Ignore any non P-O/C-O edges silently.
            pass

    # Ensure every org has a weight; default to 1.0 if missing
    for o in idx.O:
        if o not in idx.W or pd.isna(idx.W[o]):
            idx.W[o] = 1.0

    return idx


# ---------- Core scoring utilities ----------
def two_hop_scores(idx: MultiHopIndex) -> Tuple[
        Dict[NodeId, Dict[NodeId, Score]],
        Dict[NodeId, Dict[NodeId, List[NodeId]]]]:
    """
    Compute S^(2): for each prospect p, scores[p][c] = sum_{o in N(p)∩N(c)} w(o)^2
    Also record a 'witness' path p-o*-c that contributed the largest single-path amount.

    Returns:
        scores, witness
        scores[p][c] : float
        witness[p][c] : [p, o, c]  (one example path with highest single contribution)
    """
    scores = defaultdict(lambda: defaultdict(float))
    witness = defaultdict(lambda: defaultdict(list))

    O_with_C = idx.orgs_with_clients()

    for p in idx.P:
        for o in idx.P_to_O[p]:
            if o not in O_with_C:
                continue  # org cannot reach a client directly
            contrib = idx.W[o] ** 2
            for c in idx.O_to_C[o]:
                scores[p][c] += contrib
                # Update witness if this path is best single-path contributor so far
                cur = witness[p].get(c)
                if not cur or contrib > (idx.W[cur[1]] ** 2):
                    witness[p][c] = [p, o, c]
    return scores, witness


def bridge_factor(idx: MultiHopIndex, p: NodeId, q: NodeId) -> float:
    """
    α_{pq} = sum_{o in (P_to_O[p] ∩ P_to_O[q])} w(o)^2
    The 2-hop factor for p—o—q.
    """
    # iterate via the smaller neighbor set for speed
    s1, s2 = idx.P_to_O[p], idx.P_to_O[q]
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    inter = s1.intersection(s2)
    if not inter:
        return 0.0
    return sum(idx.W[o] ** 2 for o in inter)


def expand_layer(idx: MultiHopIndex,
                 prev_scores: Dict[NodeId, Dict[NodeId, Score]],
                 prev_witness: Dict[NodeId, Dict[NodeId, List[NodeId]]],
                 candidates: Set[NodeId],
                 reached: Set[NodeId]) -> Tuple[
                     Dict[NodeId, Dict[NodeId, Score]],
                     Dict[NodeId, Dict[NodeId, List[NodeId]]],
                     Set[NodeId]]:
    """
    Compute next 2-hop layer:
    For each p in 'candidates' (not yet reached), accumulate from each q in 'reached':
        α_{pq} * prev_scores[q][c]
    Also create a witness path by stitching [p, o1, q] + (prev witness of q->c).

    Returns:
        new_scores, new_witness, newly_reached
    """
    new_scores = defaultdict(lambda: defaultdict(float))
    new_witness = defaultdict(lambda: defaultdict(list))
    newly_reached = set()

    for p in candidates:
        # Build α_{pq} only where intersection is non-empty
        # Quick skip: if P_to_O[p] is empty, can't connect anywhere
        if len(idx.P_to_O[p]) == 0:
            continue
        for q in reached:
            α = bridge_factor(idx, p, q)
            if α <= 0.0:
                continue
            for c, sc_qc in prev_scores[q].items():
                val = α * sc_qc
                new_scores[p][c] += val

                # Build a witness by picking one org that realizes part of α and concatenating
                # We choose the org of max w^2 in the intersection for a strong path
                inter = idx.P_to_O[p].intersection(idx.P_to_O[q])
                if inter:
                    o1 = max(inter, key=lambda o: idx.W[o] ** 2)
                    # prev_witness[q][c] is [q, o2, (maybe mid prospects...), c] starting at q.
                    prev_path = prev_witness[q].get(c, [q])
                    # Make a simple stitched path: [p, o1, q] + prev_path (skipping duplicate q)
                    stitched = [p, o1] + prev_path
                    # Update if this single-path product beats the current witness contribution
                    # (Heuristic: compare the product contributed by o1 times best step inside prev_path)
                    cur = new_witness[p].get(c)
                    if not cur:
                        new_witness[p][c] = stitched
                    else:
                        # Very light heuristic tie-breaker: prefer path with larger first-org contribution
                        if idx.W[o1] ** 2 > idx.W[cur[1]] ** 2:
                            new_witness[p][c] = stitched

        if new_scores[p]:  # gained something
            newly_reached.add(p)

    return new_scores, new_witness, newly_reached


def run_multihop(idx: MultiHopIndex,
                 max_hops: int = 6,
                 remainder_threshold: int = 0,
                 stop_if_no_growth: bool = True) -> Dict[str, Any]:
    """
    Execute layered multi-hop inference WITHOUT decay.
    - Starts with 2-hop layer (p-o-c).
    - Then iteratively adds 4-hop, 6-hop, ... up to max_hops.
    - Early stopping:
        * if no new prospects get a score at a layer (no growth), stop.
        * if remaining prospects <= remainder_threshold, stop.

    Returns:
        {
          'scores_cumulative': Dict[p][c] -> score up to last hop,
          'best_clients': Dict[p] -> set/list of argmax clients,
          'witness': Dict[p][c] -> example path,
          'per_layer': [{'hop': 2, 'new_reached': set(...), 'scores':..., 'witness':...}, ...]
        }
    """
    assert max_hops % 2 == 0 and max_hops >= 2

    # 2-hop base
    S2, W2 = two_hop_scores(idx)
    reached = {p for p, d in S2.items() if d}  # R1
    all_scores = defaultdict(lambda: defaultdict(float))
    witness = defaultdict(lambda: defaultdict(list))

    for p, dc in S2.items():
        for c, v in dc.items():
            all_scores[p][c] += v
            witness[p][c] = W2[p][c]

    per_layer = [{"hop": 2, "new_reached": set(reached), "scores": S2, "witness": W2}]

    # 4,6,... hops
    candidates = set(idx.P) - reached
    hop = 4
    while hop <= max_hops:
        S_next, W_next, newly = expand_layer(idx, S2, W2, candidates, reached)
        if not newly and stop_if_no_growth:
            break
        # Merge cumulative
        for p, dc in S_next.items():
            for c, v in dc.items():
                all_scores[p][c] += v
                # Only set witness if we don't already have one (prefer shorter paths as explanation)
                if not witness[p].get(c):
                    witness[p][c] = W_next[p][c]

        per_layer.append({"hop": hop, "new_reached": newly, "scores": S_next, "witness": W_next})

        # Prepare next iteration
        reached |= newly
        candidates -= newly
        if len(candidates) <= remainder_threshold:
            break

        # The recurrence for the next layer uses the most recent layer as 'prev'
        S2, W2 = S_next, W_next
        hop += 2

    # Argmax set per prospect
    best_clients = {}
    for p, d in all_scores.items():
        if not d:
            continue
        maxv = max(d.values())
        best_clients[p] = [c for c, v in d.items() if v == maxv]

    return {
        "scores_cumulative": all_scores,
        "best_clients": best_clients,
        "witness": witness,
        "per_layer": per_layer,
    }


# ---------- Utilities to post-process into DataFrames ----------
def scores_to_dataframe(idx: MultiHopIndex,
                        scores: Dict[NodeId, Dict[NodeId, Score]],
                        best_clients: Dict[NodeId, List[NodeId]]) -> pd.DataFrame:
    """
    Flatten scores dict to a tidy DataFrame with labels and best-fit flag.
    """
    rows = []
    best_pairs = {(p, c) for p, cs in best_clients.items() for c in cs}
    for p, d in scores.items():
        for c, v in d.items():
            rows.append({
                "prospect_id": p,
                "prospect_label": idx.id_to_label.get(p, str(p)),
                "client_id": c,
                "client_label": idx.id_to_label.get(c, str(c)),
                "score": v,
                "is_best_fit": (p, c) in best_pairs
            })
    return pd.DataFrame(rows)


def path_to_readable(idx: MultiHopIndex, path: List[NodeId]) -> str:
    """
    Convert a node-id path to a human-readable string with labels.
    """
    if not path:
        return ""
    labs = [idx.id_to_label.get(x, str(x)) for x in path]
    return " -> ".join(labs)