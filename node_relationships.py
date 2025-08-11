def analyze_node_relationships(gremlin_client, input_id: str) -> List[Dict[str, Any]]:
    node_id = f"{input_id}"

    # --- 1) Get all 2-hop paths to Client/Prospect endpoints ---
    query_paths = f"""
g.V('{node_id}').
  repeat(bothE().otherV().simplePath()).times(2).
  has('entity_type', within('Client', 'Prospect')).
  path().
  dedup()
"""
    paths = run_query(gremlin_client, query_paths)

    client_scores: Dict[str, float] = defaultdict(float)
    entity_data: Dict[str, Dict[str, Any]] = {}

    # --- 2) Parse & aggregate in Python (cheap) ---
    _path_objects_local = _path_objects
    _get_vertex_prop_local = _get_vertex_prop
    _edge_weight_local = _edge_weight

    for path in paths:
        objs = _path_objects_local(path)
        if len(objs) != 5:
            continue  # malformed

        try:
            start_node, edge1, intermediate_node, edge2, end_node = objs

            # Skip if intermediary is a Financial Advisor
            intermediate_type = _get_vertex_prop_local(intermediate_node, "entity_type", "")
            if intermediate_type == "Financial Advisor":
                continue

            # End vertex fields
            end_graph_id = _get_vertex_prop_local(end_node, "graph_id", end_node.get("id", ""))
            end_type = _get_vertex_prop_local(end_node, "entity_type", "")
            end_label = end_node.get("label", "")
            networth = _get_vertex_prop_local(end_node, "networth", "")

            # Edges
            edge1_label = edge1.get("label", "")
            edge2_label = edge2.get("label", "")
            path_score = _edge_weight_local(edge1) * _edge_weight_local(edge2)

            # Intermediary label
            intermediate_label = intermediate_node.get("label", "")

            # Init bucket once
            rec = entity_data.get(end_graph_id)
            if rec is None:
                rec = entity_data[end_graph_id] = {
                    "id": end_graph_id,
                    "entity_type": end_type,
                    "name": end_label,
                    "networth": networth,
                    "intermediaries": {},
                    "first_hop_relationship": {},
                    "second_hop_relationship": {},
                }

            # Update per‑intermediary details
            rec["intermediaries"][intermediate_label] = intermediate_type
            rec["first_hop_relationship"][intermediate_label] = edge1_label
            rec["second_hop_relationship"][intermediate_label] = edge2_label

            # Score only Clients
            if end_type == "Client":
                client_scores[end_graph_id] += path_score

        except Exception as e:
            print(f"Error processing path: {e}")
            continue

    # --- 3) Best‑fit client(s) by max score ---
    max_score = max(client_scores.values(), default=0.0)
    best_fit_clients = {cid for cid, s in client_scores.items() if s == max_score}

    # --- 4) BATCH fetch 1‑hop Financial Advisor for all clients in ONE query ---
    client_ids = [eid for eid, d in entity_data.items() if d["entity_type"] == "Client"]
    fa_by_client: Dict[str, str] = {}

    if client_ids:
        # Build an id list usable by g.V()
        ids_list = ",".join(f"'{cid}'" for cid in client_ids)

        # Return a map: clientId -> first advisor's 'label' (or "" if none)
        advisor_query = f"""
g.V({ids_list}).
  as('c').
  both().
  has('entity_type', 'Financial Advisor').
  group().
    by(select('c').id()).
    by(values('label').fold().coalesce(unfold().limit(1), constant('')))
"""
        # Depending on driver, this is a single dict or a 1‑element list containing a dict
        fa_raw = run_query(gremlin_client, advisor_query)
        if isinstance(fa_raw, list) and fa_raw and isinstance(fa_raw[0], dict):
            fa_by_client = fa_raw[0]
        elif isinstance(fa_raw, dict):
            fa_by_client = fa_raw

    # --- 5) Build output (no extra Gremlin calls) ---
    output: List[Dict[str, Any]] = []
    for eid, data in entity_data.items():
        lead_fa = fa_by_client.get(eid, "") if data["entity_type"] == "Client" else ""
        output.append({
            "id": eid,
            "best_fit_indicator": int(eid in best_fit_clients),
            "entity_type": data["entity_type"],
            "name": data["name"],
            "lead_financial_advisor": lead_fa,
            "networth": data["networth"],
            "intermediaries": data["intermediaries"],
            "first_hop_relationship": data["first_hop_relationship"],
            "second_hop_relationship": data["second_hop_relationship"],
        })

    # --- 6) Stable UX sort ---
    return sorted(output, key=lambda x: (-x["best_fit_indicator"], x["entity_type"], x["name"]))
    
    
    
    
    
    from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Iterable

# Try to import the Gremlin error type; fall back to Exception if it's not available
try:
    from gremlin_python.driver.protocol import GremlinServerError  # type: ignore
except Exception:  # pragma: no cover
    GremlinServerError = Exception  # type: ignore


# ---------- Small utilities to make property access safe & readable ----------

def _get_vertex_prop(v: Dict[str, Any], key: str, default: Any = "") -> Any:
    """
    Vertices returned in a path usually look like:
      {'id': '...', 'label': 'Person', 'properties': {'entity_type': [{'id': 'p~1','value':'Client'}], ...}}
    This helper returns the scalar value if present, else `default`.
    """
    props = v.get("properties", {})
    val = props.get(key)
    if val is None:
        return default
    # Gremlin valueMap(true) often yields a list of dicts with a 'value' key
    if isinstance(val, list) and val:
        first = val[0]
        if isinstance(first, dict) and "value" in first:
            return first.get("value", default)
        return first
    return val


def _get_edge_prop(e: Dict[str, Any], key: str, default: Any = "") -> Any:
    """
    Edges in a path tend to be dicts like:
      {'id':'...','label':'knows','inV':'...','outV':'...','properties': {'weight': 1}}
    Handle both scalar and list-of-dict cases.
    """
    props = e.get("properties", {})
    val = props.get(key, default)
    if isinstance(val, list) and val:
        first = val[0]
        if isinstance(first, dict) and "value" in first:
            return first.get("value", default)
        return first
    return val


def _edge_weight(e: Dict[str, Any]) -> float:
    w = _get_edge_prop(e, "weight", 1)
    try:
        return float(w)
    except Exception:
        return 1.0


def _path_objects(p: Any) -> List[Dict[str, Any]]:
    """
    Gremlin 'path()' results often arrive as objects like {'labels': [...], 'objects': [ ... five elements ... ]}
    Return the list of objects if present, else [].
    """
    if isinstance(p, dict):
        return p.get("objects", []) or p.get("object", []) or []
    # Some drivers return a tuple/list directly
    if isinstance(p, (list, tuple)):
        return list(p)
    return []


# ------------------------------- Gremlin runner -------------------------------

def run_query(gremlin_client, query: str) -> List[Any]:
    """
    Submit a Gremlin query and return the fully materialized results list.
    """
    try:
        result_set = gremlin_client.submit(query)
        return result_set.all().result()
    except GremlinServerError as e:
        print(f"❌ Gremlin Server Error: {e}")
        return []


# ----------------------- Main: analyze node relationships ----------------------

def analyze_node_relationships(gremlin_client, input_id: str) -> List[Dict[str, Any]]:
    """
    From a given node, find all 2-hop paths that end at Clients or Prospects.
    - Score client endpoints using the product of the two edge weights.
    - Track the intermediary node and each hop's relationship label.
    - Mark the "best-fit" client(s): the one(s) with the maximum score.
    - For each Client endpoint, also pull a 1-hop Financial Advisor (if any).

    Returns a list of dicts with:
      id, best_fit_indicator, entity_type, name, lead_financial_advisor,
      networth, intermediaries, first_hop_relationship, second_hop_relationship

    Sorted with best-fit first, then by entity_type and name.
    """

    node_id = f"{input_id}"

    # 1) Gremlin: all 2-hop paths from the source to Client/Prospect endpoints.
    #    - bothE().otherV(): step to an edge then to the opposite vertex (undirected feel)
    #    - simplePath(): avoid revisiting vertices within the same path
    #    - times(2): exactly two hops
    #    - has('entity_type', within('Client','Prospect')): end vertex filter
    #    - path().dedup(): return unique paths as Path objects
    query_paths = f"""
g.V('{node_id}').
  repeat(bothE().otherV().simplePath()).times(2).
  has('entity_type', within('Client', 'Prospect')).
  path().
  dedup()
"""

    paths = run_query(gremlin_client, query_paths)

    client_scores: Dict[str, float] = defaultdict(float)
    entity_data: Dict[str, Dict[str, Any]] = {}

    # 2) Walk each returned path and harvest details
    for path in paths:
        objs = _path_objects(path)
        if len(objs) != 5:
            # Expecting: start_vertex, edge1, intermediary_vertex, edge2, end_vertex
            continue

        try:
            start_node, edge1, intermediate_node, edge2, end_node = objs

            # Skip if the intermediary is a Financial Advisor
            intermediate_type = _get_vertex_prop(intermediate_node, "entity_type", "")
            if intermediate_type == "Financial Advisor":
                continue

            # Key fields from the end vertex
            end_graph_id = _get_vertex_prop(end_node, "graph_id", end_node.get("id", ""))
            end_type = _get_vertex_prop(end_node, "entity_type", "")
            end_label = end_node.get("label", "")
            networth = _get_vertex_prop(end_node, "networth", "")

            # Edge labels & weights
            edge1_label = edge1.get("label", "")
            edge2_label = edge2.get("label", "")
            edge1_weight = _edge_weight(edge1)
            edge2_weight = _edge_weight(edge2)
            path_score = edge1_weight * edge2_weight

            # Intermediary details
            intermediate_label = intermediate_node.get("label", "")

            # Prepare / update the endpoint's aggregation record
            if end_graph_id not in entity_data:
                entity_data[end_graph_id] = {
                    "id": end_graph_id,
                    "entity_type": end_type,
                    "name": end_label,
                    "networth": networth,
                    "intermediaries": {},
                    "first_hop_relationship": {},
                    "second_hop_relationship": {},
                }

            entity_data[end_graph_id]["intermediaries"][intermediate_label] = intermediate_type
            entity_data[end_graph_id]["first_hop_relationship"][intermediate_label] = edge1_label
            entity_data[end_graph_id]["second_hop_relationship"][intermediate_label] = edge2_label

            # Score only Clients
            if end_type == "Client":
                client_scores[end_graph_id] += path_score

        except Exception as e:
            print(f"Error processing path: {e}")
            continue

    # 3) Identify best-fit clients: those that tie for the highest score
    max_score = max(client_scores.values(), default=0)
    best_fit_clients = {cid for cid, score in client_scores.items() if score == max_score}

    # 4) Build final output records, and for Clients fetch a 1‑hop Financial Advisor
    output: List[Dict[str, Any]] = []
    for eid, data in entity_data.items():
        lead_fa = ""
        if data["entity_type"] == "Client":
            advisor_query = f"""
g.V('{eid}').
  both().
  has('entity_type', 'Financial Advisor').
  dedup().
  valueMap(true)
"""
            advisors = run_query(gremlin_client, advisor_query)
            if advisors:
                # First advisor label (valueMap(true) can return a list)
                first = advisors[0]
                raw = first.get("label", "")
                if isinstance(raw, list) and raw:
                    raw = raw[0]
                lead_fa = raw or ""

        output.append({
            "id": eid,
            "best_fit_indicator": int(eid in best_fit_clients),
            "entity_type": data["entity_type"],
            "name": data["name"],
            "lead_financial_advisor": lead_fa,
            "networth": data["networth"],
            "intermediaries": data["intermediaries"],
            "first_hop_relationship": data["first_hop_relationship"],
            "second_hop_relationship": data["second_hop_relationship"],
        })

    # 5) Sort: best‑fit first, then by entity type and name for stable UX
    sorted_output = sorted(
        output,
        key=lambda x: (-x["best_fit_indicator"], x["entity_type"], x["name"])
    )
    return sorted_output
    
    
    
from collections import defaultdict
from typing import Dict, Any, List

def analyze_node_relationships(gremlin_client, input_id: str) -> List[Dict[str, Any]]:
    """
    Optimized version:
      - Uses a flat 'project(...)' traversal (no heavy valueMap/Path objects)
      - Prunes FA as intermediaries and keeps only Client endpoints (fewer rows)
      - Batches the 1-hop Financial Advisor lookup for all Clients in one query
      - Returns rows sorted with best-fit first, then entity_type, then name
    """
    node_id = f"{input_id}"

    # --- 1) Compact 2‑hop rows to Clients only; skip FA as intermediary (Cosmos‑safe) ---
    query_rows = f"""
g.V('{node_id}').as('s').
  bothE().as('e1').otherV().as('m').
    where(values('entity_type').without('Financial Advisor')).
  bothE().as('e2').otherV().as('end').
    has('entity_type','Client').
  project(
    'start_id','e1_label','e1_weight',
    'mid_id','mid_label','mid_type',
    'e2_label','e2_weight',
    'end_id','end_label','end_type','networth','graph_id'
  ).
    by(select('s').id()).
    by(select('e1').label()).
    by(coalesce(select('e1').values('weight'), constant(1))).
    by(select('m').id()).
    by(select('m').label()).
    by(select('m').values('entity_type')
         .fold().coalesce(unfold().limit(1), constant(''))).
    by(select('e2').label()).
    by(coalesce(select('e2').values('weight'), constant(1))).
    by(select('end').id()).
    by(select('end').label()).
    by(select('end').values('entity_type')
         .fold().coalesce(unfold().limit(1), constant(''))).
    by(select('end').values('networth')
         .fold().coalesce(unfold().limit(1), constant(''))).
    by(select('end').values('graph_id')
         .fold().coalesce(unfold().limit(1), constant('')))
  .dedup()
"""
    rows = run_query(gremlin_client, query_rows)

    client_scores: Dict[str, float] = defaultdict(float)
    entity_data: Dict[str, Dict[str, Any]] = {}

    # --- 2) Parse + aggregate (cheap Python work) ---
    for r in rows:
        end_graph_id = r.get("graph_id") or r["end_id"]
        end_type     = r["end_type"]         # will be 'Client' by construction
        end_label    = r["end_label"]
        networth     = r.get("networth", "")
        mid_label    = r["mid_label"]
        mid_type     = r["mid_type"]
        path_score   = float(r["e1_weight"]) * float(r["e2_weight"])

        rec = entity_data.get(end_graph_id)
        if rec is None:
            rec = entity_data[end_graph_id] = {
                "id": end_graph_id,
                "entity_type": end_type,
                "name": end_label,
                "networth": networth,
                "intermediaries": {},
                "first_hop_relationship": {},
                "second_hop_relationship": {},
            }

        rec["intermediaries"][mid_label]           = mid_type
        rec["first_hop_relationship"][mid_label]   = r["e1_label"]
        rec["second_hop_relationship"][mid_label]  = r["e2_label"]

        # score clients (all endpoints here are Clients)
        client_scores[end_graph_id] += path_score

    # --- 3) Best‑fit set (ties allowed) ---
    max_score = max(client_scores.values(), default=0.0)
    best_fit_clients = {cid for cid, s in client_scores.items() if s == max_score}

    # --- 4) Batch fetch 1‑hop Financial Advisor (ONE traversal for all clients) ---
    client_ids = list(entity_data.keys())  # all endpoints are Clients
    fa_by_client: Dict[str, str] = {}

    if client_ids:
        ids_list = ",".join(f"'{cid}'" for cid in client_ids)
        advisor_query = f"""
g.V({ids_list}).as('c').
  both().
  has('entity_type','Financial Advisor').
  group().
    by(select('c').id()).
    by(label().fold().coalesce(unfold().limit(1), constant('')))
"""
        # Cosmos returns either a dict or [dict]
        fa_raw = run_query(gremlin_client, advisor_query)
        if isinstance(fa_raw, list) and fa_raw and isinstance(fa_raw[0], dict):
            fa_by_client = fa_raw[0]
        elif isinstance(fa_raw, dict):
            fa_by_client = fa_raw

    # --- 5) Build final output (no extra round‑trips) ---
    output: List[Dict[str, Any]] = []
    for eid, data in entity_data.items():
        lead_fa = fa_by_client.get(eid, "")
        output.append({
            "id": eid,
            "best_fit_indicator": int(eid in best_fit_clients),
            "entity_type": data["entity_type"],     # 'Client'
            "name": data["name"],
            "lead_financial_advisor": lead_fa,      # element label of FA (e.g., 'Financial Advisor')
            "networth": data["networth"],
            "intermediaries": data["intermediaries"],
            "first_hop_relationship": data["first_hop_relationship"],
            "second_hop_relationship": data["second_hop_relationship"],
        })

    # --- 6) Stable UX sort ---
    return sorted(output, key=lambda x: (-x["best_fit_indicator"], x["entity_type"], x["name"]))
    