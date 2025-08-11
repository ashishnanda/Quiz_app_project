from collections import defaultdict
from typing import Dict, Any, List

def analyze_node_relationships(gremlin_client, input_id: str) -> List[Dict[str, Any]]:
    """
    Cosmos‑compatible, fast version:
      • 2-hop flat rows, Clients only
      • Prune FA as intermediary using .not(has(...))
      • One batched query for each client's 1-hop Financial Advisor
      • Returns best-fit first, then entity_type, then name
    """
    node_id = f"{input_id}"

    # 1) Two-hop rows → Clients only; skip FA as intermediary
    query_rows = f"""
g.V('{node_id}').as('s').
  bothE().as('e1').otherV().as('m').
    not(has('entity_type','Financial Advisor')).      // Cosmos-safe FA prune
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
    by(select('m').values('entity_type').fold().coalesce(unfold().limit(1), constant(''))).
    by(select('e2').label()).
    by(coalesce(select('e2').values('weight'), constant(1))).
    by(select('end').id()).
    by(select('end').label()).
    by(select('end').values('entity_type').fold().coalesce(unfold().limit(1), constant(''))).
    by(select('end').values('networth').fold().coalesce(unfold().limit(1), constant(''))).
    by(select('end').values('graph_id').fold().coalesce(unfold().limit(1), constant('')))
  .dedup()
"""
    rows = run_query(gremlin_client, query_rows)

    client_scores: Dict[str, float] = defaultdict(float)
    entity_data: Dict[str, Dict[str, Any]] = {}

    # 2) Aggregate in Python
    for r in rows:
        end_graph_id = r.get("graph_id") or r["end_id"]
        end_type     = r["end_type"]         # 'Client'
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

        rec["intermediaries"][mid_label]          = mid_type
        rec["first_hop_relationship"][mid_label]  = r["e1_label"]
        rec["second_hop_relationship"][mid_label] = r["e2_label"]

        client_scores[end_graph_id] += path_score

    # 3) Best‑fit set
    max_score = max(client_scores.values(), default=0.0)
    best_fit_clients = {cid for cid, s in client_scores.items() if s == max_score}

    # 4) Batched 1‑hop Financial Advisor lookup (ONE traversal)
    client_ids = list(entity_data.keys())
    fa_by_client: Dict[str, str] = {}

    if client_ids:
        ids_list = ",".join(f"'{cid}'" for cid in client_ids)
        advisor_query = f"""
g.V({ids_list}).as('c').
  both().
  has('entity_type','Financial Advisor').
  group().
    by(select('c').id()).
    by(values('label').fold().coalesce(unfold().limit(1), constant('')))
"""
        # If you want advisor *name* instead of vertex label, swap values('label') → values('name')
        fa_raw = run_query(gremlin_client, advisor_query)
        if isinstance(fa_raw, list) and fa_raw and isinstance(fa_raw[0], dict):
            fa_by_client = fa_raw[0]
        elif isinstance(fa_raw, dict):
            fa_by_client = fa_raw

    # 5) Build result rows
    output: List[Dict[str, Any]] = []
    for eid, data in entity_data.items():
        lead_fa = fa_by_client.get(eid, "")
        output.append({
            "id": eid,
            "best_fit_indicator": int(eid in best_fit_clients),
            "entity_type": data["entity_type"],     # 'Client'
            "name": data["name"],
            "lead_financial_advisor": lead_fa,
            "networth": data["networth"],
            "intermediaries": data["intermediaries"],
            "first_hop_relationship": data["first_hop_relationship"],
            "second_hop_relationship": data["second_hop_relationship"],
        })

    # 6) Stable sort
    return sorted(output, key=lambda x: (-x["best_fit_indicator"], x["entity_type"], x["name"]))