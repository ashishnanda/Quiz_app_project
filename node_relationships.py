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