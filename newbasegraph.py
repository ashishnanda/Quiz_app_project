def find_best_fit_clients(self) -> pd.DataFrame:
    """
    For each Prospect, find the closest Client using Dijkstra's algorithm
    on an undirected graph, avoiding UBS Financial Advisors in the path.
    Returns a DataFrame with Prospect -> Best-fit Client mappings.
    """
    import time
    import networkx as nx
    from heapq import heappush, heappop

    start = time.time()

    G: nx.Graph = self.graph  # Ensure this is undirected (nx.Graph, not nx.DiGraph)

    clients = {n for n, d in G.nodes(data=True) if d.get("entity_type") == "Client"}
    ubs_advisors = {n for n, d in G.nodes(data=True) if d.get("entity_type") == "UBS Financial Advisor"}
    prospects = [pid for pid in self.prospect_ids if G.has_node(pid) and G.nodes[pid].get("entity_type") == "Prospect"]

    self.unfound_prospects = [pid for pid in self.prospect_ids if not G.has_node(pid)]

    results = []

    for prospect in prospects:
        visited = set()
        heap = [(0, prospect, [])]  # (cumulative_weight, current_node, path)

        best_client = None
        best_score = float('inf')
        best_path = []

        while heap:
            cum_weight, current, path = heappop(heap)
            if current in visited:
                continue
            visited.add(current)

            if current in ubs_advisors:
                continue  # skip UBS

            new_path = path + [current]

            if current in clients:
                best_client = current
                best_score = cum_weight
                best_path = new_path
                break  # first client found with min weight

            for neighbor in G.neighbors(current):
                if neighbor not in visited:
                    edge_data = G.get_edge_data(current, neighbor)
                    weight = edge_data.get("weight", 1.0)
                    heappush(heap, (cum_weight + weight, neighbor, new_path))

        if best_client:
            relationship = []
            for i in range(len(best_path) - 1):
                u, v = best_path[i], best_path[i + 1]
                edge_data = G.get_edge_data(u, v)
                edge_desc = edge_data.get("edge_detail", "")
                relationship.append(f"{u}->{v}: {edge_desc}")

            results.append({
                "Prospect": prospect,
                "Client": best_client,
                "Relationship": "; ".join(relationship),
                "Score": best_score
            })

    result_df = pd.DataFrame(results)
    print(f"[find_best_fit_clients] Result shape: {result_df.shape}")
    print(f"[find_best_fit_clients] Unfound prospects: {len(self.unfound_prospects)}")
    print(f"[find_best_fit_clients] Completed in {time.time() - start:.2f} seconds")
    return result_df