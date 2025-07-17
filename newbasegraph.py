def load_data(self, 
              nodes_df: pd.DataFrame = None, 
              edges_df: pd.DataFrame = None, 
              prospect_ids: list = None):
    """
    Load graph data from either:
    1. Provided DataFrames (nodes_df, edges_df)
    2. Or fallback to SQL tables.
    
    Prospect IDs:
    - If `prospect_ids` list is provided, use it directly.
    - Otherwise, always fetch from prospect table via SQL (even if nodes_df is given).

    Sets:
        - self.nodes_df
        - self.edges_df
        - self.prospect_df
        - self.prospect_ids
    """
    import pandas as pd
    from sqlalchemy import select, literal_column, Table

    if nodes_df is not None and edges_df is not None:
        print("[load_data] Using provided DataFrames for nodes and edges.")
        self.nodes_df = nodes_df.copy()
        self.edges_df = edges_df.copy()
    else:
        print("[load_data] Loading nodes and edges from SQL.")
        metadata = self.metadata
        metadata.reflect(bind=self.engine, schema=self.nodes_schema_name, only=[self.nodes_table_name])
        metadata.reflect(bind=self.engine, schema=self.edge_schema_name, only=[self.edge_table_name])

        nodes_table = Table(self.nodes_table_name, metadata, autoload_with=self.engine, schema=self.nodes_schema_name)
        edges_table = Table(self.edge_table_name, metadata, autoload_with=self.engine, schema=self.edge_schema_name)

        nodes_query = select(
            literal_column("ID"),
            literal_column("Label"),
            literal_column("entity_type")
        ).select_from(nodes_table)

        edges_query = select(
            literal_column("source"),
            literal_column("target"),
            literal_column("weight"),
            literal_column("edge_detail")
        ).select_from(edges_table)

        self.nodes_df = pd.read_sql(nodes_query, self.engine)
        self.edges_df = pd.read_sql(edges_query, self.engine)

    # --- Always handle prospects separately ---
    if prospect_ids is not None:
        print("[load_data] Using provided list of prospect_ids.")
        self.prospect_ids = prospect_ids
        self.prospect_df = pd.DataFrame({"ID": prospect_ids})
    else:
        print("[load_data] Loading prospect_ids from SQL.")
        metadata.reflect(bind=self.engine, schema=self.prospect_schema, only=[self.prospect_table])
        prospect_table = Table(self.prospect_table, metadata, autoload_with=self.engine, schema=self.prospect_schema)

        prospect_query = select(
            literal_column(f"{self.prospect_column}")
        ).select_from(prospect_table)

        self.prospect_df = pd.read_sql(prospect_query, self.engine)
        self.prospect_ids = list(self.prospect_df[self.prospect_column].dropna().unique())

    print(f"[load_data] Nodes: {self.nodes_df.shape}, Edges: {self.edges_df.shape}, Prospects: {len(self.prospect_ids)}")
    
def find_best_fit_clients(self) -> pd.DataFrame:
    """
    For each Prospect, find the closest Client using Dijkstra's algorithm
    on an undirected graph, avoiding UBS Financial Advisors in the path.
    Returns a DataFrame with Prospect -> Best-fit Client mappings, even if no match is found.
    """
    import time
    import networkx as nx
    from heapq import heappush, heappop

    start = time.time()
    print("\n[Find Best Fit Clients] Starting...")

    G: nx.Graph = self.graph  # Ensure this is undirected (not DiGraph)

    # --- Node classification ---
    clients = {n for n, d in G.nodes(data=True) if d.get("entity_type") == "Client"}
    ubs_advisors = {n for n, d in G.nodes(data=True) if d.get("entity_type") == "UBS Financial Advisor"}
    prospects = [n for n in self.prospect_ids if G.has_node(n) and G.nodes[n].get("entity_type") == "Prospect"]

    # Remove self-matching possibilities
    clients = clients - set(prospects)

    self.unfound_prospects = [pid for pid in self.prospect_ids if not G.has_node(pid)]

    print(f"-> [Validation] Length of found prospect list is {len(prospects)}.")
    print(f"-> [Validation] Length of client list is {len(clients)}.")
    print(f"-> [Validation] Length of advisor list is {len(ubs_advisors)}.")
    print(f"[Validation] Length of unfound prospect list is {len(self.unfound_prospects)}.")

    results = []

    for prospect in prospects:
        visited = set()
        heap = [(0, prospect, [])]  # (cumulative_weight, current_node, path_so_far)

        best_client = None
        best_score = float('inf')
        best_path = []

        while heap:
            cum_weight, current, path = heappop(heap)

            if current in visited:
                continue
            visited.add(current)

            if current in ubs_advisors:
                continue  # skip paths through UBS advisors

            new_path = path + [current]

            if current in clients and current != prospect:
                best_client = current
                best_score = cum_weight
                best_path = new_path
                break  # found best reachable client

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
        else:
            # No reachable client found
            results.append({
                "Prospect": prospect,
                "Client": None,
                "Relationship": None,
                "Score": None
            })

    result_df = pd.DataFrame(results)
    print(f"[find_best_fit_clients] Result shape: {result_df.shape}")
    print(f"[find_best_fit_clients] Unfound prospects: {len(self.unfound_prospects)}")
    print(f"[find_best_fit_clients] Completed in {time.time() - start:.2f} seconds")
    return result_df