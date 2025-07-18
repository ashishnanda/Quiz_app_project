# prospect_matcher.py
import pandas as pd
import networkx as nx
from typing import List, Tuple, Optional, Dict, Any
from sqlalchemy import MetaData, Table, select, literal_column

class ProspectMatcher:
    def __init__(self,
                 engine,
                 nodes_schema_name: str,
                 edge_schema_name: str,
                 nodes_table_name: str,
                 edge_table_name: str,
                 prospect_schema: str,
                 prospect_table: str,
                 prospect_column: str):

        self.engine = engine
        self.nodes_schema_name = nodes_schema_name
        self.edge_schema_name = edge_schema_name
        self.nodes_table_name = nodes_table_name
        self.edge_table_name = edge_table_name
        self.prospect_schema = prospect_schema
        self.prospect_table = prospect_table
        self.prospect_column = prospect_column

        self.graph = None
        self.node_to_component = {}
        self.component_clients = {}
        self.prospect_ids = []
        self.unfound_prospects = []
        self.results = pd.DataFrame()

    def load_data(self,
                  nodes_df: Optional[pd.DataFrame] = None,
                  edges_df: Optional[pd.DataFrame] = None,
                  prospect_ids: Optional[List[str]] = None):

        if nodes_df is None:
            query = f'SELECT * FROM "{self.nodes_schema_name}"."{self.nodes_table_name}"'
            nodes_df = pd.read_sql(query, self.engine)
        if edges_df is None:
            query = f'SELECT * FROM "{self.edge_schema_name}"."{self.edge_table_name}"'
            edges_df = pd.read_sql(query, self.engine)

        nodes_df.columns = [col.strip('"') for col in nodes_df.columns]
        edges_df.columns = [col.strip('"') for col in edges_df.columns]

        self.nodes_df = nodes_df
        self.edges_df = edges_df

        if prospect_ids is not None:
            self.prospect_ids = prospect_ids
        else:
            query = f'SELECT DISTINCT "{self.prospect_column}" FROM "{self.prospect_schema}"."{self.prospect_table}"'
            df = pd.read_sql(query, self.engine)
            col = df.columns[0]
            self.prospect_ids = df[col].dropna().astype(str).tolist()

    def build_graph(self):
        self.graph = nx.Graph()
        for _, row in self.nodes_df.iterrows():
            self.graph.add_node(row['ID'], **row.to_dict())

        for _, row in self.edges_df.iterrows():
            self.graph.add_edge(row['source'], row['target'], weight=row.get('weight', 1.0))

        self._compute_connected_components()

    def _compute_connected_components(self):
        self.node_to_component = {}
        self.component_clients = {}
        for idx, component in enumerate(nx.connected_components(self.graph)):
            clients = [n for n in component if self.graph.nodes[n].get("entity_type") == "Client"]
            for node in component:
                self.node_to_component[node] = idx
            self.component_clients[idx] = clients

    def find_best_fit_clients_multiplicative(self, max_depth: int = 5, chunk_size: int = 100) -> pd.DataFrame:
        G = self.graph
        self.results = pd.DataFrame()
        prospects = [p for p in self.prospect_ids if G.has_node(p) and G.nodes[p].get("entity_type") == "Prospect"]
        self.unfound_prospects = [p for p in self.prospect_ids if not G.has_node(p)]

        for i in range(0, len(prospects), chunk_size):
            chunk = prospects[i:i + chunk_size]
            chunk_results = []
            for prospect in chunk:
                comp_id = self.node_to_component.get(prospect)
                if comp_id is None:
                    continue
                clients = self.component_clients.get(comp_id, [])

                # Run BFS once
                try:
                    bfs_levels = nx.single_source_shortest_path_length(G, prospect, cutoff=max_depth)
                except nx.NetworkXError:
                    continue

                found = False
                for k in range(2, max_depth + 1):
                    level_clients = [n for n, dist in bfs_levels.items() if dist == k and G.nodes[n].get("entity_type") == "Client"]
                    scores = {}
                    for client in level_clients:
                        try:
                            for path in nx.all_simple_paths(G, source=prospect, target=client, cutoff=k):
                                score = 1.0
                                for i in range(len(path) - 1):
                                    edge_data = G.get_edge_data(path[i], path[i+1])
                                    score *= edge_data.get('weight', 1.0)
                                scores.setdefault(client, 0.0)
                                scores[client] += score
                        except nx.NetworkXNoPath:
                            continue

                    if scores:
                        max_score = max(scores.values())
                        best_clients = [client for client, score in scores.items() if score == max_score]
                        for bc in best_clients:
                            chunk_results.append({
                                "Prospect": prospect,
                                "Client": bc,
                                "Score": scores[bc],
                                "Depth": k
                            })
                        found = True
                        break  # stop at first successful depth

            self.results = pd.concat([self.results, pd.DataFrame(chunk_results)], ignore_index=True)
        return self.results
