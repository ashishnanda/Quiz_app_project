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

        # Remove double quotes if present in column names
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

    def _find_paths_and_scores(self, G, source, target, max_depth=5):
        total_score = 0.0
        try:
            for path in nx.all_simple_paths(G, source=source, target=target, cutoff=max_depth):
                score = 1.0
                for i in range(len(path) - 1):
                    edge_data = G.get_edge_data(path[i], path[i+1])
                    score *= edge_data.get('weight', 1.0)
                total_score += score
        except nx.NetworkXNoPath:
            pass
        return total_score

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
                best_score = 0
                best_client = None
                for client in self.component_clients[comp_id]:
                    if prospect == client:
                        continue
                    score = self._find_paths_and_scores(G, prospect, client, max_depth=max_depth)
                    if score > best_score:
                        best_score = score
                        best_client = client
                chunk_results.append({
                    "Prospect": prospect,
                    "Client": best_client,
                    "Score": best_score
                })
            self.results = pd.concat([self.results, pd.DataFrame(chunk_results)], ignore_index=True)
        return self.results

    def _match_prospect_dijkstra(self, prospect, G, clients, advisors):
        results = []
        try:
            lengths = nx.single_source_dijkstra_path_length(G, prospect)
        except:
            return results

        best_score = float("inf")
        best_client = None
        for client in clients:
            if client in lengths:
                score = lengths[client]
                if score < best_score:
                    best_score = score
                    best_client = client
        if best_client:
            results.append({"Prospect": prospect, "Client": best_client, "Score": best_score})
        return results

    def find_best_fit_clients_dijkstra(self, chunk_size: int = 100) -> pd.DataFrame:
        G = self.graph
        self.results = pd.DataFrame()
        clients = {n for n, d in G.nodes(data=True) if d.get("entity_type") == "Client"}
        advisors = {n for n, d in G.nodes(data=True) if d.get("entity_type") == "UBS Financial Advisor"}
        prospects = [n for n in self.prospect_ids if G.has_node(n) and G.nodes[n].get("entity_type") == "Prospect"]
        clients = clients - set(prospects)
        self.unfound_prospects = [pid for pid in self.prospect_ids if not G.has_node(pid)]

        for i in range(0, len(prospects), chunk_size):
            chunk = prospects[i:i + chunk_size]
            chunk_results = []
            for p in chunk:
                chunk_results.extend(self._match_prospect_dijkstra(p, G, clients, advisors))
            self.results = pd.concat([self.results, pd.DataFrame(chunk_results)], ignore_index=True)
        return self.results

    def check_client_connection(self) -> Tuple[List[str], List[str]]:
        G = self.graph.copy()
        advisors = {n for n, d in G.nodes(data=True) if d.get("entity_type") == "UBS Financial Advisor"}
        G.remove_nodes_from(advisors)
        clients = [n for n, d in G.nodes(data=True) if d.get("entity_type") == "Client"]

        reachable_nodes = set()
        if clients:
            reachable_nodes = set(nx.multi_source_dijkstra_path_length(G, clients).keys())

        reachable_prospects = [p for p in self.prospect_ids if p in reachable_nodes]
        disconnected_prospects = [p for p in self.prospect_ids if p not in reachable_nodes]
        return reachable_prospects, disconnected_prospects
