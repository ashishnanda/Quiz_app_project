import pandas as pd
import networkx as nx
import heapq
from typing import List, Tuple, Optional
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
        self.metadata = MetaData()
        self.nodes_schema_name = nodes_schema_name
        self.edge_schema_name = edge_schema_name
        self.nodes_table_name = nodes_table_name
        self.edge_table_name = edge_table_name
        self.prospect_schema = prospect_schema
        self.prospect_table = prospect_table
        self.prospect_column = prospect_column

        self.nodes_df = None
        self.edges_df = None
        self.prospect_df = None
        self.prospect_ids = []
        self.unfound_prospects = []
        self.graph = nx.Graph()
        self.results = pd.DataFrame()

        self.node_to_component = {}
        self.component_clients = {}
        self.connected_components = []

    def load_data(self,
                  nodes_df: pd.DataFrame = None,
                  edges_df: pd.DataFrame = None,
                  prospect_ids: Optional[List[str]] = None):
        def clean_column_names(df: pd.DataFrame):
            df.columns = [col.replace('"', '').strip() for col in df.columns]
            return df

        if nodes_df is not None and edges_df is not None:
            self.nodes_df = clean_column_names(nodes_df.copy())
            self.edges_df = clean_column_names(edges_df.copy())
        else:
            self.metadata.reflect(bind=self.engine, schema=self.nodes_schema_name, only=[self.nodes_table_name])
            self.metadata.reflect(bind=self.engine, schema=self.edge_schema_name, only=[self.edge_table_name])

            nodes_table = Table(self.nodes_table_name, self.metadata, autoload_with=self.engine,
                                schema=self.nodes_schema_name)
            edges_table = Table(self.edge_table_name, self.metadata, autoload_with=self.engine,
                                schema=self.edge_schema_name)

            nodes_query = select(
                literal_column('"ID"'),
                literal_column('"Label"'),
                literal_column('"entity_type"')
            ).select_from(nodes_table)

            edges_query = select(
                literal_column('"source"'),
                literal_column('"target"'),
                literal_column('"weight"'),
                literal_column('"edge_detail"')
            ).select_from(edges_table)

            self.nodes_df = clean_column_names(pd.read_sql(nodes_query, self.engine))
            self.edges_df = clean_column_names(pd.read_sql(edges_query, self.engine))

        if prospect_ids is not None:
            self.prospect_ids = prospect_ids
            self.prospect_df = pd.DataFrame({self.prospect_column: prospect_ids})
        else:
            self.metadata.reflect(bind=self.engine, schema=self.prospect_schema, only=[self.prospect_table])
            prospect_table = Table(self.prospect_table, self.metadata, autoload_with=self.engine,
                                   schema=self.prospect_schema)

            prospect_query = select(
                literal_column(f'"{self.prospect_column}"')
            ).select_from(prospect_table)

            self.prospect_df = clean_column_names(pd.read_sql(prospect_query, self.engine))
            self.prospect_column = self.prospect_column.replace('"', '').strip()
            self.prospect_ids = list(self.prospect_df[self.prospect_column].dropna().unique())

    def build_graph(self):
        self.graph = nx.Graph()
        for _, row in self.nodes_df.iterrows():
            self.graph.add_node(row["ID"], label=row.get("Label"), entity_type=row.get("entity_type"))
        for _, row in self.edges_df.iterrows():
            self.graph.add_edge(row["source"], row["target"],
                                weight=row.get("weight", 1.0),
                                edge_detail=row.get("edge_detail"))
        self._compute_connected_components()

    def _compute_connected_components(self):
        self.node_to_component = {}
        self.component_clients = {}
        self.connected_components = list(nx.connected_components(self.graph))
        for idx, component in enumerate(self.connected_components):
            for node in component:
                self.node_to_component[node] = idx
            clients = [n for n in component if self.graph.nodes[n].get("entity_type") == "Client"]
            self.component_clients[idx] = clients

    def _find_paths_and_scores(self, G: nx.Graph, prospect: str, client: str, max_depth: int = 5) -> float:
        scores = []
        try:
            for path in nx.all_simple_paths(G, source=prospect, target=client, cutoff=max_depth):
                score = 1.0
                for u, v in zip(path[:-1], path[1:]):
                    edge_data = G.get_edge_data(u, v, default={})
                    score *= edge_data.get("weight", 1.0)
                scores.append(score)
        except nx.NetworkXNoPath:
            return 0.0
        return sum(scores)

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

    def _match_prospect_dijkstra(self, prospect: str, G: nx.Graph, clients: set, advisors: set) -> List[dict]:
        visited = set()
        heap = [(0, prospect, [])]
        best_clients = []
        best_score = float('inf')

        while heap:
            cum_weight, current, path = heapq.heappop(heap)
            if current in visited or current in advisors:
                continue
            visited.add(current)
            new_path = path + [current]

            if current in clients and current != prospect:
                if cum_weight < best_score:
                    best_score = cum_weight
                    best_clients = [(current, new_path)]
                elif cum_weight == best_score:
                    best_clients.append((current, new_path))
                continue

            for neighbor in G.neighbors(current):
                if neighbor not in visited:
                    edge_data = G.get_edge_data(current, neighbor)
                    weight = edge_data.get("weight", 1.0)
                    heapq.heappush(heap, (cum_weight + weight, neighbor, new_path))

        results = []
        for client, path in best_clients:
            relationship = []
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge_data = G.get_edge_data(u, v)
                edge_desc = edge_data.get("edge_detail", "")
                relationship.append(f"{u}->{v}: {edge_desc}")
            results.append({
                "Prospect": prospect,
                "Client": client,
                "Relationship": "; ".join(relationship),
                "Score": best_score
            })

        if not best_clients:
            results.append({
                "Prospect": prospect,
                "Client": None,
                "Relationship": None,
                "Score": None
            })

        return results

    def find_best_fit_clients_dijkstra(self, chunk_size: int = 100) -> pd.DataFrame:
        G = self.graph
        advisors = {n for n, d in G.nodes(data=True) if d.get("entity_type") == "UBS Financial Advisor"}
        prospects = [n for n in self.prospect_ids if G.has_node(n) and G.nodes[n].get("entity_type") == "Prospect"]
        self.unfound_prospects = [pid for pid in self.prospect_ids if not G.has_node(pid)]
        self.results = pd.DataFrame()

        for i in range(0, len(prospects), chunk_size):
            chunk = prospects[i:i + chunk_size]
            chunk_results = []
            for p in chunk:
                comp_id = self.node_to_component.get(p)
                if comp_id is None:
                    continue
                component_clients = set(self.component_clients[comp_id])
                chunk_results.extend(self._match_prospect_dijkstra(p, G, component_clients, advisors))
            self.results = pd.concat([self.results, pd.DataFrame(chunk_results)], ignore_index=True)
        return self.results

    def check_client_connection(self) -> Tuple[List[str], List[str]]:
        G = self.graph.copy()
        advisors = {n for n, d in G.nodes(data=True) if d.get("entity_type") == "UBS Financial Advisor"}
        G.remove_nodes_from(advisors)

        clients = {n for n, d in G.nodes(data=True) if d.get("entity_type") == "Client"}
        reachable_prospects = set()
        for client in clients:
            if G.has_node(client):
                reachable = nx.single_source_shortest_path_length(G, client)
                for node in reachable:
                    if G.nodes[node].get("entity_type") == "Prospect":
                        reachable_prospects.add(node)

        disconnected_prospects = [p for p in self.prospect_ids if p not in reachable_prospects]
        connected_prospects = [p for p in self.prospect_ids if p in reachable_prospects]
        return connected_prospects, disconnected_prospects