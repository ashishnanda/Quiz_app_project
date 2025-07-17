from typing import List, Tuple
import pandas as pd
import networkx as nx
from heapq import heappush, heappop
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

    def load_data(self,
                  nodes_df: pd.DataFrame = None,
                  edges_df: pd.DataFrame = None,
                  prospect_ids: List[str] = None):
        def clean_column_names(df: pd.DataFrame):
            df.columns = [col.replace('"', '').strip() for col in df.columns]
            return df

        if nodes_df is not None and edges_df is not None:
            print("[load_data] Using provided DataFrames for nodes and edges.")
            self.nodes_df = clean_column_names(nodes_df.copy())
            self.edges_df = clean_column_names(edges_df.copy())
        else:
            print("[load_data] Loading nodes and edges from SQL.")
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
            print("[load_data] Using provided list of prospect_ids.")
            self.prospect_ids = prospect_ids
            self.prospect_df = pd.DataFrame({self.prospect_column: prospect_ids})
        else:
            print("[load_data] Loading prospect_ids from SQL.")
            self.metadata.reflect(bind=self.engine, schema=self.prospect_schema, only=[self.prospect_table])
            prospect_table = Table(self.prospect_table, self.metadata, autoload_with=self.engine,
                                   schema=self.prospect_schema)

            prospect_query = select(
                literal_column(f'"{self.prospect_column}"')
            ).select_from(prospect_table)

            self.prospect_df = clean_column_names(pd.read_sql(prospect_query, self.engine))
            self.prospect_column = self.prospect_column.replace('"', '').strip()
            self.prospect_ids = list(self.prospect_df[self.prospect_column].dropna().unique())

        print(f"[load_data] Nodes: {self.nodes_df.shape}, Edges: {self.edges_df.shape}, Prospects: {len(self.prospect_ids)}")

    def build_graph(self):
        print("[build_graph] Building graph...")
        self.graph = nx.Graph()
        for _, row in self.nodes_df.iterrows():
            self.graph.add_node(row["ID"], label=row.get("Label"), entity_type=row.get("entity_type"))
        for _, row in self.edges_df.iterrows():
            self.graph.add_edge(row["source"], row["target"],
                                weight=row.get("weight", 1.0),
                                edge_detail=row.get("edge_detail"))
        print(f"[build_graph] Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

    def _match_prospect(self, prospect: str, G: nx.Graph, clients: set, ubs_advisors: set) -> List[dict]:
        visited = set()
        heap = [(0, prospect, [])]
        best_clients = []
        best_score = float('inf')

        while heap:
            cum_weight, current, path = heappop(heap)
            if current in visited or current in ubs_advisors:
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
                    heappush(heap, (cum_weight + weight, neighbor, new_path))

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

    def find_best_fit_clients(self) -> pd.DataFrame:
        print("[find_best_fit_clients] Running...")
        G = self.graph
        clients = {n for n, d in G.nodes(data=True) if d.get("entity_type") == "Client"}
        ubs_advisors = {n for n, d in G.nodes(data=True) if d.get("entity_type") == "UBS Financial Advisor"}
        prospects = [n for n in self.prospect_ids if G.has_node(n) and G.nodes[n].get("entity_type") == "Prospect"]
        clients = clients - set(prospects)
        self.unfound_prospects = [pid for pid in self.prospect_ids if not G.has_node(pid)]

        results = []
        for p in prospects:
            results.extend(self._match_prospect(p, G, clients, ubs_advisors))

        return pd.DataFrame(results)

    def check_client_connection(self) -> Tuple[List[str], List[str]]:
        print("[check_client_connection] Splitting connected and disconnected prospects...")
        G = self.graph.copy()
        advisors = {n for n, d in G.nodes(data=True) if d.get("entity_type") == "UBS Financial Advisor"}
        G.remove_edges_from([(u, v) for u, v in G.edges if u in advisors or v in advisors])
        clients = {n for n, d in G.nodes(data=True) if d.get("entity_type") == "Client"}

        visited = set()
        queue = list(clients)

        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            queue.extend([n for n in G.neighbors(node) if n not in visited])

        connected = [pid for pid in self.prospect_ids if pid in visited]
        disconnected = [pid for pid in self.prospect_ids if pid not in visited]
        print(f"[check_client_connection] Connected: {len(connected)}, Disconnected: {len(disconnected)}")
        return connected, disconnected