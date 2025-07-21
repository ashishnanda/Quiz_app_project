import networkx as nx
import pandas as pd

class GraphBuilder:
    def __init__(self, edges_df: pd.DataFrame):
        self.edges_df = edges_df

    def build_graph(self) -> nx.Graph:
        """
        Constructs a NetworkX graph using only the nodes that appear in the edges table.

        Assumes `edges_df` has two columns: 'source' and 'target'.
        """
        # Step 1: Build undirected graph from edge list
        G = nx.Graph()

        # Step 2: Add edges (and implicitly, nodes from edges)
        for _, row in self.edges_df.iterrows():
            G.add_edge(row['source'], row['target'])

        # Optional: Explicitly restrict to only nodes that appear in 'source' or 'target'
        nodes_in_edges = set(self.edges_df['source']).union(set(self.edges_df['target']))
        G = G.subgraph(nodes_in_edges).copy()  # strictly filter only those nodes

        print(f"[INFO] Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        return G