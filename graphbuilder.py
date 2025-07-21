import networkx as nx
import pandas as pd

class GraphBuilder:
    def __init__(self, edges_df: pd.DataFrame, nodes_df: pd.DataFrame):
        """
        Parameters:
        -----------
        edges_df : pd.DataFrame
            Must contain 'source' and 'target' columns.
        nodes_df : pd.DataFrame
            Must contain a 'node_id' column representing valid nodes.
        """
        self.edges_df = edges_df
        self.nodes_df = nodes_df

    def build_graph(self) -> nx.Graph:
        """
        Constructs a graph using only edges where both source and target are in the nodes table.

        Returns:
        --------
        G : nx.Graph
            A filtered undirected graph.
        """
        valid_nodes = set(self.nodes_df['node_id'])

        # Filter edges where both nodes are in valid node list
        filtered_edges = self.edges_df[
            self.edges_df['source'].isin(valid_nodes) & self.edges_df['target'].isin(valid_nodes)
        ]

        # Build graph
        G = nx.Graph()
        for _, row in filtered_edges.iterrows():
            G.add_edge(row['source'], row['target'])

        print(f"[INFO] Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        return G