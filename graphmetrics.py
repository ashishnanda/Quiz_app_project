import networkx as nx
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List


def recommend_node2vec_params(
    G: nx.Graph,
    max_walk_length: int = 80,
    min_num_walks: int = 10,
    max_total_samples: int = 10_000_000,
    coverage_threshold: float = 0.3
) -> Tuple[int, int, Dict[int, float]]:
    """
    Recommends Node2Vec walk length and number of walks per node based on graph structure and coverage.

    Parameters:
    -----------
    G : networkx.Graph
        The input graph (can be disconnected).
    max_walk_length : int, optional
        Upper limit for walk length, default is 80.
    min_num_walks : int, optional
        Lower limit for number of walks per node, default is 10.
    max_total_samples : int, optional
        Max number of total samples (nodes × walks × walk_length), default is 10 million.
    coverage_threshold : float, optional
        Desired minimum coverage (as % of graph) by a single node’s random walk, default is 0.3.

    Returns:
    --------
    walk_length : int
        Recommended length of each random walk.
    num_walks : int
        Recommended number of walks per node.
    node_coverage_map : Dict[int, float]
        Mapping from node → % of graph it covers with the recommended parameters.

    Example:
    --------
    >>> G = nx.erdos_renyi_graph(1000, 0.01)
    >>> wl, nw, cov = recommend_node2vec_params(G)
    >>> print(f"Recommended walk length: {wl}, num walks: {nw}")
    """

    def component_stats(component: nx.Graph) -> Tuple[int, float]:
        try:
            avg_path = nx.average_shortest_path_length(component)
            diameter = nx.diameter(component)
            return diameter, avg_path
        except:
            return 1, 1.0  # For very small components

    # Step 1: Component-level stats
    components = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    component_data = []

    for c in components:
        size = len(c)
        diam, avg_path = component_stats(c)
        component_data.append((size, diam, avg_path))

    large_comps = [x for x in component_data if x[0] >= 20]
    median_avg_path = np.median([x[2] for x in large_comps]) if large_comps else 4
    median_size = int(np.median([x[0] for x in large_comps])) if large_comps else 100

    # Step 2: Initial estimate
    walk_length = min(int(2 * median_avg_path), max_walk_length)
    num_walks = max(min_num_walks, max_total_samples // (G.number_of_nodes() * walk_length))

    # Step 3: Simulate random walks to evaluate coverage
    def simulate_walks(G: nx.Graph, num_walks: int, walk_length: int) -> Dict[int, float]:
        walks = defaultdict(set)
        nodes = list(G.nodes())
        for node in nodes:
            for _ in range(num_walks):
                walk = [node]
                curr = node
                for _ in range(walk_length - 1):
                    neighbors = list(G.neighbors(curr))
                    if neighbors:
                        curr = random.choice(neighbors)
                        walk.append(curr)
                    else:
                        break
                walks[node].update(walk)
        coverage_map = {node: len(visited) / G.number_of_nodes() for node, visited in walks.items()}
        return coverage_map

    node_coverage_map = simulate_walks(G, num_walks, walk_length)
    avg_coverage = np.mean(list(node_coverage_map.values()))
    
    print(f"[INFO] Coverage @ walk_length={walk_length}, num_walks={num_walks}: {avg_coverage:.2%}")

    # Optional: adjust walk_length if coverage is too low
    while avg_coverage < coverage_threshold and walk_length < max_walk_length:
        walk_length += 5
        node_coverage_map = simulate_walks(G, num_walks, walk_length)
        avg_coverage = np.mean(list(node_coverage_map.values()))
        print(f"[INFO] Adjusted walk_length={walk_length}, coverage={avg_coverage:.2%}")

    return walk_length, num_walks, node_coverage_map


if __name__ == "__main__":
    # Example usage with a synthetic graph
    G = nx.erdos_renyi_graph(500, 0.02)
    wl, nw, coverage = recommend_node2vec_params(G)
    print(f"\nRecommended walk_length = {wl}, num_walks = {nw}")