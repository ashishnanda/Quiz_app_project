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
    coverage_threshold: float = 0.3,
    plot: bool = True,
    random_seed: int = 42,
) -> Tuple[int, int, Dict[int, float]]:
    """
    Recommends Node2Vec walk length and number of walks per node based on graph structure and coverage.

    Prints and plots:
        - Component sizes, diameters, average path lengths
        - Distribution of node degrees
        - Node coverage statistics (mean, median, histogram)
        - Diagnostic information at every step

    Parameters
    ----------
    G : networkx.Graph
        The input graph (can be disconnected).
    max_walk_length : int
        Upper limit for walk length, default 80.
    min_num_walks : int
        Minimum number of walks per node, default 10.
    max_total_samples : int
        Maximum total training samples (nodes x walks x walk_length), default 10 million.
    coverage_threshold : float
        Target mean node coverage (as fraction of graph), default 0.3.
    plot : bool
        Whether to plot graphs at each step, default True.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    walk_length : int
        Recommended walk length.
    num_walks : int
        Recommended number of walks per node.
    node_coverage_map : Dict[int, float]
        Node-wise coverage proportion with final parameters.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # STEP 1: Print and plot basic graph stats
    print(f"[INFO] Number of nodes: {G.number_of_nodes()}")
    print(f"[INFO] Number of edges: {G.number_of_edges()}")

    degrees = [d for _, d in G.degree()]
    print(f"[INFO] Degree: mean={np.mean(degrees):.2f}, median={np.median(degrees)}, min={np.min(degrees)}, max={np.max(degrees)}")

    if plot:
        plt.figure(figsize=(6, 4))
        plt.hist(degrees, bins=30)
        plt.title("Degree Distribution")
        plt.xlabel("Degree")
        plt.ylabel("Number of Nodes")
        plt.show()

    # STEP 2: Connected component stats
    components = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    component_sizes = [len(c) for c in components]

    print(f"[INFO] Number of connected components: {len(components)}")
    print(f"[INFO] Component sizes: min={np.min(component_sizes)}, max={np.max(component_sizes)}, median={np.median(component_sizes)}")

    if plot:
        plt.figure(figsize=(6, 4))
        plt.hist(component_sizes, bins=30)
        plt.title("Component Size Distribution")
        plt.xlabel("Component Size")
        plt.ylabel("Frequency")
        plt.show()

    diameters, avg_paths = [], []
    for i, c in enumerate(components):
        if len(c) > 1:
            try:
                diam = nx.diameter(c)
                ap = nx.average_shortest_path_length(c)
            except Exception:
                diam, ap = np.nan, np.nan
        else:
            diam, ap = 0, 0
        diameters.append(diam)
        avg_paths.append(ap)
        print(f"[Component {i+1}] size={len(c)}, diameter={diam}, avg_path_length={ap}")

    large_comps = [i for i, sz in enumerate(component_sizes) if sz >= 20]
    median_avg_path = np.nanmedian([avg_paths[i] for i in large_comps]) if large_comps else 4
    median_size = int(np.nanmedian([component_sizes[i] for i in large_comps])) if large_comps else 100

    if plot and len(large_comps) > 0:
        plt.figure(figsize=(6, 4))
        plt.hist([avg_paths[i] for i in large_comps], bins=15)
        plt.title("Average Shortest Path Length (Large Components)")
        plt.xlabel("Avg Shortest Path Length")
        plt.ylabel("Frequency")
        plt.show()

    # STEP 3: Initial parameter estimates
    walk_length = min(int(2 * median_avg_path), max_walk_length)
    num_walks = max(min_num_walks, max_total_samples // (G.number_of_nodes() * walk_length))
    print(f"\n[INFO] Initial walk_length={walk_length}, num_walks={num_walks}")

    # STEP 4: Simulate random walks for coverage
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
    coverage_vals = list(node_coverage_map.values())
    avg_coverage = np.mean(coverage_vals)
    med_coverage = np.median(coverage_vals)
    print(f"[INFO] Coverage with walk_length={walk_length}, num_walks={num_walks}: mean={avg_coverage:.2%}, median={med_coverage:.2%}")

    if plot:
        plt.figure(figsize=(6, 4))
        plt.hist(coverage_vals, bins=30)
        plt.title(f"Node Coverage: wl={walk_length}, nw={num_walks}")
        plt.xlabel("Fraction of Graph Visited")
        plt.ylabel("Nodes")
        plt.show()

    # STEP 5: Adjust walk_length if coverage too low
    while avg_coverage < coverage_threshold and walk_length < max_walk_length:
        walk_length += 5
        print(f"\n[INFO] Increasing walk_length to {walk_length} (coverage was {avg_coverage:.2%})")
        node_coverage_map = simulate_walks(G, num_walks, walk_length)
        coverage_vals = list(node_coverage_map.values())
        avg_coverage = np.mean(coverage_vals)
        med_coverage = np.median(coverage_vals)
        print(f"[INFO] Coverage with walk_length={walk_length}: mean={avg_coverage:.2%}, median={med_coverage:.2%}")

        if plot:
            plt.figure(figsize=(6, 4))
            plt.hist(coverage_vals, bins=30)
            plt.title(f"Node Coverage: wl={walk_length}, nw={num_walks}")
            plt.xlabel("Fraction of Graph Visited")
            plt.ylabel("Nodes")
            plt.show()

    print(f"\n[RESULT] Recommended walk_length={walk_length}, num_walks={num_walks}")
    return walk_length, num_walks, node_coverage_map

if __name__ == "__main__":
    # Example usage
    G = nx.erdos_renyi_graph(500, 0.02, seed=42)
    wl, nw, coverage = recommend_node2vec_params(G)