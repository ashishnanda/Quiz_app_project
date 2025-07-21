def recommend_node2vec_params_with_outlier_handling(
    G: nx.Graph,
    component_size_threshold: int = 100,
    max_walk_length: int = 80,
    min_num_walks: int = 10,
    max_total_samples: int = 10_000_000,
    coverage_threshold: float = 0.3,
    plot: bool = True,
    random_seed: int = 42,
) -> Tuple[int, int, Dict[int, float]]:
    """
    Recommends walk length and number of walks for Node2Vec by separating small disconnected components
    from large connected ones to avoid skewing due to outliers.

    Returns recommended parameters based on large components only, and coverage map for large subgraph.

    Parameters
    ----------
    G : networkx.Graph
        The input graph.
    component_size_threshold : int
        Minimum component size to be considered for parameter tuning (default 100).
    ...

    Returns
    -------
    walk_length : int
    num_walks : int
    node_coverage_map : Dict[int, float]
    """
    import numpy as np
    import random
    from collections import defaultdict
    import matplotlib.pyplot as plt
    import networkx as nx

    random.seed(random_seed)
    np.random.seed(random_seed)

    # Get components
    components = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    large_components = [c for c in components if len(c) >= component_size_threshold]
    small_components = [c for c in components if len(c) < component_size_threshold]

    print(f"[INFO] Total components: {len(components)}")
    print(f"[INFO] Large components (≥{component_size_threshold}): {len(large_components)}")
    print(f"[INFO] Small/noisy components (<{component_size_threshold}): {len(small_components)}")

    if not large_components:
        print("[WARNING] No large components found. Using fixed fallback parameters.")
        return 5, 5, {}

    # Combine large components into one subgraph
    G_large = nx.Graph()
    for comp in large_components:
        G_large.add_nodes_from(comp.nodes(data=True))
        G_large.add_edges_from(comp.edges(data=True))

    degrees = [d for _, d in G_large.degree()]
    median_degree = np.median(degrees)
    if plot:
        plt.hist(degrees, bins=50)
        plt.title("Degree Distribution (Large Components)")
        plt.xlabel("Degree")
        plt.ylabel("Number of Nodes")
        plt.show()

    # Estimate walk_length based on average path length of large components
    diameters, avg_paths = [], []
    for comp in large_components:
        try:
            if len(comp) > 1:
                diam = nx.diameter(comp)
                ap = nx.average_shortest_path_length(comp)
            else:
                diam, ap = 1, 1
        except:
            diam, ap = 1, 1
        diameters.append(diam)
        avg_paths.append(ap)

    walk_length = min(int(2 * np.median(avg_paths)), max_walk_length)
    num_walks = max(min_num_walks, max_total_samples // (G_large.number_of_nodes() * walk_length))
    print(f"[INFO] Estimated walk_length = {walk_length}, num_walks = {num_walks}")

    # Coverage Simulation
    def simulate_walks(G: nx.Graph, num_walks: int, walk_length: int) -> Dict[int, float]:
        walks = defaultdict(set)
        for node in G.nodes():
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
        return {node: len(visited) / G.number_of_nodes() for node, visited in walks.items()}

    coverage_map = simulate_walks(G_large, num_walks, walk_length)
    coverage_vals = list(coverage_map.values())
    avg_cov = np.mean(coverage_vals)
    print(f"[INFO] Initial Coverage: mean = {avg_cov:.2%}")

    if plot:
        plt.hist(coverage_vals, bins=30)
        plt.title("Node Coverage in Large Components")
        plt.xlabel("Fraction of Graph Visited")
        plt.ylabel("Nodes")
        plt.show()

    return walk_length, num_walks, coverage_map