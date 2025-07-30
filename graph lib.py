from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Literal
import pandas as pd
import networkx as nx
import logging
import time
import numpy as np

# ----------------------------------------
# Configuration and result data classes
# ----------------------------------------

@dataclass
class ExperimentConfig:
    # Data ingestion
    nodes_source: Any                  # e.g. SQLTable spec or DataFrame
    edges_source: Any                  # e.g. SQLTable spec or DataFrame
    node_columns: List[str]
    edge_columns: List[str]
    component_selector: Any            # instance implementing .select(list_of_graphs)
    models: List[Any]                  # list of ModelSpec

    # Subsetting (mutually exclusive options)
    sample_size: Optional[int] = None
    node_id_table: Optional[Any] = None   # e.g. SQLTable spec
    node_id_column: Optional[str] = None
    node_id_prefix: Optional[str] = None
    k_hops: int = 0
    node_id_schema: Optional[str] = None  # e.g. 'p{}' to format IDs

    # Summary
    summary_column: str = 'entity_type'
    summary_sort_by: Literal['count', 'value'] = 'count'

    # Logging & resources
    verbose: bool = True
    log_level: str = 'INFO'
    free_memory: bool = True

@dataclass
class ModelSpec:
    name: str
    variant: str
    params: Dict[str, Any]
    use_edge_weights: bool = False

@dataclass
class ExperimentResult:
    graph_id: str
    model_name: str
    run_timestamp: float
    embeddings: pd.DataFrame
    nearest_neighbors: pd.DataFrame
    metrics: Dict[str, float]

# ----------------------------------------
# Utility classes
# ----------------------------------------

class Timer:
    """Simple timer for named steps."""
    def __init__(self):
        self._starts = {}
        self._elapsed = {}

    def start(self, key: str):
        self._starts[key] = time.time()

    def stop(self, key: str) -> float:
        if key in self._starts:
            elapsed = time.time() - self._starts[key]
            self._elapsed[key] = elapsed
            return elapsed
        return 0.0

    def elapsed(self, key: str) -> float:
        return self._elapsed.get(key, 0.0)

def compute_nearest_neighbors(embeddings: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Compute top-n cosine-similar neighbors for each node.
    Returns a DataFrame with columns: node_id, top1_id, top1_score, ..., topN_id, topN_score.
    """
    mat = embeddings.values
    normed = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    scores = normed @ normed.T
    np.fill_diagonal(scores, -np.inf)
    idx = np.argsort(-scores, axis=1)[:, :top_n]
    result = []
    node_ids = embeddings.index.tolist()
    for i, neighbors in enumerate(idx):
        row = {'node_id': node_ids[i]}
        for j, nbr in enumerate(neighbors, 1):
            row[f'top{j}_id'] = node_ids[nbr]
            row[f'top{j}_score'] = float(scores[i, nbr])
        result.append(row)
    return pd.DataFrame(result)

# ----------------------------------------
# Component selector example
# ----------------------------------------

class ComponentSelector:
    """Interface: implement .select(list_of_subgraphs) -> list_of_subgraphs."""
    def select(self, subgraphs: List[nx.Graph]) -> List[nx.Graph]:
        raise NotImplementedError

class SizeThreshold(ComponentSelector):
    def __init__(self, min_nodes: int):
        self.min_nodes = min_nodes

    def select(self, subgraphs: List[nx.Graph]) -> List[nx.Graph]:
        return [g for g in subgraphs if g.number_of_nodes() >= self.min_nodes]

# ----------------------------------------
# Model wrappers (stubs)
# ----------------------------------------

class BaseModelWrapper:
    def __init__(self, params: Dict[str, Any], use_edge_weights: bool):
        self.params = params
        self.use_edge_weights = use_edge_weights

    def train_one_epoch(self, graph: nx.Graph) -> float:
        raise NotImplementedError

    def get_embeddings(self) -> pd.DataFrame:
        raise NotImplementedError

    def free_memory(self):
        pass

class Node2VecWrapper(BaseModelWrapper):
    def train_one_epoch(self, graph: nx.Graph) -> float:
        # implement skip-gram with/without negative sampling
        return 0.0

    def get_embeddings(self) -> pd.DataFrame:
        # return DataFrame indexed by node IDs
        return pd.DataFrame()

class GINContrastiveWrapper(BaseModelWrapper):
    def train_one_epoch(self, graph: nx.Graph) -> float:
        # two-view InfoNCE loss
        return 0.0

    def get_embeddings(self) -> pd.DataFrame:
        return pd.DataFrame()

class GINEdgePredWrapper(BaseModelWrapper):
    def train_one_epoch(self, graph: nx.Graph) -> float:
        # edge prediction BCE loss
        return 0.0

    def get_embeddings(self) -> pd.DataFrame:
        return pd.DataFrame()

class GINAutoencoderWrapper(BaseModelWrapper):
    def train_one_epoch(self, graph: nx.Graph) -> float:
        # reconstruction MSE loss
        return 0.0

    def get_embeddings(self) -> pd.DataFrame:
        return pd.DataFrame()

class GINHop2Wrapper(BaseModelWrapper):
    def train_one_epoch(self, graph: nx.Graph) -> float:
        # 2-hop adjacency BCE loss
        return 0.0

    def get_embeddings(self) -> pd.DataFrame:
        return pd.DataFrame()

# ----------------------------------------
# Model factory
# ----------------------------------------

class ModelFactory:
    @staticmethod
    def create(name: str, variant: str, params: Dict[str, Any], use_edge_weights: bool):
        if name == 'node2vec':
            if variant == 'nosample':
                return Node2VecWrapper({**params, 'negative_sampling': False}, use_edge_weights)
            return Node2VecWrapper({**params, 'negative_sampling': True}, use_edge_weights)
        elif name == 'gin':
            if variant == 'contrastive':
                return GINContrastiveWrapper(params, use_edge_weights)
            elif variant == 'edge_pred':
                return GINEdgePredWrapper(params, use_edge_weights)
            elif variant == 'autoencoder':
                return GINAutoencoderWrapper(params, use_edge_weights)
            elif variant == 'hop2':
                return GINHop2Wrapper(params, use_edge_weights)
        raise ValueError(f"Unknown model variant: {name}/{variant}")

# ----------------------------------------
# GraphExperiment class
# ----------------------------------------

class GraphExperiment:
    """
    Pipeline for loading graphs, preprocessing, running models, and analyzing results.
    """
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        logging.basicConfig(level=cfg.log_level)
        self.logger = logging.getLogger('GraphExperiment')
        self.timer = Timer()

    def run_all(self) -> List[ExperimentResult]:
        components = self.prepare()
        return self.execute(components)

    def prepare(self) -> List[nx.Graph]:
        # Ingest
        self.timer.start("ingest")
        nodes_df, edges_df = self._ingest()
        self.timer.stop("ingest")
        self._log_status("Ingest complete", "ingest")

        # Subset
        self.timer.start("subset")
        nodes_sub, edges_sub = self._subset(nodes_df, edges_df)
        self.timer.stop("subset")
        self._log_status("Subset complete", "subset")

        # Summarize
        self._summarize(nodes_sub, edges_sub)

        # Split
        return self._split_components(nodes_sub, edges_sub)

    def execute(self, components: List[nx.Graph]) -> List[ExperimentResult]:
        results = []
        for idx, graph in enumerate(components):
            graph_id = f"component_{idx}"
            for spec in self.cfg.models:
                results.append(self._run_model(graph_id, graph, spec))
        return results

    def _ingest(self):
        # TODO: replace with SQL reads or provided DataFrame logic
        nodes_df = pd.DataFrame()
        edges_df = pd.DataFrame()
        return nodes_df, edges_df

    def _subset(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame):
        # TODO: implement sampling, table-based IDs, k-hop expansion
        return nodes_df, edges_df

    def _summarize(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame):
        self.logger.info(f"nodes_df shape: {nodes_df.shape}, edges_df shape: {edges_df.shape}")
        for col in nodes_df.columns:
            non_null = nodes_df[col].notna().sum()
            nulls = nodes_df[col].isna().sum()
            zeros = (nodes_df[col] == 0).sum()
            empties = (nodes_df[col] == "").sum()
            self.logger.debug(f"{col}: non-null={non_null}, nulls={nulls}, zeros={zeros}, empties={empties}")
        vc = nodes_df[self.cfg.summary_column].value_counts()
        df_vc = vc.reset_index().rename(columns={'index': 'value', self.cfg.summary_column: 'count'})
        if self.cfg.summary_sort_by == 'value':
            df_vc = df_vc.sort_values('value')
        else:
            df_vc = df_vc.sort_values('count', ascending=False)
        self.logger.info(f"Summary counts:\n{df_vc.head()}")

    def _split_components(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> List[nx.Graph]:
        G = nx.Graph()
        # TODO: build graph from edges_df
        subgraphs = list(nx.connected_components(G))
        # convert node sets back to Graph objects if needed
        return self.cfg.component_selector.select([G.subgraph(c).copy() for c in subgraphs])

    def _run_model(self, graph_id: str, graph: nx.Graph, spec: ModelSpec) -> ExperimentResult:
        self.logger.info(f"Running {spec.name}/{spec.variant} on {graph_id}")
        start_all = time.time()
        model = ModelFactory.create(spec.name, spec.variant, spec.params, spec.use_edge_weights)
        losses, times = [], []
        for epoch in range(1, spec.params.get('epochs', 1) + 1):
            t0 = time.time()
            loss = model.train_one_epoch(graph)
            losses.append(loss)
            times.append(time.time() - t0)
            self.logger.debug(f"{graph_id}|{spec.name}|epoch={epoch}|loss={loss:.4f}")
        total_time = time.time() - start_all
        avg_time = sum(times) / len(times) if times else 0.0

        embeddings = model.get_embeddings()
        nn_table = compute_nearest_neighbors(embeddings, top_n=spec.params.get('top_n', 5))

        if self.cfg.free_memory:
            model.free_memory()

        return ExperimentResult(
            graph_id=graph_id,
            model_name=f"{spec.name}/{spec.variant}",
            run_timestamp=time.time(),
            embeddings=embeddings,
            nearest_neighbors=nn_table,
            metrics={
                'total_time_sec': total_time,
                'avg_epoch_time_sec': avg_time,
                'final_loss': losses[-1] if losses else 0.0
            }
        )

    def _log_status(self, message: str, timer_key: str):
        elapsed = self.timer.elapsed(timer_key)
        self.logger.info(f"{message} (took {elapsed:.2f}s)")
