from dataclasses import dataclass
from typing import Any, Optional, Union, Tuple, List
import pandas as pd
import networkx as nx
import logging
import time
import re
from sqlalchemy import text, create_engine  # ensure SQLAlchemy is installed

# ----------------------------------------
# SQLSource dataclass
# ----------------------------------------
@dataclass
class SQLSource:
    """
    Describes a SQL table to read.
    """
    table_name: str
    schema: Optional[str] = None
    conn: Any = None  # SQLAlchemy Engine or DBAPI connection

# ----------------------------------------
# ExperimentConfig dataclass
# ----------------------------------------
@dataclass
class ExperimentConfig:
    # Data ingestion
    nodes_source: Union[pd.DataFrame, SQLSource]
    edges_source: Union[pd.DataFrame, SQLSource]
    node_columns: List[str]
    edge_columns: List[str]
    component_selector: Any           # instance implementing .select(list_of_graphs)
    models: List[Any]

    # Subsetting (placeholders)
    sample_size: Optional[int] = None
    node_id_table: Optional[SQLSource] = None
    node_id_column: Optional[str] = None
    node_id_prefix: Optional[str] = None
    k_hops: int = 0
    node_id_schema: Optional[str] = None

    # Summary (placeholders)
    summary_column: str = 'entity_type'
    summary_sort_by: str = 'count'

    # Logging & resources
    verbose: bool = True
    log_level: str = 'INFO'
    free_memory: bool = True

# ----------------------------------------
# Timer utility
# ----------------------------------------
class Timer:
    """Simple timer for named steps."""
    def __init__(self):
        self._starts = {}
        self._elapsed = {}

    def start(self, key: str):
        self._starts[key] = time.time()

    def stop(self, key: str) -> float:
        elapsed = time.time() - self._starts.get(key, time.time())
        self._elapsed[key] = elapsed
        return elapsed

    def elapsed(self, key: str) -> float:
        return self._elapsed.get(key, 0.0)

# ----------------------------------------
# GraphExperiment with ingestion
# ----------------------------------------
class GraphExperiment:
    """
    Pipeline with implemented ingestion (SQL & DataFrame support).
    """
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        logging.basicConfig(level=cfg.log_level)
        self.logger = logging.getLogger('GraphExperiment')
        self._nodes_df_cache: Optional[pd.DataFrame] = None
        self._edges_df_cache: Optional[pd.DataFrame] = None
        self.timer = Timer()

    def clear_cache(self):
        """Clear cached DataFrames so next ingest is fresh."""
        self._nodes_df_cache = None
        self._edges_df_cache = None
        self.logger.info("Cleared nodes and edges cache")

    def _ingest(self, refresh: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load nodes & edges DataFrames with:
        - Caching and optional refresh
        - Strict column checks (fail-fast)
        - SQL SELECT-based reads
        - Type normalization (IDs to str, whitespace -> underscores)
        - Logging of shapes and errors

        Args:
            refresh: Force re-loading from source even if cached.
        Returns:
            Tuple of (nodes_df, edges_df).
        """
        # Nodes
        if not refresh and self._nodes_df_cache is not None:
            self.logger.info("Using cached nodes DataFrame")
            nodes_df = self._nodes_df_cache
        else:
            nodes_df = self._load_source(
                source=self.cfg.nodes_source,
                cols=self.cfg.node_columns,
                source_name="nodes"
            )
            self._nodes_df_cache = nodes_df

        # Edges
        if not refresh and self._edges_df_cache is not None:
            self.logger.info("Using cached edges DataFrame")
            edges_df = self._edges_df_cache
        else:
            edges_df = self._load_source(
                source=self.cfg.edges_source,
                cols=self.cfg.edge_columns,
                source_name="edges"
            )
            self._edges_df_cache = edges_df

        return nodes_df, edges_df

    def _load_source(
        self,
        source: Union[pd.DataFrame, SQLSource],
        cols: List[str],
        source_name: str
    ) -> pd.DataFrame:
        """
        Helper to load a DataFrame from pandas DF or SQLSource via SELECT.
        Enforces strict column presence and ID normalization.

        Raises:
            KeyError: if any requested column is missing.
            ValueError: if source type is unsupported.
        """
        self.logger.info(f"Loading {source_name} with columns: {cols}")

        # 1. Read data
        if isinstance(source, pd.DataFrame):
            df = source.copy()
        elif isinstance(source, SQLSource):
            # Build SELECT statement with case-sensitive quoting
            quoted_cols = ', '.join(f'"{c}"' for c in cols)
            schema_prefix = f'"{source.schema}".' if source.schema else ''
            table_quoted = f'{schema_prefix}"{source.table_name}"'
            sql = text(f"SELECT {quoted_cols} FROM {table_quoted}")
            try:
                df = pd.read_sql_query(sql, source.conn)
            except Exception as e:
                self.logger.error(f"SQL load failed for {source_name}: {e}")
                raise

            # Remove any lingering quotes in column names
            df.columns = [re.sub(r'^"|"$', '', col) for col in df.columns]
        else:
            msg = f"Unsupported source type for {source_name}: {type(source)}"
            self.logger.error(msg)
            raise ValueError(msg)

        # 2. Strict column check
        missing = [c for c in cols if c not in df.columns]
        if missing:
            msg = f"{source_name} missing columns: {missing}"
            self.logger.error(msg)
            raise KeyError(msg)

        # 3. Normalize ID-like columns
        for id_col in ['id', 'source', 'target']:
            if id_col in df.columns:
                df[id_col] = (
                    df[id_col]
                    .astype(str)
                    .str.replace(r'\s+', '_', regex=True)
                )

        self.logger.info(f"Loaded {source_name}: {df.shape[0]} rows x {df.shape[1]} cols")
        return df