# AI Enterprise Data Analyst - Data Lineage Engine
# Production-grade data lineage tracking
# Handles: column relationships, transformation tracking, impact analysis

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import pandas as pd

try:
    from app.core.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


# ============================================================================
# Enums
# ============================================================================

class OperationType(str, Enum):
    """Types of data operations."""
    LOAD = "load"
    TRANSFORM = "transform"
    FILTER = "filter"
    AGGREGATE = "aggregate"
    JOIN = "join"
    DERIVE = "derive"
    DROP = "drop"
    RENAME = "rename"


class RelationshipType(str, Enum):
    """Types of column relationships."""
    DERIVED = "derived"
    COPY = "copy"
    RENAMED = "renamed"
    AGGREGATED = "aggregated"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class LineageNode:
    """Node in lineage graph."""
    node_id: str
    name: str
    node_type: str  # 'source', 'column', 'dataset', 'transformation'
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class LineageEdge:
    """Edge in lineage graph."""
    source_id: str
    target_id: str
    relationship: RelationshipType
    operation: OperationType
    description: str = ""


@dataclass
class DataLineageResult:
    """Complete lineage tracking result."""
    n_nodes: int = 0
    n_edges: int = 0
    
    nodes: Dict[str, LineageNode] = field(default_factory=dict)
    edges: List[LineageEdge] = field(default_factory=list)
    
    # Derived info
    source_columns: List[str] = field(default_factory=list)
    derived_columns: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "nodes": {
                nid: {
                    "name": n.name,
                    "type": n.node_type
                }
                for nid, n in self.nodes.items()
            },
            "edges": [
                {
                    "source": e.source_id,
                    "target": e.target_id,
                    "relationship": e.relationship.value
                }
                for e in self.edges
            ],
            "source_columns": self.source_columns,
            "derived_columns": self.derived_columns
        }


# ============================================================================
# Data Lineage Engine
# ============================================================================

class DataLineageEngine:
    """
    Production-grade Data Lineage engine.
    
    Features:
    - Column-level lineage tracking
    - Transformation logging
    - Impact analysis
    - Upstream/downstream tracing
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.nodes: Dict[str, LineageNode] = {}
        self.edges: List[LineageEdge] = []
    
    def register_source(
        self,
        name: str,
        columns: List[str],
        metadata: Dict[str, Any] = None
    ) -> str:
        """Register a data source."""
        source_id = self._generate_id(f"source_{name}")
        
        self.nodes[source_id] = LineageNode(
            node_id=source_id,
            name=name,
            node_type="source",
            metadata=metadata or {}
        )
        
        # Register columns
        for col in columns:
            col_id = self._generate_id(f"{name}_{col}")
            self.nodes[col_id] = LineageNode(
                node_id=col_id,
                name=col,
                node_type="column",
                metadata={"source": name}
            )
            
            self.edges.append(LineageEdge(
                source_id=source_id,
                target_id=col_id,
                relationship=RelationshipType.DERIVED,
                operation=OperationType.LOAD,
                description=f"Column {col} from source {name}"
            ))
        
        if self.verbose:
            logger.info(f"Registered source: {name} with {len(columns)} columns")
        
        return source_id
    
    def log_transformation(
        self,
        operation: OperationType,
        source_columns: List[str],
        target_columns: List[str],
        description: str = ""
    ):
        """Log a transformation."""
        for source in source_columns:
            source_id = self._find_node_id(source)
            
            for target in target_columns:
                target_id = self._generate_id(f"derived_{target}")
                
                if target_id not in self.nodes:
                    self.nodes[target_id] = LineageNode(
                        node_id=target_id,
                        name=target,
                        node_type="column",
                        metadata={"derived": True}
                    )
                
                self.edges.append(LineageEdge(
                    source_id=source_id or source,
                    target_id=target_id,
                    relationship=RelationshipType.DERIVED,
                    operation=operation,
                    description=description
                ))
        
        if self.verbose:
            logger.info(f"Logged transformation: {operation.value}")
    
    def log_rename(
        self,
        old_name: str,
        new_name: str
    ):
        """Log a column rename."""
        source_id = self._find_node_id(old_name)
        target_id = self._generate_id(f"renamed_{new_name}")
        
        self.nodes[target_id] = LineageNode(
            node_id=target_id,
            name=new_name,
            node_type="column",
            metadata={"renamed_from": old_name}
        )
        
        self.edges.append(LineageEdge(
            source_id=source_id or old_name,
            target_id=target_id,
            relationship=RelationshipType.RENAMED,
            operation=OperationType.RENAME
        ))
    
    def log_aggregation(
        self,
        source_columns: List[str],
        target_column: str,
        aggregation: str
    ):
        """Log an aggregation."""
        target_id = self._generate_id(f"agg_{target_column}")
        
        self.nodes[target_id] = LineageNode(
            node_id=target_id,
            name=target_column,
            node_type="column",
            metadata={"aggregation": aggregation}
        )
        
        for source in source_columns:
            source_id = self._find_node_id(source)
            
            self.edges.append(LineageEdge(
                source_id=source_id or source,
                target_id=target_id,
                relationship=RelationshipType.AGGREGATED,
                operation=OperationType.AGGREGATE,
                description=f"{aggregation} of {source}"
            ))
    
    def get_upstream(self, column: str) -> List[str]:
        """Get upstream columns."""
        node_id = self._find_node_id(column)
        if not node_id:
            return []
        
        upstream = []
        visited = set()
        
        def trace(nid: str):
            if nid in visited:
                return
            visited.add(nid)
            
            for edge in self.edges:
                if edge.target_id == nid:
                    if edge.source_id in self.nodes:
                        upstream.append(self.nodes[edge.source_id].name)
                    trace(edge.source_id)
        
        trace(node_id)
        return list(set(upstream))
    
    def get_downstream(self, column: str) -> List[str]:
        """Get downstream columns."""
        node_id = self._find_node_id(column)
        if not node_id:
            return []
        
        downstream = []
        visited = set()
        
        def trace(nid: str):
            if nid in visited:
                return
            visited.add(nid)
            
            for edge in self.edges:
                if edge.source_id == nid:
                    if edge.target_id in self.nodes:
                        downstream.append(self.nodes[edge.target_id].name)
                    trace(edge.target_id)
        
        trace(node_id)
        return list(set(downstream))
    
    def impact_analysis(self, column: str) -> Dict[str, Any]:
        """Analyze impact of changes to a column."""
        downstream = self.get_downstream(column)
        
        return {
            "column": column,
            "directly_impacts": [d for d in downstream if d != column][:10],
            "total_impacted": len(downstream),
            "risk_level": "high" if len(downstream) > 5 else "medium" if len(downstream) > 0 else "low"
        }
    
    def get_lineage(self) -> DataLineageResult:
        """Get complete lineage."""
        source_cols = [
            n.name for n in self.nodes.values()
            if n.node_type == "column" and not n.metadata.get("derived")
        ]
        
        derived_cols = [
            n.name for n in self.nodes.values()
            if n.node_type == "column" and n.metadata.get("derived")
        ]
        
        return DataLineageResult(
            n_nodes=len(self.nodes),
            n_edges=len(self.edges),
            nodes=self.nodes.copy(),
            edges=self.edges.copy(),
            source_columns=source_cols,
            derived_columns=derived_cols
        )
    
    def _generate_id(self, name: str) -> str:
        """Generate unique ID."""
        return hashlib.md5(name.encode()).hexdigest()[:12]
    
    def _find_node_id(self, name: str) -> Optional[str]:
        """Find node ID by name."""
        for nid, node in self.nodes.items():
            if node.name == name:
                return nid
        return None


# ============================================================================
# Factory Functions
# ============================================================================

def get_lineage_engine() -> DataLineageEngine:
    """Get data lineage engine."""
    return DataLineageEngine()


def track_dataframe_lineage(
    df: pd.DataFrame,
    source_name: str = "input"
) -> DataLineageEngine:
    """Create lineage tracker from DataFrame."""
    engine = DataLineageEngine(verbose=False)
    engine.register_source(source_name, df.columns.tolist())
    return engine
