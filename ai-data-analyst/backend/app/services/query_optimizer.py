# AI Enterprise Data Analyst - Query Optimizer
# SQL query optimization and intelligent query planning

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import re

import pandas as pd

from app.core.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Query Types
# ============================================================================

class QueryType(str, Enum):
    """SQL query types."""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    CREATE = "create"
    DROP = "drop"
    ALTER = "alter"


class JoinType(str, Enum):
    """Join types."""
    INNER = "inner"
    LEFT = "left"
    RIGHT = "right"
    FULL = "full"
    CROSS = "cross"


@dataclass
class QueryPlan:
    """Query execution plan."""
    
    original_query: str
    optimized_query: str
    query_type: QueryType
    
    # Plan details
    tables: list[str] = field(default_factory=list)
    joins: list[dict] = field(default_factory=list)
    filters: list[str] = field(default_factory=list)
    
    # Cost estimates
    estimated_rows: int = 0
    estimated_cost: float = 0
    
    # Recommendations
    recommendations: list[str] = field(default_factory=list)
    index_suggestions: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "optimized_query": self.optimized_query,
            "query_type": self.query_type.value,
            "tables": self.tables,
            "joins": self.joins,
            "filters": self.filters,
            "estimated_rows": self.estimated_rows,
            "estimated_cost": round(self.estimated_cost, 2),
            "recommendations": self.recommendations,
            "index_suggestions": self.index_suggestions
        }


# ============================================================================
# Query Parser
# ============================================================================

class SQLParser:
    """Simple SQL query parser."""
    
    def parse(self, query: str) -> dict[str, Any]:
        """Parse SQL query into components."""
        query = query.strip().lower()
        
        # Detect query type
        query_type = self._detect_type(query)
        
        result = {
            "query_type": query_type,
            "tables": self._extract_tables(query),
            "columns": self._extract_columns(query),
            "joins": self._extract_joins(query),
            "where_clause": self._extract_where(query),
            "group_by": self._extract_group_by(query),
            "order_by": self._extract_order_by(query),
            "limit": self._extract_limit(query)
        }
        
        return result
    
    def _detect_type(self, query: str) -> QueryType:
        """Detect query type."""
        query = query.strip().lower()
        
        if query.startswith("select"):
            return QueryType.SELECT
        elif query.startswith("insert"):
            return QueryType.INSERT
        elif query.startswith("update"):
            return QueryType.UPDATE
        elif query.startswith("delete"):
            return QueryType.DELETE
        elif query.startswith("create"):
            return QueryType.CREATE
        elif query.startswith("drop"):
            return QueryType.DROP
        elif query.startswith("alter"):
            return QueryType.ALTER
        
        return QueryType.SELECT
    
    def _extract_tables(self, query: str) -> list[str]:
        """Extract table names."""
        tables = []
        
        # FROM clause
        from_match = re.search(r'from\s+(\w+)', query, re.IGNORECASE)
        if from_match:
            tables.append(from_match.group(1))
        
        # JOIN clauses
        join_matches = re.findall(r'join\s+(\w+)', query, re.IGNORECASE)
        tables.extend(join_matches)
        
        return list(set(tables))
    
    def _extract_columns(self, query: str) -> list[str]:
        """Extract column names."""
        select_match = re.search(r'select\s+(.*?)\s+from', query, re.IGNORECASE | re.DOTALL)
        if not select_match:
            return ["*"]
        
        columns_str = select_match.group(1)
        
        if columns_str.strip() == "*":
            return ["*"]
        
        columns = [c.strip() for c in columns_str.split(",")]
        return columns
    
    def _extract_joins(self, query: str) -> list[dict]:
        """Extract join information."""
        joins = []
        
        join_pattern = r'(left|right|inner|full|cross)?\s*join\s+(\w+)\s+(?:on\s+(.+?))?(?=\s+(?:left|right|inner|full|cross|where|group|order|limit|$))'
        
        for match in re.finditer(join_pattern, query, re.IGNORECASE):
            join_type = match.group(1) or "inner"
            table = match.group(2)
            condition = match.group(3)
            
            joins.append({
                "type": join_type.upper(),
                "table": table,
                "condition": condition.strip() if condition else None
            })
        
        return joins
    
    def _extract_where(self, query: str) -> Optional[str]:
        """Extract WHERE clause."""
        where_match = re.search(r'where\s+(.*?)(?=\s+(?:group|order|limit|$))', query, re.IGNORECASE | re.DOTALL)
        return where_match.group(1).strip() if where_match else None
    
    def _extract_group_by(self, query: str) -> list[str]:
        """Extract GROUP BY columns."""
        group_match = re.search(r'group\s+by\s+(.*?)(?=\s+(?:having|order|limit|$))', query, re.IGNORECASE)
        if not group_match:
            return []
        return [c.strip() for c in group_match.group(1).split(",")]
    
    def _extract_order_by(self, query: str) -> list[str]:
        """Extract ORDER BY columns."""
        order_match = re.search(r'order\s+by\s+(.*?)(?=\s+(?:limit|$))', query, re.IGNORECASE)
        if not order_match:
            return []
        return [c.strip() for c in order_match.group(1).split(",")]
    
    def _extract_limit(self, query: str) -> Optional[int]:
        """Extract LIMIT value."""
        limit_match = re.search(r'limit\s+(\d+)', query, re.IGNORECASE)
        return int(limit_match.group(1)) if limit_match else None


# ============================================================================
# Query Optimizer
# ============================================================================

class QueryOptimizer:
    """SQL query optimizer."""
    
    def __init__(self):
        self.parser = SQLParser()
    
    def optimize(self, query: str) -> QueryPlan:
        """Optimize SQL query."""
        parsed = self.parser.parse(query)
        
        optimized = query
        recommendations = []
        index_suggestions = []
        
        # Apply optimizations
        optimized, recs = self._optimize_select(optimized, parsed)
        recommendations.extend(recs)
        
        optimized, recs = self._optimize_joins(optimized, parsed)
        recommendations.extend(recs)
        
        optimized, recs = self._optimize_where(optimized, parsed)
        recommendations.extend(recs)
        
        # Index suggestions
        index_suggestions = self._suggest_indexes(parsed)
        
        return QueryPlan(
            original_query=query,
            optimized_query=optimized,
            query_type=parsed["query_type"],
            tables=parsed["tables"],
            joins=parsed["joins"],
            filters=[parsed["where_clause"]] if parsed["where_clause"] else [],
            recommendations=recommendations,
            index_suggestions=index_suggestions
        )
    
    def _optimize_select(
        self,
        query: str,
        parsed: dict
    ) -> tuple[str, list[str]]:
        """Optimize SELECT clause."""
        recommendations = []
        
        if "*" in parsed["columns"]:
            recommendations.append(
                "Avoid SELECT *. Specify only needed columns to reduce I/O."
            )
        
        return query, recommendations
    
    def _optimize_joins(
        self,
        query: str,
        parsed: dict
    ) -> tuple[str, list[str]]:
        """Optimize JOIN operations."""
        recommendations = []
        
        if len(parsed["joins"]) > 3:
            recommendations.append(
                f"Query has {len(parsed['joins'])} joins. Consider denormalization or CTEs."
            )
        
        return query, recommendations
    
    def _optimize_where(
        self,
        query: str,
        parsed: dict
    ) -> tuple[str, list[str]]:
        """Optimize WHERE clause."""
        recommendations = []
        
        where = parsed.get("where_clause", "")
        
        if where:
            # Check for non-sargable conditions
            if "like '%" in where.lower():
                recommendations.append(
                    "Leading wildcard in LIKE prevents index usage. Consider full-text search."
                )
            
            if re.search(r'or\s+', where, re.IGNORECASE):
                recommendations.append(
                    "OR conditions can prevent index usage. Consider UNION for better performance."
                )
            
            if re.search(r'not\s+in', where, re.IGNORECASE):
                recommendations.append(
                    "NOT IN can be slow. Consider LEFT JOIN with NULL check."
                )
        
        return query, recommendations
    
    def _suggest_indexes(self, parsed: dict) -> list[str]:
        """Suggest indexes based on query."""
        suggestions = []
        
        where = parsed.get("where_clause", "")
        
        if where:
            # Extract column names from WHERE
            columns = re.findall(r'(\w+)\s*[=<>]', where)
            
            for col in columns:
                suggestions.append(
                    f"Consider index on column '{col}' used in WHERE clause"
                )
        
        # ORDER BY columns
        for col in parsed.get("order_by", []):
            col_name = col.split()[0]
            suggestions.append(
                f"Consider index on column '{col_name}' used in ORDER BY"
            )
        
        return suggestions[:5]  # Limit suggestions


# ============================================================================
# Query Builder
# ============================================================================

class QueryBuilder:
    """Fluent SQL query builder."""
    
    def __init__(self):
        self._select: list[str] = []
        self._from: str = ""
        self._joins: list[str] = []
        self._where: list[str] = []
        self._group_by: list[str] = []
        self._having: list[str] = []
        self._order_by: list[str] = []
        self._limit: Optional[int] = None
        self._offset: Optional[int] = None
    
    def select(self, *columns: str) -> "QueryBuilder":
        """Add SELECT columns."""
        self._select.extend(columns)
        return self
    
    def from_table(self, table: str) -> "QueryBuilder":
        """Set FROM table."""
        self._from = table
        return self
    
    def join(
        self,
        table: str,
        condition: str,
        join_type: JoinType = JoinType.INNER
    ) -> "QueryBuilder":
        """Add JOIN clause."""
        self._joins.append(f"{join_type.value.upper()} JOIN {table} ON {condition}")
        return self
    
    def where(self, condition: str) -> "QueryBuilder":
        """Add WHERE condition."""
        self._where.append(condition)
        return self
    
    def and_where(self, condition: str) -> "QueryBuilder":
        """Add AND WHERE condition."""
        self._where.append(condition)
        return self
    
    def or_where(self, condition: str) -> "QueryBuilder":
        """Add OR WHERE condition."""
        if self._where:
            last = self._where.pop()
            self._where.append(f"({last} OR {condition})")
        else:
            self._where.append(condition)
        return self
    
    def group_by(self, *columns: str) -> "QueryBuilder":
        """Add GROUP BY columns."""
        self._group_by.extend(columns)
        return self
    
    def having(self, condition: str) -> "QueryBuilder":
        """Add HAVING condition."""
        self._having.append(condition)
        return self
    
    def order_by(self, column: str, desc: bool = False) -> "QueryBuilder":
        """Add ORDER BY clause."""
        direction = "DESC" if desc else "ASC"
        self._order_by.append(f"{column} {direction}")
        return self
    
    def limit(self, count: int) -> "QueryBuilder":
        """Set LIMIT."""
        self._limit = count
        return self
    
    def offset(self, count: int) -> "QueryBuilder":
        """Set OFFSET."""
        self._offset = count
        return self
    
    def build(self) -> str:
        """Build the SQL query."""
        parts = []
        
        # SELECT
        columns = ", ".join(self._select) if self._select else "*"
        parts.append(f"SELECT {columns}")
        
        # FROM
        if self._from:
            parts.append(f"FROM {self._from}")
        
        # JOINs
        parts.extend(self._joins)
        
        # WHERE
        if self._where:
            parts.append(f"WHERE {' AND '.join(self._where)}")
        
        # GROUP BY
        if self._group_by:
            parts.append(f"GROUP BY {', '.join(self._group_by)}")
        
        # HAVING
        if self._having:
            parts.append(f"HAVING {' AND '.join(self._having)}")
        
        # ORDER BY
        if self._order_by:
            parts.append(f"ORDER BY {', '.join(self._order_by)}")
        
        # LIMIT
        if self._limit:
            parts.append(f"LIMIT {self._limit}")
        
        # OFFSET
        if self._offset:
            parts.append(f"OFFSET {self._offset}")
        
        return "\n".join(parts)
    
    def reset(self) -> "QueryBuilder":
        """Reset builder."""
        self.__init__()
        return self


# ============================================================================
# Query Optimizer Service
# ============================================================================

class QueryOptimizerService:
    """
    Query optimization service.
    
    Features:
    - Query parsing and analysis
    - Performance recommendations
    - Index suggestions
    - Query building
    - Query rewriting
    """
    
    def __init__(self):
        self.optimizer = QueryOptimizer()
        self.parser = SQLParser()
    
    def analyze(self, query: str) -> QueryPlan:
        """Analyze and optimize query."""
        return self.optimizer.optimize(query)
    
    def parse(self, query: str) -> dict[str, Any]:
        """Parse query into components."""
        return self.parser.parse(query)
    
    def builder(self) -> QueryBuilder:
        """Get new query builder."""
        return QueryBuilder()
    
    def estimate_cost(self, query: str, table_stats: dict = None) -> float:
        """Estimate query cost."""
        parsed = self.parser.parse(query)
        
        # Simple cost model
        cost = 1.0
        
        # Add cost for each table
        cost *= len(parsed["tables"])
        
        # Add cost for joins
        cost *= (1 + len(parsed["joins"]) * 0.5)
        
        # Full table scan if no WHERE
        if not parsed["where_clause"]:
            cost *= 10
        
        # Add cost for sorting
        if parsed["order_by"]:
            cost *= 1.2
        
        return cost


# Factory function
def get_query_optimizer() -> QueryOptimizerService:
    """Get query optimizer service instance."""
    return QueryOptimizerService()
