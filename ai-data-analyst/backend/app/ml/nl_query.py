# AI Enterprise Data Analyst - Natural Language Query Engine
# Convert natural language to SQL/Pandas operations

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import re

import pandas as pd

from app.core.logging import get_logger
from app.services.llm_service import LLMService, get_llm_service

logger = get_logger(__name__)


# ============================================================================
# Query Types
# ============================================================================

class NLQueryType(str, Enum):
    """Types of natural language queries."""
    AGGREGATION = "aggregation"
    FILTER = "filter"
    SORT = "sort"
    GROUP = "group"
    JOIN = "join"
    COMPARISON = "comparison"
    TIME_SERIES = "time_series"
    STATISTICAL = "statistical"


@dataclass
class ParsedQuery:
    """Parsed natural language query."""
    
    original: str
    query_type: NLQueryType
    
    # Extracted entities
    columns: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    filters: list[dict] = field(default_factory=list)
    aggregations: list[str] = field(default_factory=list)
    group_by: list[str] = field(default_factory=list)
    order_by: list[str] = field(default_factory=list)
    limit: Optional[int] = None
    
    # Generated code
    sql: str = ""
    pandas_code: str = ""
    
    confidence: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "original": self.original,
            "query_type": self.query_type.value,
            "columns": self.columns,
            "filters": self.filters,
            "aggregations": self.aggregations,
            "group_by": self.group_by,
            "sql": self.sql,
            "pandas_code": self.pandas_code,
            "confidence": round(self.confidence, 2)
        }


@dataclass
class QueryResult:
    """Natural language query result."""
    
    query: ParsedQuery
    data: pd.DataFrame
    execution_time_ms: float
    
    summary: str = ""
    visualization_hint: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query.to_dict(),
            "row_count": len(self.data),
            "columns": self.data.columns.tolist(),
            "execution_time_ms": round(self.execution_time_ms, 2),
            "summary": self.summary,
            "visualization_hint": self.visualization_hint
        }


# ============================================================================
# Query Intent Classifier
# ============================================================================

class QueryIntentClassifier:
    """Classify query intent from natural language."""
    
    PATTERNS = {
        NLQueryType.AGGREGATION: [
            r'\b(total|sum|count|average|mean|max|min|how many)\b',
            r'\b(aggregate|summarize|calculate)\b'
        ],
        NLQueryType.FILTER: [
            r'\b(where|filter|only|just|with|without|exclude|include)\b',
            r'\b(greater|less|equal|between|contains|starts|ends)\b'
        ],
        NLQueryType.SORT: [
            r'\b(sort|order|rank|top|bottom|highest|lowest|best|worst)\b'
        ],
        NLQueryType.GROUP: [
            r'\b(by|per|each|every|group|breakdown|segment)\b'
        ],
        NLQueryType.COMPARISON: [
            r'\b(compare|versus|vs|difference|between|compared)\b'
        ],
        NLQueryType.TIME_SERIES: [
            r'\b(trend|over time|monthly|weekly|daily|yearly|growth)\b',
            r'\b(forecast|predict|projection)\b'
        ],
        NLQueryType.STATISTICAL: [
            r'\b(correlation|distribution|variance|outlier|anomaly)\b',
            r'\b(significant|p-value|hypothesis)\b'
        ]
    }
    
    def classify(self, query: str) -> list[NLQueryType]:
        """Classify query into intent types."""
        query_lower = query.lower()
        detected = []
        
        for query_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    detected.append(query_type)
                    break
        
        if not detected:
            detected.append(NLQueryType.FILTER)
        
        return detected


# ============================================================================
# Entity Extractor
# ============================================================================

class EntityExtractor:
    """Extract entities from natural language queries."""
    
    def extract(
        self,
        query: str,
        available_columns: list[str]
    ) -> dict[str, Any]:
        """Extract entities from query."""
        query_lower = query.lower()
        
        entities = {
            "columns": [],
            "filters": [],
            "aggregations": [],
            "group_by": [],
            "order_by": [],
            "limit": None
        }
        
        # Extract columns
        for col in available_columns:
            if col.lower() in query_lower or col.replace('_', ' ').lower() in query_lower:
                entities["columns"].append(col)
        
        # Extract aggregations
        agg_patterns = {
            "sum": r'\b(total|sum of)\b',
            "count": r'\b(count|how many|number of)\b',
            "avg": r'\b(average|mean|avg)\b',
            "max": r'\b(maximum|max|highest|top)\b',
            "min": r'\b(minimum|min|lowest|bottom)\b'
        }
        
        for agg, pattern in agg_patterns.items():
            if re.search(pattern, query_lower):
                entities["aggregations"].append(agg)
        
        # Extract limit
        limit_match = re.search(r'\b(top|first|last)\s+(\d+)\b', query_lower)
        if limit_match:
            entities["limit"] = int(limit_match.group(2))
        
        # Extract filters
        filter_patterns = [
            (r"(\w+)\s*[=>]\s*(\d+)", "gte"),
            (r"(\w+)\s*[=<]\s*(\d+)", "lte"),
            (r"(\w+)\s*=\s*['\"]([\w\s]+)['\"]", "eq"),
            (r"where\s+(\w+)\s+is\s+['\"]([\w\s]+)['\"]", "eq")
        ]
        
        for pattern, op in filter_patterns:
            for match in re.finditer(pattern, query_lower):
                col, val = match.groups()
                if any(c.lower() == col.lower() for c in available_columns):
                    entities["filters"].append({
                        "column": col,
                        "operator": op,
                        "value": val
                    })
        
        return entities


# ============================================================================
# SQL Generator
# ============================================================================

class SQLGenerator:
    """Generate SQL from parsed entities."""
    
    def generate(
        self,
        entities: dict[str, Any],
        table_name: str = "data"
    ) -> str:
        """Generate SQL query."""
        parts = []
        
        # SELECT clause
        columns = entities.get("columns", ["*"])
        aggregations = entities.get("aggregations", [])
        
        if aggregations:
            select_parts = []
            for col in columns:
                for agg in aggregations:
                    select_parts.append(f"{agg.upper()}({col})")
            if entities.get("group_by"):
                select_parts = entities["group_by"] + select_parts
            parts.append(f"SELECT {', '.join(select_parts)}")
        else:
            parts.append(f"SELECT {', '.join(columns) if columns else '*'}")
        
        # FROM clause
        parts.append(f"FROM {table_name}")
        
        # WHERE clause
        filters = entities.get("filters", [])
        if filters:
            conditions = []
            for f in filters:
                op_map = {"eq": "=", "gte": ">=", "lte": "<=", "ne": "!="}
                op = op_map.get(f["operator"], "=")
                val = f"'{f['value']}'" if isinstance(f["value"], str) else f["value"]
                conditions.append(f"{f['column']} {op} {val}")
            parts.append(f"WHERE {' AND '.join(conditions)}")
        
        # GROUP BY clause
        group_by = entities.get("group_by", [])
        if group_by:
            parts.append(f"GROUP BY {', '.join(group_by)}")
        
        # ORDER BY clause
        order_by = entities.get("order_by", [])
        if order_by:
            parts.append(f"ORDER BY {', '.join(order_by)}")
        
        # LIMIT clause
        limit = entities.get("limit")
        if limit:
            parts.append(f"LIMIT {limit}")
        
        return " ".join(parts)


# ============================================================================
# Pandas Generator
# ============================================================================

class PandasGenerator:
    """Generate Pandas code from parsed entities."""
    
    def generate(
        self,
        entities: dict[str, Any],
        df_name: str = "df"
    ) -> str:
        """Generate Pandas code."""
        operations = [df_name]
        
        # Filters
        filters = entities.get("filters", [])
        if filters:
            conditions = []
            for f in filters:
                col = f["column"]
                val = f"'{f['value']}'" if isinstance(f["value"], str) else f["value"]
                op_map = {"eq": "==", "gte": ">=", "lte": "<=", "ne": "!="}
                op = op_map.get(f["operator"], "==")
                conditions.append(f"({df_name}['{col}'] {op} {val})")
            operations.append(f"[{' & '.join(conditions)}]")
        
        # Group by
        group_by = entities.get("group_by", [])
        aggregations = entities.get("aggregations", [])
        columns = entities.get("columns", [])
        
        if group_by and aggregations:
            operations.append(f".groupby({group_by})")
            if columns:
                operations.append(f"[{columns}]")
            operations.append(f".{aggregations[0]}()")
        elif aggregations:
            if columns:
                operations.append(f"[{columns}]")
            operations.append(f".{aggregations[0]}()")
        elif columns:
            operations.append(f"[{columns}]")
        
        # Sort
        order_by = entities.get("order_by", [])
        if order_by:
            operations.append(f".sort_values({order_by}, ascending=False)")
        
        # Limit
        limit = entities.get("limit")
        if limit:
            operations.append(f".head({limit})")
        
        return "".join(operations)


# ============================================================================
# Natural Language Query Engine
# ============================================================================

class NLQueryEngine:
    """
    Natural Language to Query Engine.
    
    Features:
    - Intent classification
    - Entity extraction
    - SQL generation
    - Pandas code generation
    - LLM-powered parsing
    - Result summarization
    """
    
    def __init__(self, llm_service: LLMService = None):
        self.llm = llm_service or get_llm_service()
        self.classifier = QueryIntentClassifier()
        self.extractor = EntityExtractor()
        self.sql_generator = SQLGenerator()
        self.pandas_generator = PandasGenerator()
    
    def parse(
        self,
        query: str,
        available_columns: list[str],
        table_name: str = "data"
    ) -> ParsedQuery:
        """Parse natural language query."""
        # Classify intent
        intents = self.classifier.classify(query)
        primary_intent = intents[0] if intents else NLQueryType.FILTER
        
        # Extract entities
        entities = self.extractor.extract(query, available_columns)
        
        # Generate SQL
        sql = self.sql_generator.generate(entities, table_name)
        
        # Generate Pandas
        pandas_code = self.pandas_generator.generate(entities)
        
        return ParsedQuery(
            original=query,
            query_type=primary_intent,
            columns=entities["columns"],
            filters=entities["filters"],
            aggregations=entities["aggregations"],
            group_by=entities["group_by"],
            order_by=entities["order_by"],
            limit=entities["limit"],
            sql=sql,
            pandas_code=pandas_code,
            confidence=0.8 if entities["columns"] else 0.5
        )
    
    def execute(
        self,
        query: str,
        df: pd.DataFrame,
        table_name: str = "data"
    ) -> QueryResult:
        """Execute natural language query on DataFrame."""
        import time
        start = time.time()
        
        # Parse query
        parsed = self.parse(query, df.columns.tolist(), table_name)
        
        # Execute using Pandas
        try:
            result_df = eval(parsed.pandas_code)
            
            if isinstance(result_df, pd.Series):
                result_df = result_df.to_frame()
        except Exception as e:
            logger.warning(f"Pandas execution failed: {e}, returning full data")
            result_df = df
        
        # Generate summary
        summary = self._generate_summary(parsed, result_df)
        
        # Visualization hint
        viz_hint = self._suggest_visualization(parsed)
        
        return QueryResult(
            query=parsed,
            data=result_df,
            execution_time_ms=(time.time() - start) * 1000,
            summary=summary,
            visualization_hint=viz_hint
        )
    
    def _generate_summary(
        self,
        parsed: ParsedQuery,
        result: pd.DataFrame
    ) -> str:
        """Generate human-readable summary."""
        if parsed.aggregations:
            return f"Calculated {', '.join(parsed.aggregations)} returning {len(result)} rows"
        elif parsed.filters:
            return f"Filtered data to {len(result)} matching records"
        else:
            return f"Retrieved {len(result)} records"
    
    def _suggest_visualization(self, parsed: ParsedQuery) -> str:
        """Suggest appropriate visualization."""
        if parsed.query_type == NLQueryType.TIME_SERIES:
            return "line_chart"
        elif parsed.query_type == NLQueryType.COMPARISON:
            return "bar_chart"
        elif parsed.aggregations and parsed.group_by:
            return "bar_chart"
        elif NLQueryType.AGGREGATION in [parsed.query_type]:
            return "metric_card"
        return "table"
    
    async def parse_with_llm(
        self,
        query: str,
        schema: dict[str, str]
    ) -> ParsedQuery:
        """Use LLM for advanced query parsing."""
        prompt = f"""Parse this natural language query into SQL components.

Query: {query}

Available columns and types:
{schema}

Respond with JSON containing:
- columns: list of column names to select
- filters: list of filter conditions
- aggregations: list of aggregation functions
- group_by: list of grouping columns
- order_by: list of sorting columns
- limit: number of rows to return (or null)
"""
        
        response = await self.llm.complete(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        # Parse LLM response
        try:
            import json
            entities = json.loads(response.content)
            
            sql = self.sql_generator.generate(entities)
            pandas_code = self.pandas_generator.generate(entities)
            
            return ParsedQuery(
                original=query,
                query_type=NLQueryType.FILTER,
                columns=entities.get("columns", []),
                filters=entities.get("filters", []),
                aggregations=entities.get("aggregations", []),
                group_by=entities.get("group_by", []),
                order_by=entities.get("order_by", []),
                limit=entities.get("limit"),
                sql=sql,
                pandas_code=pandas_code,
                confidence=0.9
            )
        except:
            return self.parse(query, list(schema.keys()))


# Factory function
def get_nl_query_engine() -> NLQueryEngine:
    """Get natural language query engine instance."""
    return NLQueryEngine()
