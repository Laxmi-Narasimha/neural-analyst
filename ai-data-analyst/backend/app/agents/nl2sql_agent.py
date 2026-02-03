# AI Enterprise Data Analyst - NL2SQL Agent
# Agent for converting natural language to SQL queries

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import UUID

import pandas as pd
import re

from app.agents.base_agent import (
    BaseAgent,
    AgentRole,
    AgentContext,
    AgentTool,
)
from app.services.llm_service import get_llm_service, Message as LLMMessage
from app.core.logging import get_logger, LogContext

logger = get_logger(__name__)


@dataclass
class SQLGenerationResult:
    """Result of SQL generation."""
    
    natural_language_query: str
    generated_sql: str
    is_valid: bool = True
    explanation: str = ""
    warning: Optional[str] = None
    tables_used: list[str] = field(default_factory=list)
    columns_used: list[str] = field(default_factory=list)
    query_type: str = "SELECT"  # SELECT, INSERT, UPDATE, DELETE
    estimated_complexity: str = "simple"  # simple, moderate, complex
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.natural_language_query,
            "sql": self.generated_sql,
            "is_valid": self.is_valid,
            "explanation": self.explanation,
            "warning": self.warning,
            "tables_used": self.tables_used,
            "columns_used": self.columns_used,
            "query_type": self.query_type,
            "complexity": self.estimated_complexity
        }


class SchemaExtractor:
    """Extract schema information from dataframes for SQL context."""
    
    @staticmethod
    def extract_schema(df: pd.DataFrame, table_name: str = "data") -> dict[str, Any]:
        """Extract schema from a pandas DataFrame."""
        columns = []
        
        for col in df.columns:
            col_info = {
                "name": col,
                "type": str(df[col].dtype),
                "nullable": df[col].isnull().any(),
                "unique_count": int(df[col].nunique()),
                "sample_values": df[col].dropna().head(3).tolist()
            }
            
            # Map pandas dtype to SQL type
            dtype = str(df[col].dtype)
            if "int" in dtype:
                col_info["sql_type"] = "INTEGER"
            elif "float" in dtype:
                col_info["sql_type"] = "FLOAT"
            elif "datetime" in dtype:
                col_info["sql_type"] = "TIMESTAMP"
            elif "bool" in dtype:
                col_info["sql_type"] = "BOOLEAN"
            else:
                col_info["sql_type"] = "VARCHAR"
            
            columns.append(col_info)
        
        return {
            "table_name": table_name,
            "columns": columns,
            "row_count": len(df),
            "column_count": len(df.columns)
        }
    
    @staticmethod
    def schema_to_ddl(schema: dict[str, Any]) -> str:
        """Convert schema to DDL statement."""
        columns_ddl = []
        
        for col in schema["columns"]:
            null_spec = "" if col["nullable"] else " NOT NULL"
            columns_ddl.append(f"  {col['name']} {col['sql_type']}{null_spec}")
        
        return f"CREATE TABLE {schema['table_name']} (\n" + \
               ",\n".join(columns_ddl) + \
               "\n);"
    
    @staticmethod
    def schema_to_prompt_context(schema: dict[str, Any]) -> str:
        """Convert schema to natural language context for LLM."""
        lines = [f"Table: {schema['table_name']} ({schema['row_count']} rows)"]
        lines.append("Columns:")
        
        for col in schema["columns"]:
            samples = ", ".join(str(v) for v in col["sample_values"][:3])
            lines.append(f"  - {col['name']} ({col['sql_type']}): e.g., {samples}")
        
        return "\n".join(lines)


class NL2SQLEngine:
    """
    Engine for converting natural language to SQL.
    
    Uses LLM for understanding and SQL generation with
    schema awareness and validation.
    """
    
    def __init__(self, llm_client=None) -> None:
        self.llm_client = llm_client or get_llm_service()
        self.schema_extractor = SchemaExtractor()
    
    async def generate_sql(
        self,
        query: str,
        schema: dict[str, Any],
        dialect: str = "postgresql"
    ) -> SQLGenerationResult:
        """Generate SQL from natural language query."""
        
        # Build schema context
        schema_context = self.schema_extractor.schema_to_prompt_context(schema)
        
        # LLM prompt
        system_prompt = f"""You are an expert SQL query generator. Generate {dialect} SQL queries based on natural language questions.

Database Schema:
{schema_context}

Rules:
1. Generate only valid {dialect} SQL
2. Use proper column names from the schema
3. Include appropriate JOINs when needed
4. Add ORDER BY when relevant
5. Limit results for safety (max 1000 rows)
6. Use aliases for readability
7. Handle NULL values properly

Respond with ONLY the SQL query, no explanations."""

        # Generate SQL
        response = await self.llm_client.complete(
            messages=[
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=f"Write a SQL query to: {query}")
            ],
            temperature=0.0
        )
        
        sql = self._clean_sql(response.content)
        
        # Validate and analyze
        is_valid = self._validate_sql(sql)
        tables = self._extract_tables(sql)
        columns = self._extract_columns(sql, schema)
        query_type = self._detect_query_type(sql)
        complexity = self._estimate_complexity(sql)
        
        # Generate explanation
        explanation = await self._generate_explanation(query, sql, schema)
        
        # Check for warnings
        warning = self._check_warnings(sql)
        
        return SQLGenerationResult(
            natural_language_query=query,
            generated_sql=sql,
            is_valid=is_valid,
            explanation=explanation,
            warning=warning,
            tables_used=tables,
            columns_used=columns,
            query_type=query_type,
            estimated_complexity=complexity
        )
    
    def _clean_sql(self, sql: str) -> str:
        """Clean and format SQL."""
        # Remove markdown code blocks
        sql = re.sub(r'```sql\n?', '', sql)
        sql = re.sub(r'```\n?', '', sql)
        sql = sql.strip()
        
        # Add semicolon if missing
        if sql and not sql.endswith(';'):
            sql += ';'
        
        return sql
    
    def _validate_sql(self, sql: str) -> bool:
        """Basic SQL validation."""
        if not sql:
            return False
        
        # Check for basic SQL structure
        sql_upper = sql.upper()
        valid_starts = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH', 'CREATE', 'ALTER', 'DROP']
        
        return any(sql_upper.strip().startswith(start) for start in valid_starts)
    
    def _extract_tables(self, sql: str) -> list[str]:
        """Extract table names from SQL."""
        tables = []
        
        # FROM clause
        from_match = re.findall(r'FROM\s+(\w+)', sql, re.IGNORECASE)
        tables.extend(from_match)
        
        # JOIN clauses
        join_match = re.findall(r'JOIN\s+(\w+)', sql, re.IGNORECASE)
        tables.extend(join_match)
        
        return list(set(tables))
    
    def _extract_columns(self, sql: str, schema: dict[str, Any]) -> list[str]:
        """Extract column references from SQL."""
        schema_columns = {col["name"].lower() for col in schema["columns"]}
        
        # Find all potential column references
        words = re.findall(r'\b\w+\b', sql)
        columns = [w for w in words if w.lower() in schema_columns]
        
        return list(set(columns))
    
    def _detect_query_type(self, sql: str) -> str:
        """Detect query type."""
        sql_upper = sql.upper().strip()
        
        if sql_upper.startswith('SELECT') or sql_upper.startswith('WITH'):
            return "SELECT"
        elif sql_upper.startswith('INSERT'):
            return "INSERT"
        elif sql_upper.startswith('UPDATE'):
            return "UPDATE"
        elif sql_upper.startswith('DELETE'):
            return "DELETE"
        else:
            return "OTHER"
    
    def _estimate_complexity(self, sql: str) -> str:
        """Estimate query complexity."""
        sql_upper = sql.upper()
        
        complexity_score = 0
        
        # Joins
        complexity_score += sql_upper.count('JOIN') * 2
        
        # Subqueries
        complexity_score += sql_upper.count('SELECT') - 1
        
        # Window functions
        complexity_score += sql_upper.count('OVER(') * 2
        complexity_score += sql_upper.count('PARTITION BY') * 2
        
        # CTEs
        complexity_score += sql_upper.count('WITH') * 2
        
        # Aggregations
        for agg in ['GROUP BY', 'HAVING', 'SUM', 'COUNT', 'AVG', 'MIN', 'MAX']:
            complexity_score += sql_upper.count(agg)
        
        if complexity_score <= 2:
            return "simple"
        elif complexity_score <= 6:
            return "moderate"
        else:
            return "complex"
    
    async def _generate_explanation(
        self,
        query: str,
        sql: str,
        schema: dict[str, Any]
    ) -> str:
        """Generate explanation of the SQL query."""
        response = await self.llm_client.complete(
            messages=[
                LLMMessage(
                    role="system",
                    content="Explain this SQL query in simple terms. Be concise."
                ),
                LLMMessage(
                    role="user",
                    content=f"Question: {query}\n\nSQL: {sql}"
                )
            ],
            temperature=0.2
        )
        
        return response.content
    
    def _check_warnings(self, sql: str) -> Optional[str]:
        """Check for potential issues in SQL."""
        warnings = []
        sql_upper = sql.upper()
        
        # Check for DELETE/UPDATE without WHERE
        if 'DELETE' in sql_upper and 'WHERE' not in sql_upper:
            warnings.append("DELETE without WHERE clause - will affect all rows")
        
        if 'UPDATE' in sql_upper and 'WHERE' not in sql_upper:
            warnings.append("UPDATE without WHERE clause - will affect all rows")
        
        # Check for SELECT *
        if 'SELECT *' in sql_upper:
            warnings.append("SELECT * may retrieve unnecessary columns")
        
        # Check for missing LIMIT
        if 'SELECT' in sql_upper and 'LIMIT' not in sql_upper:
            warnings.append("No LIMIT clause - may return many rows")
        
        return "; ".join(warnings) if warnings else None


class NL2SQLAgent(BaseAgent[dict[str, Any]]):
    """
    NL2SQL Agent for natural language to SQL conversion.
    
    Capabilities:
    - Convert questions to SQL queries
    - Schema-aware query generation
    - Query validation and explanation
    - Execute queries on datasets
    """
    
    name: str = "NL2SQLAgent"
    description: str = "Convert natural language to SQL queries"
    role: AgentRole = AgentRole.SPECIALIST
    
    def __init__(self, llm_client=None) -> None:
        super().__init__(llm_client or get_llm_service())
        self.engine = NL2SQLEngine(self._llm_client)
    
    def _register_tools(self) -> None:
        """Register NL2SQL tools."""
        
        self.register_tool(AgentTool(
            name="generate_sql",
            description="Convert natural language question to SQL query",
            function=self._generate_sql,
            parameters={
                "question": {"type": "string"},
                "schema": {"type": "object"},
                "dialect": {"type": "string", "default": "postgresql"}
            },
            required_params=["question", "schema"]
        ))
        
        self.register_tool(AgentTool(
            name="execute_query",
            description="Execute SQL query on a dataset",
            function=self._execute_query,
            parameters={
                "data": {"type": "object"},
                "sql": {"type": "string"}
            },
            required_params=["data", "sql"]
        ))
        
        self.register_tool(AgentTool(
            name="explain_query",
            description="Explain what a SQL query does",
            function=self._explain_query,
            parameters={
                "sql": {"type": "string"}
            },
            required_params=["sql"]
        ))
    
    async def _execute_core(self, context: AgentContext) -> dict[str, Any]:
        """Execute NL2SQL task."""
        response = await self._llm_client.complete(
            messages=[
                LLMMessage(
                    role="system",
                    content="You are a SQL expert. Help translate questions to queries."
                ),
                LLMMessage(role="user", content=context.task_description)
            ]
        )
        
        return {
            "analysis": response.content,
            "suggestions": ["Try asking about specific columns", "Use aggregations for summaries"]
        }
    
    async def _generate_sql(
        self,
        question: str,
        schema: dict[str, Any],
        dialect: str = "postgresql"
    ) -> dict[str, Any]:
        """Generate SQL from question."""
        result = await self.engine.generate_sql(question, schema, dialect)
        return result.to_dict()
    
    async def _execute_query(
        self,
        data: dict,
        sql: str
    ) -> dict[str, Any]:
        """Execute SQL query on data using pandas."""
        try:
            import pandasql as ps
            
            df = pd.DataFrame(data)
            
            # Use pandasql to run SQL on DataFrame
            result_df = ps.sqldf(sql, {"data": df})
            
            return {
                "success": True,
                "row_count": len(result_df),
                "columns": result_df.columns.tolist(),
                "data": result_df.to_dict(orient="records")[:100]
            }
        except ImportError:
            # Fallback without pandasql
            return {
                "success": False,
                "error": "pandasql not installed. Install with: pip install pandasql",
                "sql": sql
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _explain_query(
        self,
        sql: str
    ) -> dict[str, Any]:
        """Explain a SQL query."""
        response = await self._llm_client.complete(
            messages=[
                LLMMessage(
                    role="system",
                    content="Explain this SQL query step by step in simple terms."
                ),
                LLMMessage(role="user", content=sql)
            ]
        )
        
        return {
            "sql": sql,
            "explanation": response.content
        }


# Factory function
def get_nl2sql_agent() -> NL2SQLAgent:
    """Get NL2SQL agent instance."""
    return NL2SQLAgent()
