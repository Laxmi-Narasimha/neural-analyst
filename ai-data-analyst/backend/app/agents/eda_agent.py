# AI Enterprise Data Analyst - EDA Agent
# Automated Exploratory Data Analysis with statistical insights

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import UUID

import numpy as np
import pandas as pd
from scipy import stats

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
class EDAReport:
    """Comprehensive EDA report."""
    
    # Dataset overview
    n_rows: int = 0
    n_columns: int = 0
    memory_mb: float = 0.0
    
    # Column types
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    datetime_columns: list[str] = field(default_factory=list)
    text_columns: list[str] = field(default_factory=list)
    
    # Missing values
    total_missing: int = 0
    missing_percentage: float = 0.0
    columns_with_missing: dict[str, float] = field(default_factory=dict)
    
    # Duplicates
    duplicate_rows: int = 0
    duplicate_percentage: float = 0.0
    
    # Numeric statistics
    numeric_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    
    # Categorical statistics
    categorical_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    
    # Correlations
    high_correlations: list[dict[str, Any]] = field(default_factory=list)
    
    # Outliers
    outliers: dict[str, dict[str, Any]] = field(default_factory=dict)
    
    # Skewness and Kurtosis
    distributions: dict[str, dict[str, float]] = field(default_factory=dict)
    
    # Insights
    insights: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overview": {
                "n_rows": self.n_rows,
                "n_columns": self.n_columns,
                "memory_mb": round(self.memory_mb, 2)
            },
            "column_types": {
                "numeric": len(self.numeric_columns),
                "categorical": len(self.categorical_columns),
                "datetime": len(self.datetime_columns),
                "text": len(self.text_columns)
            },
            "missing_values": {
                "total": self.total_missing,
                "percentage": round(self.missing_percentage, 2),
                "by_column": {k: round(v, 2) for k, v in self.columns_with_missing.items()}
            },
            "duplicates": {
                "count": self.duplicate_rows,
                "percentage": round(self.duplicate_percentage, 2)
            },
            "numeric_statistics": self.numeric_stats,
            "categorical_statistics": self.categorical_stats,
            "high_correlations": self.high_correlations[:10],
            "outliers": self.outliers,
            "distributions": self.distributions,
            "insights": self.insights,
            "warnings": self.warnings
        }


class EDAEngine:
    """
    Automated EDA engine for comprehensive data exploration.
    
    Performs:
    - Basic statistics
    - Distribution analysis
    - Correlation analysis
    - Outlier detection
    - Missing value analysis
    - Automatic insight generation
    """
    
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.report = EDAReport()
        
        # Identify column types
        self._identify_columns()
    
    def _identify_columns(self) -> None:
        """Identify column types."""
        for col in self.df.columns:
            dtype = self.df[col].dtype
            
            if pd.api.types.is_numeric_dtype(dtype):
                self.report.numeric_columns.append(col)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                self.report.datetime_columns.append(col)
            elif self.df[col].dtype == 'object':
                avg_len = self.df[col].astype(str).str.len().mean()
                if avg_len > 100:
                    self.report.text_columns.append(col)
                else:
                    self.report.categorical_columns.append(col)
    
    def run_full_analysis(self) -> EDAReport:
        """Run complete EDA analysis."""
        self._basic_overview()
        self._missing_analysis()
        self._duplicate_analysis()
        self._numeric_analysis()
        self._categorical_analysis()
        self._correlation_analysis()
        self._outlier_analysis()
        self._distribution_analysis()
        self._generate_insights()
        
        return self.report
    
    def _basic_overview(self) -> None:
        """Basic dataset overview."""
        self.report.n_rows = len(self.df)
        self.report.n_columns = len(self.df.columns)
        self.report.memory_mb = self.df.memory_usage(deep=True).sum() / (1024 * 1024)
    
    def _missing_analysis(self) -> None:
        """Analyze missing values."""
        missing = self.df.isnull().sum()
        self.report.total_missing = int(missing.sum())
        total_cells = self.report.n_rows * self.report.n_columns
        self.report.missing_percentage = (self.report.total_missing / total_cells * 100) if total_cells > 0 else 0
        
        for col in missing[missing > 0].index:
            pct = missing[col] / self.report.n_rows * 100
            self.report.columns_with_missing[col] = pct
            
            if pct > 50:
                self.report.warnings.append(f"Column '{col}' has {pct:.1f}% missing values")
    
    def _duplicate_analysis(self) -> None:
        """Analyze duplicate rows."""
        self.report.duplicate_rows = int(self.df.duplicated().sum())
        self.report.duplicate_percentage = (
            self.report.duplicate_rows / self.report.n_rows * 100
        ) if self.report.n_rows > 0 else 0
        
        if self.report.duplicate_percentage > 5:
            self.report.warnings.append(
                f"{self.report.duplicate_percentage:.1f}% duplicate rows detected"
            )
    
    def _numeric_analysis(self) -> None:
        """Analyze numeric columns."""
        for col in self.report.numeric_columns:
            series = self.df[col].dropna()
            
            if len(series) == 0:
                continue
            
            self.report.numeric_stats[col] = {
                "count": len(series),
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "25%": float(series.quantile(0.25)),
                "50%": float(series.median()),
                "75%": float(series.quantile(0.75)),
                "max": float(series.max()),
                "range": float(series.max() - series.min()),
                "iqr": float(series.quantile(0.75) - series.quantile(0.25)),
                "zeros": int((series == 0).sum()),
                "zeros_pct": float((series == 0).sum() / len(series) * 100)
            }
    
    def _categorical_analysis(self) -> None:
        """Analyze categorical columns."""
        for col in self.report.categorical_columns:
            series = self.df[col].dropna()
            
            if len(series) == 0:
                continue
            
            value_counts = series.value_counts()
            
            self.report.categorical_stats[col] = {
                "unique_count": int(series.nunique()),
                "unique_percentage": float(series.nunique() / len(series) * 100),
                "top_5": value_counts.head(5).to_dict(),
                "mode": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                "mode_frequency": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
            }
            
            # Check high cardinality
            if series.nunique() > 100:
                self.report.warnings.append(
                    f"Column '{col}' has high cardinality ({series.nunique()} unique values)"
                )
    
    def _correlation_analysis(self) -> None:
        """Analyze correlations between numeric columns."""
        if len(self.report.numeric_columns) < 2:
            return
        
        # Calculate correlation matrix
        corr_matrix = self.df[self.report.numeric_columns].corr()
        
        # Find high correlations
        for i, col1 in enumerate(self.report.numeric_columns):
            for col2 in self.report.numeric_columns[i+1:]:
                corr = corr_matrix.loc[col1, col2]
                
                if abs(corr) > 0.7 and not np.isnan(corr):
                    self.report.high_correlations.append({
                        "column1": col1,
                        "column2": col2,
                        "correlation": round(corr, 4),
                        "strength": "very strong" if abs(corr) > 0.9 else "strong"
                    })
        
        # Sort by absolute correlation
        self.report.high_correlations.sort(
            key=lambda x: abs(x["correlation"]),
            reverse=True
        )
    
    def _outlier_analysis(self) -> None:
        """Detect outliers using IQR method."""
        for col in self.report.numeric_columns:
            series = self.df[col].dropna()
            
            if len(series) < 10:
                continue
            
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            
            if len(outliers) > 0:
                self.report.outliers[col] = {
                    "count": len(outliers),
                    "percentage": round(len(outliers) / len(series) * 100, 2),
                    "lower_bound": round(lower_bound, 4),
                    "upper_bound": round(upper_bound, 4),
                    "min_outlier": round(float(outliers.min()), 4),
                    "max_outlier": round(float(outliers.max()), 4)
                }
    
    def _distribution_analysis(self) -> None:
        """Analyze distributions of numeric columns."""
        for col in self.report.numeric_columns:
            series = self.df[col].dropna()
            
            if len(series) < 20:
                continue
            
            skewness = float(stats.skew(series))
            kurtosis = float(stats.kurtosis(series))
            
            # Normality test
            if len(series) < 5000:
                _, p_value = stats.shapiro(series.sample(min(5000, len(series))))
            else:
                _, p_value = stats.normaltest(series.sample(5000))
            
            self.report.distributions[col] = {
                "skewness": round(skewness, 4),
                "kurtosis": round(kurtosis, 4),
                "normality_p_value": round(p_value, 4),
                "is_normal": p_value > 0.05,
                "skew_interpretation": (
                    "symmetric" if abs(skewness) < 0.5 else
                    "moderately skewed" if abs(skewness) < 1 else
                    "highly skewed"
                ),
                "skew_direction": "right" if skewness > 0 else "left" if skewness < 0 else "none"
            }
    
    def _generate_insights(self) -> None:
        """Generate automatic insights from the analysis."""
        insights = []
        
        # Size insights
        if self.report.n_rows > 100000:
            insights.append(f"Large dataset with {self.report.n_rows:,} rows - consider sampling for initial analysis")
        
        # Missing value insights
        if self.report.missing_percentage > 10:
            insights.append(f"Significant missing data ({self.report.missing_percentage:.1f}%) - imputation recommended")
        
        # Correlation insights
        if self.report.high_correlations:
            top_corr = self.report.high_correlations[0]
            insights.append(
                f"Strong correlation ({top_corr['correlation']:.2f}) between "
                f"'{top_corr['column1']}' and '{top_corr['column2']}'"
            )
        
        # Outlier insights
        cols_with_outliers = [col for col, data in self.report.outliers.items() if data["percentage"] > 5]
        if cols_with_outliers:
            insights.append(f"{len(cols_with_outliers)} columns have >5% outliers")
        
        # Distribution insights
        non_normal = [col for col, data in self.report.distributions.items() if not data["is_normal"]]
        if non_normal:
            insights.append(f"{len(non_normal)} numeric columns are non-normally distributed")
        
        # Highly skewed columns
        highly_skewed = [
            col for col, data in self.report.distributions.items()
            if data["skew_interpretation"] == "highly skewed"
        ]
        if highly_skewed:
            insights.append(f"Columns {highly_skewed[:3]} are highly skewed - consider transformation")
        
        # Constant columns
        for col, data in self.report.categorical_stats.items():
            if data["unique_count"] == 1:
                insights.append(f"Column '{col}' has constant value - consider removing")
        
        self.report.insights = insights


class EDAAgent(BaseAgent[dict[str, Any]]):
    """
    EDA Agent for automated exploratory data analysis.
    
    Capabilities:
    - Comprehensive statistical analysis
    - Distribution analysis
    - Correlation detection
    - Outlier identification
    - Automatic insight generation
    - Data quality assessment
    """
    
    name: str = "EDAAgent"
    description: str = "Automated exploratory data analysis with insights"
    role: AgentRole = AgentRole.SPECIALIST
    
    def __init__(self, llm_client=None) -> None:
        super().__init__(llm_client or get_llm_service())
    
    def _register_tools(self) -> None:
        """Register EDA tools."""
        
        self.register_tool(AgentTool(
            name="run_eda",
            description="Run complete exploratory data analysis",
            function=self._run_eda,
            parameters={
                "data": {"type": "object", "description": "Data as dict"},
                "target_column": {"type": "string", "description": "Optional target column"}
            },
            required_params=["data"]
        ))
        
        self.register_tool(AgentTool(
            name="analyze_column",
            description="Deep analysis of a specific column",
            function=self._analyze_column,
            parameters={
                "data": {"type": "object"},
                "column_name": {"type": "string"}
            },
            required_params=["data", "column_name"]
        ))
        
        self.register_tool(AgentTool(
            name="find_patterns",
            description="Find patterns and relationships in data",
            function=self._find_patterns,
            parameters={
                "data": {"type": "object"},
                "columns": {"type": "array", "items": {"type": "string"}}
            },
            required_params=["data"]
        ))
        
        self.register_tool(AgentTool(
            name="suggest_cleaning",
            description="Suggest data cleaning steps",
            function=self._suggest_cleaning,
            parameters={
                "data": {"type": "object"}
            },
            required_params=["data"]
        ))
    
    async def _execute_core(self, context: AgentContext) -> dict[str, Any]:
        """Execute EDA based on context."""
        response = await self._llm_client.complete(
            messages=[
                LLMMessage(
                    role="system",
                    content="You are an EDA expert. Provide insights about the data."
                ),
                LLMMessage(role="user", content=context.task_description)
            ]
        )
        
        return {
            "analysis": response.content,
            "next_steps": ["Run full EDA", "Check data quality", "Analyze correlations"]
        }
    
    async def _run_eda(
        self,
        data: dict,
        target_column: str = None
    ) -> dict[str, Any]:
        """Run complete EDA analysis."""
        df = pd.DataFrame(data)
        engine = EDAEngine(df)
        report = engine.run_full_analysis()
        
        return {
            "status": "success",
            "report": report.to_dict()
        }
    
    async def _analyze_column(
        self,
        data: dict,
        column_name: str
    ) -> dict[str, Any]:
        """Deep analysis of a specific column."""
        df = pd.DataFrame(data)
        
        if column_name not in df.columns:
            return {"error": f"Column '{column_name}' not found"}
        
        series = df[column_name]
        
        analysis = {
            "column": column_name,
            "dtype": str(series.dtype),
            "count": len(series),
            "missing": int(series.isnull().sum()),
            "missing_pct": round(series.isnull().sum() / len(series) * 100, 2),
            "unique": int(series.nunique())
        }
        
        if pd.api.types.is_numeric_dtype(series):
            analysis["statistics"] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "median": float(series.median())
            }
        else:
            value_counts = series.value_counts()
            analysis["top_values"] = value_counts.head(10).to_dict()
        
        return analysis
    
    async def _find_patterns(
        self,
        data: dict,
        columns: list[str] = None
    ) -> dict[str, Any]:
        """Find patterns in data."""
        df = pd.DataFrame(data)
        
        if columns:
            df = df[columns]
        
        patterns = {
            "correlations": [],
            "constant_columns": [],
            "high_cardinality": []
        }
        
        # Numeric correlations
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    r = corr.loc[col1, col2]
                    if abs(r) > 0.5:
                        patterns["correlations"].append({
                            "columns": [col1, col2],
                            "correlation": round(r, 3)
                        })
        
        # Constant columns
        for col in df.columns:
            if df[col].nunique() <= 1:
                patterns["constant_columns"].append(col)
        
        # High cardinality
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() > df.shape[0] * 0.5:
                patterns["high_cardinality"].append(col)
        
        return patterns
    
    async def _suggest_cleaning(
        self,
        data: dict
    ) -> dict[str, Any]:
        """Suggest data cleaning steps."""
        df = pd.DataFrame(data)
        
        suggestions = []
        
        # Missing values
        missing = df.isnull().sum()
        for col in missing[missing > 0].index:
            pct = missing[col] / len(df) * 100
            if pct > 50:
                suggestions.append({
                    "column": col,
                    "issue": "High missing values",
                    "action": "Consider dropping column",
                    "priority": "high"
                })
            elif pct > 0:
                suggestions.append({
                    "column": col,
                    "issue": f"{pct:.1f}% missing",
                    "action": "Impute with median/mode",
                    "priority": "medium"
                })
        
        # Duplicates
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            suggestions.append({
                "column": "all",
                "issue": f"{dup_count} duplicate rows",
                "action": "Remove duplicates",
                "priority": "high"
            })
        
        # Constant columns
        for col in df.columns:
            if df[col].nunique() == 1:
                suggestions.append({
                    "column": col,
                    "issue": "Constant value",
                    "action": "Remove column",
                    "priority": "medium"
                })
        
        return {
            "suggestions": suggestions,
            "total_issues": len(suggestions)
        }


# Factory function
def get_eda_agent() -> EDAAgent:
    """Get EDA agent instance."""
    return EDAAgent()
