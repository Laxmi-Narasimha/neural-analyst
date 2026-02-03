# AI Enterprise Data Analyst - Inventory Analysis Engine
# Production-grade inventory and stock analysis
# Handles: stock levels, turnover, ABC analysis, reorder points

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
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

class ABCCategory(str, Enum):
    """ABC inventory classification."""
    A = "A"  # High value, low quantity
    B = "B"  # Medium value, medium quantity
    C = "C"  # Low value, high quantity


class StockStatus(str, Enum):
    """Stock status."""
    OUT_OF_STOCK = "out_of_stock"
    LOW_STOCK = "low_stock"
    NORMAL = "normal"
    OVERSTOCK = "overstock"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class InventoryItem:
    """Inventory item analysis."""
    item_id: str
    current_stock: float
    avg_daily_sales: float
    days_of_stock: float
    abc_category: ABCCategory
    stock_status: StockStatus
    reorder_point: float
    reorder_quantity: float
    value: float


@dataclass
class InventorySummary:
    """Inventory summary metrics."""
    total_items: int
    total_value: float
    total_units: float
    
    # Stock status counts
    out_of_stock: int
    low_stock: int
    overstock: int
    
    # ABC distribution
    a_items: int
    b_items: int
    c_items: int
    
    # Key metrics
    avg_turnover: float
    avg_days_of_stock: float


@dataclass
class InventoryResult:
    """Complete inventory analysis result."""
    summary: InventorySummary = None
    items: List[InventoryItem] = field(default_factory=list)
    
    # Recommendations
    reorder_list: List[str] = field(default_factory=list)
    overstock_list: List[str] = field(default_factory=list)
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "total_items": self.summary.total_items if self.summary else 0,
                "total_value": round(self.summary.total_value, 2) if self.summary else 0,
                "out_of_stock": self.summary.out_of_stock if self.summary else 0,
                "low_stock": self.summary.low_stock if self.summary else 0,
                "overstock": self.summary.overstock if self.summary else 0,
                "avg_turnover": round(self.summary.avg_turnover, 2) if self.summary else 0
            },
            "abc_distribution": {
                "A": self.summary.a_items if self.summary else 0,
                "B": self.summary.b_items if self.summary else 0,
                "C": self.summary.c_items if self.summary else 0
            },
            "reorder_needed": self.reorder_list[:20],
            "overstock_alert": self.overstock_list[:10]
        }


# ============================================================================
# Inventory Analysis Engine
# ============================================================================

class InventoryAnalysisEngine:
    """
    Production-grade Inventory Analysis engine.
    
    Features:
    - ABC classification
    - Stock status assessment
    - Reorder point calculation
    - Turnover analysis
    - Days of stock
    """
    
    def __init__(
        self,
        lead_time_days: int = 7,
        safety_stock_days: int = 3,
        verbose: bool = True
    ):
        self.lead_time_days = lead_time_days
        self.safety_stock_days = safety_stock_days
        self.verbose = verbose
    
    def analyze(
        self,
        df: pd.DataFrame,
        item_col: str = None,
        quantity_col: str = None,
        sales_col: str = None,
        value_col: str = None,
        date_col: str = None
    ) -> InventoryResult:
        """Analyze inventory."""
        start_time = datetime.now()
        
        # Auto-detect columns
        if item_col is None:
            item_col = self._detect_column(df, ['item', 'product', 'sku', 'id'])
        if quantity_col is None:
            quantity_col = self._detect_column(df, ['quantity', 'stock', 'qty', 'inventory'])
        if sales_col is None:
            sales_col = self._detect_column(df, ['sales', 'sold', 'demand'])
        if value_col is None:
            value_col = self._detect_column(df, ['value', 'price', 'cost'])
        
        if self.verbose:
            logger.info(f"Inventory analysis: {len(df)} records")
        
        # Aggregate by item
        agg_dict = {}
        if quantity_col and quantity_col in df.columns:
            agg_dict['current_stock'] = (quantity_col, 'sum')
        if sales_col and sales_col in df.columns:
            agg_dict['total_sales'] = (sales_col, 'sum')
        if value_col and value_col in df.columns:
            agg_dict['total_value'] = (value_col, 'sum')
        
        if item_col and item_col in df.columns and agg_dict:
            inventory = df.groupby(item_col).agg(**agg_dict).reset_index()
        else:
            # Work with data as-is
            inventory = df.copy()
        
        # Ensure required columns
        if 'current_stock' not in inventory.columns:
            inventory['current_stock'] = 100  # Default
        if 'total_sales' not in inventory.columns:
            inventory['total_sales'] = 10  # Default
        if 'total_value' not in inventory.columns:
            inventory['total_value'] = inventory['current_stock'] * 10
        
        # Calculate days in data
        n_days = 30  # Default
        if date_col and date_col in df.columns:
            dates = pd.to_datetime(df[date_col], errors='coerce')
            n_days = max(1, (dates.max() - dates.min()).days)
        
        # Calculate metrics
        inventory['avg_daily_sales'] = inventory['total_sales'] / n_days
        inventory['days_of_stock'] = np.where(
            inventory['avg_daily_sales'] > 0,
            inventory['current_stock'] / inventory['avg_daily_sales'],
            999
        )
        
        # ABC Classification
        inventory = self._abc_classification(inventory)
        
        # Reorder points
        inventory['reorder_point'] = (
            inventory['avg_daily_sales'] * 
            (self.lead_time_days + self.safety_stock_days)
        )
        
        inventory['reorder_quantity'] = (
            inventory['avg_daily_sales'] * 30  # 30-day supply
        )
        
        # Stock status
        inventory['stock_status'] = inventory.apply(
            lambda row: self._determine_status(
                row['current_stock'],
                row['reorder_point'],
                row['avg_daily_sales']
            ),
            axis=1
        )
        
        # Create item objects
        items = []
        for _, row in inventory.iterrows():
            items.append(InventoryItem(
                item_id=str(row.get(item_col, row.name)),
                current_stock=float(row['current_stock']),
                avg_daily_sales=float(row['avg_daily_sales']),
                days_of_stock=float(row['days_of_stock']),
                abc_category=ABCCategory(row['abc_category']),
                stock_status=StockStatus(row['stock_status']),
                reorder_point=float(row['reorder_point']),
                reorder_quantity=float(row['reorder_quantity']),
                value=float(row['total_value'])
            ))
        
        # Summary
        summary = InventorySummary(
            total_items=len(inventory),
            total_value=float(inventory['total_value'].sum()),
            total_units=float(inventory['current_stock'].sum()),
            out_of_stock=int((inventory['stock_status'] == 'out_of_stock').sum()),
            low_stock=int((inventory['stock_status'] == 'low_stock').sum()),
            overstock=int((inventory['stock_status'] == 'overstock').sum()),
            a_items=int((inventory['abc_category'] == 'A').sum()),
            b_items=int((inventory['abc_category'] == 'B').sum()),
            c_items=int((inventory['abc_category'] == 'C').sum()),
            avg_turnover=float(inventory['total_sales'].sum() / inventory['current_stock'].sum()) if inventory['current_stock'].sum() > 0 else 0,
            avg_days_of_stock=float(inventory['days_of_stock'].mean())
        )
        
        # Lists
        reorder_mask = inventory['current_stock'] <= inventory['reorder_point']
        reorder_list = inventory[reorder_mask][item_col].tolist() if item_col in inventory.columns else []
        
        overstock_mask = inventory['stock_status'] == 'overstock'
        overstock_list = inventory[overstock_mask][item_col].tolist() if item_col in inventory.columns else []
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return InventoryResult(
            summary=summary,
            items=items,
            reorder_list=reorder_list,
            overstock_list=overstock_list,
            processing_time_sec=processing_time
        )
    
    def _abc_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform ABC classification."""
        df = df.copy()
        df['value_rank'] = df['total_value'].rank(ascending=False, pct=True)
        
        df['abc_category'] = 'C'
        df.loc[df['value_rank'] <= 0.2, 'abc_category'] = 'A'
        df.loc[(df['value_rank'] > 0.2) & (df['value_rank'] <= 0.5), 'abc_category'] = 'B'
        
        return df
    
    def _determine_status(
        self,
        current_stock: float,
        reorder_point: float,
        avg_daily_sales: float
    ) -> str:
        """Determine stock status."""
        if current_stock <= 0:
            return 'out_of_stock'
        
        if current_stock <= reorder_point:
            return 'low_stock'
        
        # Overstock: more than 90 days supply
        if avg_daily_sales > 0 and current_stock / avg_daily_sales > 90:
            return 'overstock'
        
        return 'normal'
    
    def _detect_column(self, df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
        """Detect column by patterns."""
        for pattern in patterns:
            for col in df.columns:
                if pattern in col.lower():
                    return col
        return None


# ============================================================================
# Factory Functions
# ============================================================================

def get_inventory_engine() -> InventoryAnalysisEngine:
    """Get inventory analysis engine."""
    return InventoryAnalysisEngine()


def quick_inventory_analysis(
    df: pd.DataFrame,
    item_col: str = None,
    quantity_col: str = None
) -> Dict[str, Any]:
    """Quick inventory analysis."""
    engine = InventoryAnalysisEngine(verbose=False)
    result = engine.analyze(df, item_col=item_col, quantity_col=quantity_col)
    return result.to_dict()


def abc_classify(
    items: List[str],
    values: List[float]
) -> Dict[str, str]:
    """Quick ABC classification."""
    df = pd.DataFrame({'item': items, 'total_value': values, 'current_stock': 1, 'total_sales': 1})
    engine = InventoryAnalysisEngine(verbose=False)
    result = engine.analyze(df, item_col='item')
    return {item.item_id: item.abc_category.value for item in result.items}
