# AI Enterprise Data Analyst - Market Basket Analysis Engine
# Production-grade association rules and basket analysis
# Handles: any transaction data, Apriori algorithm

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

from app.core.logging import get_logger
try:
    from app.core.exceptions import DataProcessingException
except ImportError:
    class DataProcessingException(Exception): pass

logger = get_logger(__name__)
warnings.filterwarnings('ignore')


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class AssociationRule:
    """Single association rule."""
    antecedent: Tuple[str, ...]  # Left side (if)
    consequent: Tuple[str, ...]  # Right side (then)
    support: float
    confidence: float
    lift: float
    conviction: float
    leverage: float
    antecedent_support: float
    consequent_support: float


@dataclass
class FrequentItemset:
    """Frequent itemset."""
    items: Tuple[str, ...]
    support: float
    count: int


@dataclass
class BasketConfig:
    """Configuration for basket analysis."""
    # Column mappings
    transaction_id_col: Optional[str] = None
    item_col: Optional[str] = None
    
    # Algorithm parameters
    min_support: float = 0.01  # Minimum support threshold
    min_confidence: float = 0.5  # Minimum confidence threshold
    min_lift: float = 1.0  # Minimum lift threshold
    max_itemset_size: int = 4  # Maximum items in itemset


@dataclass
class BasketResult:
    """Complete basket analysis result."""
    n_transactions: int = 0
    n_unique_items: int = 0
    avg_basket_size: float = 0.0
    
    # Frequent itemsets
    frequent_itemsets: List[FrequentItemset] = field(default_factory=list)
    
    # Association rules
    rules: List[AssociationRule] = field(default_factory=list)
    
    # Top associations
    top_pairs: List[Dict[str, Any]] = field(default_factory=list)
    
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "n_transactions": self.n_transactions,
                "n_unique_items": self.n_unique_items,
                "avg_basket_size": round(self.avg_basket_size, 2),
                "n_rules": len(self.rules)
            },
            "top_rules": [
                {
                    "if_buy": list(r.antecedent),
                    "then_buy": list(r.consequent),
                    "support": round(r.support, 4),
                    "confidence": round(r.confidence, 4),
                    "lift": round(r.lift, 2)
                }
                for r in sorted(self.rules, key=lambda x: -x.lift)[:20]
            ],
            "top_pairs": self.top_pairs[:20]
        }


# ============================================================================
# Apriori Algorithm Implementation
# ============================================================================

class AprioriAlgorithm:
    """Apriori algorithm for frequent itemset mining."""
    
    def __init__(self, min_support: float, max_size: int):
        self.min_support = min_support
        self.max_size = max_size
        self._n_transactions = 0
    
    def find_frequent_itemsets(
        self,
        transactions: List[Set[str]]
    ) -> List[FrequentItemset]:
        """Find all frequent itemsets using Apriori."""
        self._n_transactions = len(transactions)
        min_count = int(self.min_support * self._n_transactions)
        
        # Find frequent 1-itemsets
        item_counts = defaultdict(int)
        for trans in transactions:
            for item in trans:
                item_counts[item] += 1
        
        frequent_1 = {
            frozenset([item]): count 
            for item, count in item_counts.items() 
            if count >= min_count
        }
        
        all_frequent = []
        for items, count in frequent_1.items():
            all_frequent.append(FrequentItemset(
                items=tuple(sorted(items)),
                support=count / self._n_transactions,
                count=count
            ))
        
        current_frequent = frequent_1
        k = 2
        
        while current_frequent and k <= self.max_size:
            # Generate candidates
            candidates = self._generate_candidates(current_frequent, k)
            
            # Count support
            candidate_counts = defaultdict(int)
            for trans in transactions:
                trans_set = frozenset(trans)
                for candidate in candidates:
                    if candidate.issubset(trans_set):
                        candidate_counts[candidate] += 1
            
            # Filter by min support
            current_frequent = {
                items: count 
                for items, count in candidate_counts.items() 
                if count >= min_count
            }
            
            for items, count in current_frequent.items():
                all_frequent.append(FrequentItemset(
                    items=tuple(sorted(items)),
                    support=count / self._n_transactions,
                    count=count
                ))
            
            k += 1
        
        return all_frequent
    
    def _generate_candidates(
        self,
        frequent_itemsets: Dict[frozenset, int],
        k: int
    ) -> List[frozenset]:
        """Generate candidate itemsets of size k."""
        items = set()
        for itemset in frequent_itemsets:
            items.update(itemset)
        
        candidates = []
        for combo in combinations(items, k):
            candidate = frozenset(combo)
            # Check if all k-1 subsets are frequent
            is_valid = True
            for item in candidate:
                subset = candidate - {item}
                if subset not in frequent_itemsets:
                    is_valid = False
                    break
            if is_valid:
                candidates.append(candidate)
        
        return candidates


# ============================================================================
# Rule Generator
# ============================================================================

class RuleGenerator:
    """Generate association rules from frequent itemsets."""
    
    def __init__(
        self,
        min_confidence: float,
        min_lift: float,
        n_transactions: int
    ):
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.n_transactions = n_transactions
    
    def generate_rules(
        self,
        frequent_itemsets: List[FrequentItemset]
    ) -> List[AssociationRule]:
        """Generate association rules."""
        # Create support lookup
        support_lookup = {
            frozenset(fi.items): fi.support 
            for fi in frequent_itemsets
        }
        
        rules = []
        
        for itemset in frequent_itemsets:
            if len(itemset.items) < 2:
                continue
            
            items = frozenset(itemset.items)
            itemset_support = itemset.support
            
            # Generate all possible rules from this itemset
            for i in range(1, len(items)):
                for antecedent in combinations(items, i):
                    antecedent = frozenset(antecedent)
                    consequent = items - antecedent
                    
                    # Get supports
                    ant_support = support_lookup.get(antecedent, 0)
                    con_support = support_lookup.get(consequent, 0)
                    
                    if ant_support == 0 or con_support == 0:
                        continue
                    
                    # Calculate metrics
                    confidence = itemset_support / ant_support
                    lift = confidence / con_support if con_support > 0 else 0
                    
                    # Apply thresholds
                    if confidence < self.min_confidence or lift < self.min_lift:
                        continue
                    
                    # Calculate additional metrics
                    leverage = itemset_support - (ant_support * con_support)
                    conviction = (1 - con_support) / (1 - confidence) if confidence < 1 else float('inf')
                    
                    rules.append(AssociationRule(
                        antecedent=tuple(sorted(antecedent)),
                        consequent=tuple(sorted(consequent)),
                        support=itemset_support,
                        confidence=confidence,
                        lift=lift,
                        conviction=conviction if conviction != float('inf') else 999,
                        leverage=leverage,
                        antecedent_support=ant_support,
                        consequent_support=con_support
                    ))
        
        return rules


# ============================================================================
# Basket Analysis Engine
# ============================================================================

class BasketAnalysisEngine:
    """
    Complete Market Basket Analysis engine.
    
    Features:
    - Apriori algorithm for frequent itemsets
    - Association rule generation
    - Multiple metrics (support, confidence, lift)
    - Product recommendations
    """
    
    def __init__(self, config: BasketConfig = None, verbose: bool = True):
        self.config = config or BasketConfig()
        self.verbose = verbose
    
    def analyze(
        self,
        df: pd.DataFrame,
        transaction_id_col: str = None,
        item_col: str = None
    ) -> BasketResult:
        """Perform basket analysis."""
        start_time = datetime.now()
        
        # Auto-detect columns
        transaction_id_col = transaction_id_col or self._detect_transaction_col(df)
        item_col = item_col or self._detect_item_col(df)
        
        if self.verbose:
            logger.info(f"Basket analysis: transaction={transaction_id_col}, item={item_col}")
        
        # Convert to list of sets (transactions)
        transactions = (
            df.groupby(transaction_id_col)[item_col]
            .apply(set)
            .tolist()
        )
        
        n_transactions = len(transactions)
        n_items = df[item_col].nunique()
        avg_basket = df.groupby(transaction_id_col)[item_col].count().mean()
        
        if self.verbose:
            logger.info(f"Processing {n_transactions} transactions, {n_items} unique items")
        
        # Find frequent itemsets
        apriori = AprioriAlgorithm(
            min_support=self.config.min_support,
            max_size=self.config.max_itemset_size
        )
        frequent_itemsets = apriori.find_frequent_itemsets(transactions)
        
        if self.verbose:
            logger.info(f"Found {len(frequent_itemsets)} frequent itemsets")
        
        # Generate rules
        rule_gen = RuleGenerator(
            min_confidence=self.config.min_confidence,
            min_lift=self.config.min_lift,
            n_transactions=n_transactions
        )
        rules = rule_gen.generate_rules(frequent_itemsets)
        
        if self.verbose:
            logger.info(f"Generated {len(rules)} rules")
        
        # Top pairs
        top_pairs = self._get_top_pairs(frequent_itemsets)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return BasketResult(
            n_transactions=n_transactions,
            n_unique_items=n_items,
            avg_basket_size=avg_basket,
            frequent_itemsets=frequent_itemsets,
            rules=rules,
            top_pairs=top_pairs,
            processing_time_sec=processing_time
        )
    
    def get_recommendations(
        self,
        items: List[str],
        result: BasketResult,
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """Get product recommendations based on items."""
        recommendations = []
        items_set = set(items)
        
        for rule in result.rules:
            if set(rule.antecedent).issubset(items_set):
                for item in rule.consequent:
                    if item not in items_set:
                        recommendations.append({
                            "item": item,
                            "confidence": rule.confidence,
                            "lift": rule.lift,
                            "based_on": list(rule.antecedent)
                        })
        
        # Sort by lift, remove duplicates
        seen = set()
        unique_recs = []
        for rec in sorted(recommendations, key=lambda x: -x['lift']):
            if rec['item'] not in seen:
                seen.add(rec['item'])
                unique_recs.append(rec)
        
        return unique_recs[:top_n]
    
    def _get_top_pairs(
        self,
        frequent_itemsets: List[FrequentItemset]
    ) -> List[Dict[str, Any]]:
        """Get top frequently bought together pairs."""
        pairs = [fi for fi in frequent_itemsets if len(fi.items) == 2]
        pairs.sort(key=lambda x: -x.support)
        
        return [
            {
                "items": list(p.items),
                "support": round(p.support, 4),
                "count": p.count
            }
            for p in pairs[:20]
        ]
    
    def _detect_transaction_col(self, df: pd.DataFrame) -> str:
        patterns = ['transaction', 'order', 'invoice', 'basket', 'cart', 'bill', 'receipt']
        for col in df.columns:
            if any(p in col.lower() for p in patterns):
                return col
        return df.columns[0]
    
    def _detect_item_col(self, df: pd.DataFrame) -> str:
        patterns = ['item', 'product', 'sku', 'article', 'good']
        for col in df.columns:
            if any(p in col.lower() for p in patterns):
                return col
        # Return first string column that's not transaction
        for col in df.columns:
            if df[col].dtype == 'object':
                return col
        return df.columns[1] if len(df.columns) > 1 else df.columns[0]


# ============================================================================
# Factory Functions
# ============================================================================

def get_basket_engine(config: BasketConfig = None) -> BasketAnalysisEngine:
    """Get basket analysis engine."""
    return BasketAnalysisEngine(config=config)


def quick_basket(
    df: pd.DataFrame,
    min_support: float = 0.01,
    min_confidence: float = 0.5
) -> Dict[str, Any]:
    """Quick basket analysis."""
    config = BasketConfig(min_support=min_support, min_confidence=min_confidence)
    engine = BasketAnalysisEngine(config=config, verbose=False)
    result = engine.analyze(df)
    return result.to_dict()
