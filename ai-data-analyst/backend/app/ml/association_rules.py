# AI Enterprise Data Analyst - Association Rules Engine
# Market Basket Analysis with Apriori, FP-Growth algorithms

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations

from app.core.logging import get_logger
try:
    from app.core.exceptions import ValidationException
except ImportError:
    class ValidationException(Exception): pass

logger = get_logger(__name__)


# ============================================================================
# Association Rule Types
# ============================================================================

class AssociationAlgorithm(str, Enum):
    """Association rule mining algorithms."""
    APRIORI = "apriori"
    FP_GROWTH = "fp_growth"
    ECLAT = "eclat"


@dataclass
class ItemSet:
    """Represents a frequent itemset."""
    
    items: frozenset
    support: float
    count: int = 0
    
    def __hash__(self):
        return hash(self.items)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "items": list(self.items),
            "support": round(self.support, 4),
            "count": self.count
        }


@dataclass
class AssociationRule:
    """Represents an association rule."""
    
    antecedent: frozenset  # If
    consequent: frozenset  # Then
    support: float
    confidence: float
    lift: float
    conviction: float = 0.0
    leverage: float = 0.0
    count: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "if": list(self.antecedent),
            "then": list(self.consequent),
            "support": round(self.support, 4),
            "confidence": round(self.confidence, 4),
            "lift": round(self.lift, 4),
            "conviction": round(self.conviction, 4),
            "leverage": round(self.leverage, 6),
            "count": self.count,
            "interpretation": self._interpret()
        }
    
    def _interpret(self) -> str:
        """Human-readable interpretation."""
        if self.lift > 2:
            strength = "strong positive"
        elif self.lift > 1.2:
            strength = "positive"
        elif self.lift < 0.8:
            strength = "negative"
        else:
            strength = "weak"
        
        return f"{strength} association (lift={self.lift:.2f})"


# ============================================================================
# Apriori Algorithm
# ============================================================================

class AprioriAlgorithm:
    """
    Classic Apriori algorithm for frequent itemset mining.
    
    Time Complexity: O(2^n) worst case, but pruning helps
    """
    
    def __init__(
        self,
        min_support: float = 0.01,
        max_length: int = 5
    ):
        self.min_support = min_support
        self.max_length = max_length
    
    def find_frequent_itemsets(
        self,
        transactions: list[set],
        n_transactions: int = None
    ) -> list[ItemSet]:
        """Find all frequent itemsets using Apriori."""
        n = n_transactions or len(transactions)
        min_count = int(self.min_support * n)
        
        # Find frequent 1-itemsets
        item_counts = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                item_counts[frozenset([item])] += 1
        
        frequent_itemsets = []
        current_frequent = []
        
        for itemset, count in item_counts.items():
            if count >= min_count:
                current_frequent.append(itemset)
                frequent_itemsets.append(ItemSet(
                    items=itemset,
                    support=count / n,
                    count=count
                ))
        
        # Generate larger itemsets
        k = 2
        while current_frequent and k <= self.max_length:
            candidates = self._generate_candidates(current_frequent, k)
            
            # Count support
            item_counts = defaultdict(int)
            for transaction in transactions:
                transaction_set = frozenset(transaction)
                for candidate in candidates:
                    if candidate.issubset(transaction_set):
                        item_counts[candidate] += 1
            
            current_frequent = []
            for itemset, count in item_counts.items():
                if count >= min_count:
                    current_frequent.append(itemset)
                    frequent_itemsets.append(ItemSet(
                        items=itemset,
                        support=count / n,
                        count=count
                    ))
            
            k += 1
        
        return frequent_itemsets
    
    def _generate_candidates(
        self,
        frequent_itemsets: list[frozenset],
        k: int
    ) -> set[frozenset]:
        """Generate candidate itemsets of size k."""
        candidates = set()
        
        for i, itemset1 in enumerate(frequent_itemsets):
            for itemset2 in frequent_itemsets[i + 1:]:
                union = itemset1 | itemset2
                if len(union) == k:
                    # Check if all (k-1) subsets are frequent
                    all_subsets_frequent = True
                    for item in union:
                        subset = union - {item}
                        if subset not in frequent_itemsets:
                            all_subsets_frequent = False
                            break
                    
                    if all_subsets_frequent:
                        candidates.add(union)
        
        return candidates


# ============================================================================
# FP-Growth Algorithm
# ============================================================================

class FPNode:
    """Node in FP-Tree."""
    
    def __init__(self, item: Any, parent: 'FPNode' = None):
        self.item = item
        self.count = 0
        self.parent = parent
        self.children: dict[Any, 'FPNode'] = {}
        self.link = None  # Link to next same-item node


class FPTree:
    """FP-Tree data structure for FP-Growth algorithm."""
    
    def __init__(self):
        self.root = FPNode(None)
        self.header_table: dict[Any, FPNode] = {}
        self.item_counts: dict[Any, int] = defaultdict(int)
    
    def insert(self, transaction: list, count: int = 1):
        """Insert a transaction into the tree."""
        current = self.root
        
        for item in transaction:
            self.item_counts[item] += count
            
            if item in current.children:
                current.children[item].count += count
            else:
                new_node = FPNode(item, current)
                new_node.count = count
                current.children[item] = new_node
                
                # Update header table
                if item in self.header_table:
                    node = self.header_table[item]
                    while node.link is not None:
                        node = node.link
                    node.link = new_node
                else:
                    self.header_table[item] = new_node
            
            current = current.children[item]
    
    def get_prefix_paths(self, item: Any) -> list[tuple[list, int]]:
        """Get all prefix paths ending in item."""
        paths = []
        node = self.header_table.get(item)
        
        while node is not None:
            path = []
            current = node.parent
            while current.item is not None:
                path.append(current.item)
                current = current.parent
            
            if path:
                paths.append((path[::-1], node.count))
            
            node = node.link
        
        return paths


class FPGrowthAlgorithm:
    """
    FP-Growth algorithm for frequent itemset mining.
    
    More efficient than Apriori - no candidate generation.
    """
    
    def __init__(
        self,
        min_support: float = 0.01,
        max_length: int = 5
    ):
        self.min_support = min_support
        self.max_length = max_length
    
    def find_frequent_itemsets(
        self,
        transactions: list[set],
        n_transactions: int = None
    ) -> list[ItemSet]:
        """Find all frequent itemsets using FP-Growth."""
        n = n_transactions or len(transactions)
        min_count = int(self.min_support * n)
        
        # Count item frequencies
        item_counts = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                item_counts[item] += 1
        
        # Filter and sort items by frequency
        frequent_items = {
            item: count for item, count in item_counts.items()
            if count >= min_count
        }
        
        if not frequent_items:
            return []
        
        # Sort transactions by item frequency
        def sort_transaction(transaction):
            filtered = [item for item in transaction if item in frequent_items]
            return sorted(filtered, key=lambda x: frequent_items[x], reverse=True)
        
        sorted_transactions = [sort_transaction(t) for t in transactions]
        
        # Build FP-tree
        tree = FPTree()
        for transaction in sorted_transactions:
            if transaction:
                tree.insert(transaction)
        
        # Mine frequent itemsets
        frequent_itemsets = []
        
        # Add single items
        for item, count in frequent_items.items():
            frequent_itemsets.append(ItemSet(
                items=frozenset([item]),
                support=count / n,
                count=count
            ))
        
        # Mine with FP-Growth
        self._mine_tree(tree, [], min_count, n, frequent_itemsets)
        
        return frequent_itemsets
    
    def _mine_tree(
        self,
        tree: FPTree,
        prefix: list,
        min_count: int,
        n: int,
        result: list[ItemSet]
    ):
        """Recursively mine FP-tree."""
        if len(prefix) >= self.max_length:
            return
        
        # Sort items by count (ascending for bottom-up)
        items = sorted(tree.item_counts.items(), key=lambda x: x[1])
        
        for item, count in items:
            if count < min_count:
                continue
            
            new_prefix = prefix + [item]
            
            if len(new_prefix) > 1:
                result.append(ItemSet(
                    items=frozenset(new_prefix),
                    support=count / n,
                    count=count
                ))
            
            # Build conditional FP-tree
            prefix_paths = tree.get_prefix_paths(item)
            
            if prefix_paths:
                conditional_tree = FPTree()
                for path, path_count in prefix_paths:
                    conditional_tree.insert(path, path_count)
                
                self._mine_tree(conditional_tree, new_prefix, min_count, n, result)


# ============================================================================
# Rule Generator
# ============================================================================

class RuleGenerator:
    """Generate association rules from frequent itemsets."""
    
    def __init__(
        self,
        min_confidence: float = 0.5,
        min_lift: float = 1.0
    ):
        self.min_confidence = min_confidence
        self.min_lift = min_lift
    
    def generate_rules(
        self,
        frequent_itemsets: list[ItemSet],
        n_transactions: int
    ) -> list[AssociationRule]:
        """Generate association rules from frequent itemsets."""
        # Build support lookup
        support_map = {
            itemset.items: itemset.support
            for itemset in frequent_itemsets
        }
        count_map = {
            itemset.items: itemset.count
            for itemset in frequent_itemsets
        }
        
        rules = []
        
        for itemset in frequent_itemsets:
            if len(itemset.items) < 2:
                continue
            
            # Generate all non-empty subsets as antecedents
            items = list(itemset.items)
            
            for i in range(1, len(items)):
                for antecedent_tuple in combinations(items, i):
                    antecedent = frozenset(antecedent_tuple)
                    consequent = itemset.items - antecedent
                    
                    if not consequent:
                        continue
                    
                    ant_support = support_map.get(antecedent, 0)
                    con_support = support_map.get(consequent, 0)
                    
                    if ant_support == 0 or con_support == 0:
                        continue
                    
                    # Calculate metrics
                    confidence = itemset.support / ant_support
                    lift = confidence / con_support
                    
                    if confidence < self.min_confidence or lift < self.min_lift:
                        continue
                    
                    # Conviction
                    if confidence < 1:
                        conviction = (1 - con_support) / (1 - confidence)
                    else:
                        conviction = float('inf')
                    
                    # Leverage
                    leverage = itemset.support - (ant_support * con_support)
                    
                    rules.append(AssociationRule(
                        antecedent=antecedent,
                        consequent=consequent,
                        support=itemset.support,
                        confidence=confidence,
                        lift=lift,
                        conviction=conviction if conviction != float('inf') else 999.99,
                        leverage=leverage,
                        count=count_map.get(itemset.items, 0)
                    ))
        
        # Sort by lift then confidence
        rules.sort(key=lambda r: (r.lift, r.confidence), reverse=True)
        
        return rules


# ============================================================================
# Association Rules Engine
# ============================================================================

class AssociationRulesEngine:
    """
    Production association rules engine.
    
    Features:
    - Multiple algorithms (Apriori, FP-Growth)
    - Rule generation with metrics
    - Product recommendations
    - Cross-sell analysis
    """
    
    def __init__(
        self,
        min_support: float = 0.01,
        min_confidence: float = 0.5,
        min_lift: float = 1.0,
        algorithm: AssociationAlgorithm = AssociationAlgorithm.FP_GROWTH
    ):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.algorithm = algorithm
        
        self._frequent_itemsets: list[ItemSet] = []
        self._rules: list[AssociationRule] = []
    
    def analyze(
        self,
        df: pd.DataFrame,
        transaction_col: str,
        item_col: str
    ) -> dict[str, Any]:
        """
        Run market basket analysis.
        
        Args:
            df: Transaction data
            transaction_col: Column with transaction/basket ID
            item_col: Column with item names
        """
        # Group items by transaction
        transactions = (
            df.groupby(transaction_col)[item_col]
            .apply(set)
            .tolist()
        )
        
        n_transactions = len(transactions)
        
        logger.info(f"Analyzing {n_transactions} transactions with {self.algorithm.value}")
        
        # Find frequent itemsets
        if self.algorithm == AssociationAlgorithm.APRIORI:
            algo = AprioriAlgorithm(self.min_support)
        else:
            algo = FPGrowthAlgorithm(self.min_support)
        
        self._frequent_itemsets = algo.find_frequent_itemsets(
            transactions, n_transactions
        )
        
        # Generate rules
        rule_gen = RuleGenerator(self.min_confidence, self.min_lift)
        self._rules = rule_gen.generate_rules(
            self._frequent_itemsets, n_transactions
        )
        
        return {
            "algorithm": self.algorithm.value,
            "n_transactions": n_transactions,
            "n_frequent_itemsets": len(self._frequent_itemsets),
            "n_rules": len(self._rules),
            "top_itemsets": [
                i.to_dict() 
                for i in sorted(self._frequent_itemsets, key=lambda x: x.support, reverse=True)[:20]
            ],
            "top_rules": [r.to_dict() for r in self._rules[:50]],
            "summary": self._generate_summary()
        }
    
    def get_recommendations(
        self,
        basket: list[str],
        top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Get product recommendations for items in basket."""
        basket_set = frozenset(basket)
        recommendations = []
        
        for rule in self._rules:
            if rule.antecedent.issubset(basket_set):
                # Don't recommend items already in basket
                new_items = rule.consequent - basket_set
                if new_items:
                    recommendations.append({
                        "items": list(new_items),
                        "confidence": round(rule.confidence, 4),
                        "lift": round(rule.lift, 4),
                        "because": list(rule.antecedent)
                    })
        
        # Deduplicate and sort
        seen = set()
        unique_recs = []
        for rec in recommendations:
            key = frozenset(rec["items"])
            if key not in seen:
                seen.add(key)
                unique_recs.append(rec)
        
        return sorted(unique_recs, key=lambda x: x["lift"], reverse=True)[:top_k]
    
    def cross_sell_matrix(
        self,
        items: list[str] = None
    ) -> pd.DataFrame:
        """Generate cross-sell matrix for items."""
        if items is None:
            # Get top items
            item_supports = defaultdict(float)
            for itemset in self._frequent_itemsets:
                if len(itemset.items) == 1:
                    item_supports[list(itemset.items)[0]] = itemset.support
            
            items = sorted(item_supports.keys(), key=lambda x: item_supports[x], reverse=True)[:10]
        
        matrix = pd.DataFrame(index=items, columns=items, data=0.0)
        
        for rule in self._rules:
            for ant in rule.antecedent:
                for con in rule.consequent:
                    if ant in items and con in items:
                        matrix.loc[ant, con] = max(matrix.loc[ant, con], rule.lift)
        
        return matrix
    
    def _generate_summary(self) -> dict[str, Any]:
        """Generate analysis summary."""
        if not self._rules:
            return {"message": "No significant rules found"}
        
        top_rule = self._rules[0]
        avg_lift = np.mean([r.lift for r in self._rules])
        
        return {
            "strongest_association": {
                "if_buy": list(top_rule.antecedent),
                "likely_buy": list(top_rule.consequent),
                "lift": round(top_rule.lift, 2),
                "confidence": f"{top_rule.confidence * 100:.1f}%"
            },
            "average_lift": round(avg_lift, 2),
            "strong_rules": len([r for r in self._rules if r.lift > 2]),
            "insight": f"Customers who buy {list(top_rule.antecedent)} are {top_rule.lift:.1f}x more likely to buy {list(top_rule.consequent)}"
        }


# Factory function
def get_association_rules_engine(
    min_support: float = 0.01,
    min_confidence: float = 0.5
) -> AssociationRulesEngine:
    """Get association rules engine instance."""
    return AssociationRulesEngine(min_support, min_confidence)
