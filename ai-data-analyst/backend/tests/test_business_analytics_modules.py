# AI Enterprise Data Analyst - Integration Tests: Business Analytics Modules
# Comprehensive tests for business analytics modules

import sys
import os
import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.synthetic_data_generator import SyntheticDataGenerator


class TestRFMAnalysis(unittest.TestCase):
    """Tests for rfm_analysis module."""
    
    @classmethod
    def setUpClass(cls):
        cls.generator = SyntheticDataGenerator(seed=42)
        cls.transaction_data = cls.generator.generate_transaction_data(n_transactions=2000, n_customers=200)
    
    def test_rfm_analysis(self):
        """Test RFM analysis."""
        from app.ml.rfm_analysis import RFMAnalysisEngine
        
        # Clean data
        data = self.transaction_data.dropna(subset=['customer_id', 'transaction_date', 'amount'])
        data = data[data['amount'] > 0]
        
        if len(data) < 100:
            self.skipTest("Not enough valid data")
        
        engine = RFMAnalysisEngine()
        result = engine.analyze(
            data,
            customer_id_col='customer_id',
            date_col='transaction_date',
            amount_col='amount'
        )
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.customer_rfm)
    
    def test_rfm_segments(self):
        """Test RFM segment assignment."""
        from app.ml.rfm_analysis import RFMAnalysisEngine
        
        data = self.transaction_data.dropna(subset=['customer_id', 'transaction_date', 'amount'])
        data = data[data['amount'] > 0]
        
        if len(data) < 100:
            self.skipTest("Not enough valid data")
        
        engine = RFMAnalysisEngine()
        result = engine.analyze(
            data,
            customer_id_col='customer_id',
            date_col='transaction_date',
            amount_col='amount'
        )
        
        self.assertIsNotNone(result.segment_summary)


class TestCohortAnalysis(unittest.TestCase):
    """Tests for cohort_analysis module."""
    
    @classmethod
    def setUpClass(cls):
        cls.generator = SyntheticDataGenerator(seed=42)
        cls.transaction_data = cls.generator.generate_transaction_data(n_transactions=2000, n_customers=200)
    
    def test_retention_cohort(self):
        """Test retention cohort analysis."""
        from app.ml.cohort_analysis import CohortAnalysisEngine
        
        data = self.transaction_data.dropna(subset=['customer_id', 'transaction_date'])
        
        if len(data) < 100:
            self.skipTest("Not enough valid data")
        
        engine = CohortAnalysisEngine()
        result = engine.analyze(
            data,
            customer_id_col='customer_id',
            date_col='transaction_date'
        )
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.cohort_matrix)


class TestFunnelAnalysis(unittest.TestCase):
    """Tests for funnel_analysis module."""
    
    @classmethod
    def setUpClass(cls):
        # Create funnel event data
        np.random.seed(42)
        n_users = 1000
        
        events = []
        for user in range(n_users):
            events.append({'user_id': user, 'event': 'page_view', 'timestamp': datetime.now()})
            if np.random.random() < 0.7:
                events.append({'user_id': user, 'event': 'add_to_cart', 'timestamp': datetime.now()})
                if np.random.random() < 0.5:
                    events.append({'user_id': user, 'event': 'checkout', 'timestamp': datetime.now()})
                    if np.random.random() < 0.6:
                        events.append({'user_id': user, 'event': 'purchase', 'timestamp': datetime.now()})
        
        cls.funnel_data = pd.DataFrame(events)
    
    def test_funnel_analysis(self):
        """Test funnel analysis."""
        from app.ml.funnel_analysis import FunnelAnalysisEngine
        
        steps = ['page_view', 'add_to_cart', 'checkout', 'purchase']
        
        engine = FunnelAnalysisEngine()
        result = engine.analyze(
            self.funnel_data,
            steps=steps,
            user_id_col='user_id',
            event_col='event'
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.n_steps, 4)
        self.assertGreater(result.overall_conversion, 0)


class TestPriceAnalysis(unittest.TestCase):
    """Tests for price_analysis module."""
    
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        # Create price/quantity data
        n = 200
        prices = np.random.uniform(10, 100, n)
        quantities = 1000 / prices + np.random.normal(0, 10, n)  # Inverse relationship
        quantities = np.maximum(1, quantities)
        
        cls.price_data = pd.DataFrame({
            'price': prices,
            'quantity': quantities,
            'product': ['A', 'B', 'C'] * (n // 3) + ['A'] * (n % 3)
        })
    
    def test_price_elasticity(self):
        """Test price elasticity calculation."""
        from app.ml.price_analysis import PriceAnalysisEngine
        
        engine = PriceAnalysisEngine()
        result = engine.analyze(
            self.price_data,
            price_col='price',
            quantity_col='quantity'
        )
        
        self.assertIsNotNone(result)


class TestInventoryAnalysis(unittest.TestCase):
    """Tests for inventory_analysis module."""
    
    @classmethod
    def setUpClass(cls):
        cls.generator = SyntheticDataGenerator(seed=42)
        cls.inventory_data = cls.generator.generate_inventory_data(n_products=100)
    
    def test_abc_classification(self):
        """Test ABC classification."""
        from app.ml.inventory_analysis import InventoryAnalysisEngine
        
        engine = InventoryAnalysisEngine()
        result = engine.analyze(self.inventory_data)
        
        self.assertIsNotNone(result)


class TestGrowthAnalysis(unittest.TestCase):
    """Tests for growth_analysis module."""
    
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        # Create growth data
        dates = pd.date_range('2020-01-01', periods=24, freq='M')
        values = 100 * (1.05 ** np.arange(24)) + np.random.normal(0, 5, 24)
        
        cls.growth_data = pd.DataFrame({
            'date': dates,
            'value': values
        })
    
    def test_growth_metrics(self):
        """Test growth metrics calculation."""
        from app.ml.growth_analysis import GrowthAnalysisEngine
        
        engine = GrowthAnalysisEngine()
        result = engine.analyze(
            self.growth_data,
            date_col='date',
            value_col='value'
        )
        
        self.assertIsNotNone(result)


class TestSessionAnalysis(unittest.TestCase):
    """Tests for session_analysis module."""
    
    @classmethod
    def setUpClass(cls):
        cls.generator = SyntheticDataGenerator(seed=42)
        cls.session_data = cls.generator.generate_user_session_data(n_events=2000, n_users=100)
    
    def test_session_reconstruction(self):
        """Test session reconstruction."""
        from app.ml.session_analysis import SessionAnalysisEngine
        
        data = self.session_data.dropna(subset=['user_id', 'timestamp'])
        
        if len(data) < 100:
            self.skipTest("Not enough valid data")
        
        engine = SessionAnalysisEngine()
        result = engine.analyze(
            data,
            user_col='user_id',
            timestamp_col='timestamp'
        )
        
        self.assertIsNotNone(result)


class TestMetricCalculator(unittest.TestCase):
    """Tests for metric_calculator module."""
    
    @classmethod
    def setUpClass(cls):
        cls.generator = SyntheticDataGenerator(seed=42)
        cls.transaction_data = cls.generator.generate_transaction_data(n_transactions=1000)
    
    def test_revenue_metrics(self):
        """Test revenue metrics calculation."""
        from app.ml.metric_calculator import MetricCalculatorEngine
        
        data = self.transaction_data.dropna(subset=['amount'])
        data = data[data['amount'] > 0]
        # Filter out infinity values
        data = data[~np.isinf(data['amount'])]
        
        engine = MetricCalculatorEngine()
        result = engine.calculate_sum(data, metric_name='total_revenue', value_col='amount')
        
        self.assertIsNotNone(result)


class TestProfitAnalysis(unittest.TestCase):
    """Tests for profit_analysis module."""
    
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        n = 100
        cls.profit_data = pd.DataFrame({
            'revenue': np.random.uniform(1000, 10000, n),
            'cost': np.random.uniform(500, 5000, n),
            'units': np.random.randint(10, 100, n)
        })
        cls.profit_data['profit'] = cls.profit_data['revenue'] - cls.profit_data['cost']
    
    def test_profit_analysis(self):
        """Test profit analysis."""
        from app.ml.profit_analysis import ProfitAnalysisEngine
        
        engine = ProfitAnalysisEngine()
        result = engine.analyze(
            self.profit_data,
            revenue_col='revenue',
            cost_col='cost'
        )
        
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
