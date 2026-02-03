# AI Enterprise Data Analyst - Integration Tests: Infrastructure Modules
# Comprehensive tests for data catalog, lineage, alerting, and reporting modules

import sys
import os
import unittest
from datetime import datetime, timedelta
from typing import Dict, Any

import numpy as np
import pandas as pd

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.synthetic_data_generator import SyntheticDataGenerator


class TestDataCatalog(unittest.TestCase):
    """Tests for data_catalog module."""
    
    @classmethod
    def setUpClass(cls):
        cls.generator = SyntheticDataGenerator(seed=42)
        cls.customer_data = cls.generator.generate_customer_data(n_customers=100)
        cls.transaction_data = cls.generator.generate_transaction_data(n_transactions=500)
    
    def test_register_dataset(self):
        """Test dataset registration."""
        from app.ml.data_catalog import DataCatalogEngine
        
        engine = DataCatalogEngine()
        dataset_id = engine.register_dataset(
            name='customers',
            df=self.customer_data,
            description='Customer master data',
            tags=['customers', 'master']
        )
        
        self.assertIsNotNone(dataset_id)
    
    def test_schema_inference(self):
        """Test schema is inferred correctly."""
        from app.ml.data_catalog import DataCatalogEngine
        
        engine = DataCatalogEngine()
        dataset_id = engine.register_dataset(
            name='transactions',
            df=self.transaction_data
        )
        
        entry = engine.get_dataset(dataset_id)
        
        self.assertIsNotNone(entry)
        self.assertIsNotNone(entry.schema)
        self.assertGreater(len(entry.schema.columns), 0)
    
    def test_search_by_tags(self):
        """Test search by tags."""
        from app.ml.data_catalog import DataCatalogEngine
        
        engine = DataCatalogEngine()
        engine.register_dataset('customers', self.customer_data, tags=['customer', 'crm'])
        engine.register_dataset('transactions', self.transaction_data, tags=['sales', 'transactions'])
        
        results = engine.search(tags=['customer'])
        
        self.assertGreater(len(results), 0)
        self.assertTrue(any('customer' in r.tags for r in results))
    
    def test_search_by_query(self):
        """Test full-text search."""
        from app.ml.data_catalog import DataCatalogEngine
        
        engine = DataCatalogEngine()
        engine.register_dataset(
            'customers',
            self.customer_data,
            description='Customer profile database'
        )
        
        results = engine.search(query='customer')
        
        self.assertGreater(len(results), 0)
    
    def test_empty_catalog(self):
        """Test operations on empty catalog."""
        from app.ml.data_catalog import DataCatalogEngine
        
        engine = DataCatalogEngine()
        
        results = engine.search(query='nonexistent')
        self.assertEqual(len(results), 0)
        
        status = engine.get_status()
        self.assertEqual(status.n_datasets, 0)


class TestDataLineage(unittest.TestCase):
    """Tests for data_lineage module."""
    
    @classmethod
    def setUpClass(cls):
        cls.generator = SyntheticDataGenerator(seed=42)
        cls.data = cls.generator.generate_customer_data(n_customers=50)
    
    def test_register_source(self):
        """Test source registration."""
        from app.ml.data_lineage import DataLineageEngine
        
        engine = DataLineageEngine()
        source_id = engine.register_source(
            name='raw_customers',
            columns=['id', 'name', 'email']
        )
        
        self.assertIsNotNone(source_id)
    
    def test_log_transformation(self):
        """Test transformation logging."""
        from app.ml.data_lineage import DataLineageEngine, OperationType
        
        engine = DataLineageEngine()
        engine.register_source('source', ['col_a', 'col_b'])
        
        engine.log_transformation(
            operation=OperationType.TRANSFORM,
            source_columns=['col_a'],
            target_columns=['col_a_transformed'],
            description='Normalize values'
        )
        
        lineage = engine.get_lineage()
        self.assertGreater(lineage.n_edges, 0)
    
    def test_upstream_tracing(self):
        """Test upstream column tracing."""
        from app.ml.data_lineage import DataLineageEngine, OperationType
        
        engine = DataLineageEngine()
        engine.register_source('source', ['col_a', 'col_b'])
        
        engine.log_transformation(
            operation=OperationType.DERIVE,
            source_columns=['col_a'],
            target_columns=['derived_col']
        )
        
        upstream = engine.get_upstream('derived_col')
        self.assertGreater(len(upstream), 0)
    
    def test_impact_analysis(self):
        """Test impact analysis."""
        from app.ml.data_lineage import DataLineageEngine, OperationType
        
        engine = DataLineageEngine()
        engine.register_source('source', ['col_a'])
        
        engine.log_transformation(
            operation=OperationType.DERIVE,
            source_columns=['col_a'],
            target_columns=['col_b']
        )
        
        engine.log_transformation(
            operation=OperationType.DERIVE,
            source_columns=['col_b'],
            target_columns=['col_c']
        )
        
        impact = engine.impact_analysis('col_a')
        
        self.assertIsNotNone(impact)
        self.assertIn('risk_level', impact)


class TestAlertManager(unittest.TestCase):
    """Tests for alert_manager module."""
    
    @classmethod
    def setUpClass(cls):
        cls.generator = SyntheticDataGenerator(seed=42)
        cls.data = cls.generator.generate_customer_data(n_customers=200)
    
    def test_threshold_alert(self):
        """Test threshold-based alerting."""
        from app.ml.alert_manager import AlertManagerEngine, AlertSeverity
        
        engine = AlertManagerEngine()
        engine.add_threshold_rule(
            name='High value alert',
            metric='avg_lifetime_value',
            condition='>',
            threshold=1000,
            severity=AlertSeverity.WARNING
        )
        
        result = engine.check_rules(
            metrics={'avg_lifetime_value': 1500}
        )
        
        self.assertGreater(result.n_active_alerts, 0)
    
    def test_missing_data_alert(self):
        """Test missing data alerts."""
        from app.ml.alert_manager import AlertManagerEngine
        
        engine = AlertManagerEngine()
        engine.add_missing_data_rule(
            name='Email completeness',
            column='email',
            max_missing_pct=5.0
        )
        
        result = engine.check_rules(df=self.data)
        
        # Data has missing emails from generator
        self.assertIsNotNone(result)
    
    def test_anomaly_alert(self):
        """Test anomaly detection alert."""
        from app.ml.alert_manager import AlertManagerEngine
        
        # Create data with anomalies
        data = pd.DataFrame({
            'value': [10, 11, 12, 10, 100, 11, 10, 9, 12]  # 100 is anomaly
        })
        
        engine = AlertManagerEngine()
        engine.add_anomaly_rule(
            name='Value anomaly',
            column='value',
            n_std=2.0
        )
        
        result = engine.check_rules(df=data)
        
        self.assertIsNotNone(result)
    
    def test_alert_lifecycle(self):
        """Test alert acknowledge and resolve."""
        from app.ml.alert_manager import AlertManagerEngine, AlertStatus
        
        engine = AlertManagerEngine()
        engine.add_threshold_rule(
            name='Test alert',
            metric='value',
            condition='>',
            threshold=50
        )
        
        # Trigger alert
        result = engine.check_rules(metrics={'value': 100})
        
        if result.alerts:
            alert_id = result.alerts[0].alert_id
            
            # Acknowledge
            engine.acknowledge(alert_id)
            
            alert = next((a for a in engine.alerts if a.alert_id == alert_id), None)
            if alert:
                self.assertEqual(alert.status, AlertStatus.ACKNOWLEDGED)
            
            # Resolve
            engine.resolve(alert_id)
            alert = next((a for a in engine.alerts if a.alert_id == alert_id), None)
            if alert:
                self.assertEqual(alert.status, AlertStatus.RESOLVED)


class TestExperimentTracker(unittest.TestCase):
    """Tests for experiment_tracker module."""
    
    def test_create_experiment(self):
        """Test experiment creation."""
        from app.ml.experiment_tracker import ExperimentTrackerEngine
        
        engine = ExperimentTrackerEngine()
        exp_name = engine.create_experiment('test_experiment', 'Test description')
        
        self.assertEqual(exp_name, 'test_experiment')
    
    def test_log_run(self):
        """Test run logging."""
        from app.ml.experiment_tracker import ExperimentTrackerEngine
        
        engine = ExperimentTrackerEngine()
        run = engine.start_run('my_experiment')
        
        engine.log_param('learning_rate', 0.01)
        engine.log_param('epochs', 100)
        engine.log_metric('accuracy', 0.85)
        engine.log_metric('loss', 0.15)
        
        engine.end_run()
        
        self.assertIsNotNone(run)
        self.assertIn('learning_rate', run.parameters)
        self.assertIn('accuracy', run.metrics)
    
    def test_get_best_run(self):
        """Test finding best run."""
        from app.ml.experiment_tracker import ExperimentTrackerEngine
        
        engine = ExperimentTrackerEngine()
        
        # Create multiple runs
        for acc in [0.7, 0.8, 0.85, 0.75]:
            run = engine.start_run('my_experiment')
            engine.log_metric('accuracy', acc)
            engine.end_run()
        
        best = engine.get_best_run('my_experiment', 'accuracy', maximize=True)
        
        self.assertIsNotNone(best)
        self.assertEqual(best.metrics['accuracy'][-1], 0.85)
    
    def test_compare_runs(self):
        """Test run comparison."""
        from app.ml.experiment_tracker import ExperimentTrackerEngine
        
        engine = ExperimentTrackerEngine()
        
        run1 = engine.start_run('experiment', run_name='run_1')
        engine.log_metric('accuracy', 0.8)
        engine.end_run()
        
        run2 = engine.start_run('experiment', run_name='run_2')
        engine.log_metric('accuracy', 0.9)
        engine.end_run()
        
        comparison = engine.compare_runs('experiment', ['run_1', 'run_2'])
        
        self.assertEqual(len(comparison), 2)


class TestReportGenerator(unittest.TestCase):
    """Tests for report_generator module."""
    
    @classmethod
    def setUpClass(cls):
        cls.generator = SyntheticDataGenerator(seed=42)
        cls.data = cls.generator.generate_customer_data(n_customers=200)
    
    def test_executive_summary(self):
        """Test executive summary generation."""
        from app.ml.report_generator import ReportGeneratorEngine, ReportType
        
        engine = ReportGeneratorEngine()
        result = engine.generate(
            self.data,
            report_type=ReportType.EXECUTIVE_SUMMARY
        )
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.markdown)
        self.assertGreater(len(result.markdown), 100)
    
    def test_data_profile_report(self):
        """Test data profile report."""
        from app.ml.report_generator import ReportGeneratorEngine, ReportType
        
        engine = ReportGeneratorEngine()
        result = engine.generate(
            self.data,
            report_type=ReportType.DATA_PROFILE
        )
        
        self.assertIsNotNone(result)
        self.assertGreater(len(result.sections), 0)
    
    def test_html_output(self):
        """Test HTML output generation."""
        from app.ml.report_generator import ReportGeneratorEngine
        
        engine = ReportGeneratorEngine()
        result = engine.generate(self.data)
        
        self.assertIsNotNone(result.html)
        self.assertIn('<html>', result.html)
    
    def test_empty_dataframe_report(self):
        """Test report handles empty DataFrame."""
        from app.ml.report_generator import ReportGeneratorEngine
        
        engine = ReportGeneratorEngine()
        empty_df = pd.DataFrame()
        
        result = engine.generate(empty_df)
        
        self.assertIsNotNone(result)


class TestBenchmarking(unittest.TestCase):
    """Tests for benchmarking module."""
    
    def test_basic_benchmark(self):
        """Test basic function benchmarking."""
        from app.ml.benchmarking import BenchmarkingEngine
        
        engine = BenchmarkingEngine()
        
        def simple_func():
            return sum(range(1000))
        
        result = engine.benchmark('simple_sum', simple_func, iterations=5)
        
        self.assertIsNotNone(result)
        self.assertGreater(result.avg_time_sec, 0)
    
    def test_benchmark_comparison(self):
        """Test benchmark comparison."""
        from app.ml.benchmarking import BenchmarkingEngine
        
        engine = BenchmarkingEngine()
        
        def list_sum():
            return sum(range(10000))
        
        def numpy_sum():
            return np.sum(np.arange(10000))
        
        engine.benchmark('list_sum', list_sum, iterations=5)
        engine.benchmark('numpy_sum', numpy_sum, iterations=5)
        
        comparison = engine.compare('list_sum', 'numpy_sum')
        
        self.assertIsNotNone(comparison)
        self.assertIsNotNone(comparison.speedup)


class TestCustomAggregations(unittest.TestCase):
    """Tests for custom_aggregations module."""
    
    @classmethod
    def setUpClass(cls):
        cls.generator = SyntheticDataGenerator(seed=42)
        cls.transaction_data = cls.generator.generate_transaction_data(n_transactions=1000)
    
    def test_basic_aggregation(self):
        """Test basic aggregation."""
        from app.ml.custom_aggregations import CustomAggregationsEngine, AggregationSpec, AggFunction
        
        engine = CustomAggregationsEngine()
        
        specs = [
            AggregationSpec('amount', AggFunction.SUM, 'total_amount'),
            AggregationSpec('amount', AggFunction.MEAN, 'avg_amount'),
            AggregationSpec('transaction_id', AggFunction.COUNT, 'count')
        ]
        
        result = engine.aggregate(
            self.transaction_data,
            group_by='customer_id',
            aggregations=specs
        )
        
        self.assertIsNotNone(result.aggregated_df)
        self.assertGreater(len(result.aggregated_df), 0)
    
    def test_window_aggregation(self):
        """Test window/rolling aggregation."""
        from app.ml.custom_aggregations import CustomAggregationsEngine
        
        engine = CustomAggregationsEngine()
        
        # Sort by date first
        data = self.transaction_data.sort_values('transaction_date').dropna(subset=['amount'])
        
        result = engine.window_aggregate(
            data.head(100),
            value_col='amount',
            window_size=3,
            functions=['mean', 'sum']
        )
        
        self.assertIsNotNone(result)
        self.assertIn('amount_rolling_mean_3', result.columns)
    
    def test_cumulative_aggregation(self):
        """Test cumulative aggregation."""
        from app.ml.custom_aggregations import CustomAggregationsEngine
        
        engine = CustomAggregationsEngine()
        
        data = self.transaction_data.head(100).dropna(subset=['amount'])
        
        result = engine.cumulative_aggregate(
            data,
            value_col='amount'
        )
        
        self.assertIsNotNone(result)
        self.assertIn('amount_cumsum', result.columns)


class TestEDAAutomation(unittest.TestCase):
    """Tests for eda_automation module."""
    
    @classmethod
    def setUpClass(cls):
        cls.generator = SyntheticDataGenerator(seed=42)
        cls.data = cls.generator.generate_customer_data(n_customers=200)
    
    def test_auto_eda(self):
        """Test automated EDA."""
        from app.ml.eda_automation import EDAAutomationEngine
        
        engine = EDAAutomationEngine()
        result = engine.analyze(self.data)
        
        self.assertIsNotNone(result)
        self.assertGreater(len(result.insights), 0)
    
    def test_eda_recommendations(self):
        """Test EDA recommendations."""
        from app.ml.eda_automation import EDAAutomationEngine
        
        engine = EDAAutomationEngine()
        result = engine.analyze(self.data)
        
        self.assertIsNotNone(result.recommendations)


if __name__ == '__main__':
    unittest.main(verbosity=2)
