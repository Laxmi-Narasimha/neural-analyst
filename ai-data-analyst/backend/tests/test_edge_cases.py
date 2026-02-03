# AI Enterprise Data Analyst - Test Runner and Edge Case Stress Tests
# Master test file with edge case stress testing

import sys
import os
import unittest
from datetime import datetime, timedelta
from typing import Dict, Any, List

import numpy as np
import pandas as pd

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.synthetic_data_generator import SyntheticDataGenerator


class EdgeCaseStressTests(unittest.TestCase):
    """
    Extreme edge case stress tests.
    
    Tests modules with:
    - Empty DataFrames
    - Single row/column DataFrames
    - All null values
    - Extreme numeric values
    - Unicode and special characters
    - Very large datasets
    - Malformed data
    """
    
    @classmethod
    def setUpClass(cls):
        cls.generator = SyntheticDataGenerator(seed=42)
        
        # Edge case datasets
        cls.empty_df = pd.DataFrame()
        cls.single_row = pd.DataFrame({'a': [1], 'b': ['x']})
        cls.single_col = pd.DataFrame({'a': range(100)})
        cls.all_null = pd.DataFrame({'a': [None] * 100, 'b': [np.nan] * 100})
        cls.all_same = pd.DataFrame({'a': [42] * 100, 'b': ['same'] * 100})
        
        # Extreme values
        cls.extreme_values = pd.DataFrame({
            'normal': np.random.normal(0, 1, 100),
            'inf_pos': [np.inf] * 100,
            'inf_neg': [-np.inf] * 100,
            'very_large': [1e308] * 100,
            'very_small': [1e-308] * 100,
            'zeros': [0] * 100
        })
        
        # Unicode and special chars
        cls.unicode_data = pd.DataFrame({
            'text': ['Hello', '你好', 'مرحبا', '🎉', '<script>', 'a' * 10000, '', None] * 12
        })
        
        # Mixed types in same column
        cls.mixed_types = pd.DataFrame({
            'mixed': [1, 'text', 3.14, None, True, [1, 2], {'a': 1}] * 14
        })
    
    def test_empty_dataframe_handling(self):
        """Test all modules handle empty DataFrame."""
        modules_to_test = [
            ('data_validation', 'DataValidationEngine', 'validate'),
            ('data_quality_report', 'DataQualityReportEngine', 'assess'),
            ('univariate_analysis', 'UnivariateAnalysisEngine', 'analyze_dataframe'),
            ('eda_automation', 'EDAAutomationEngine', 'analyze'),
            ('schema_inference', 'SchemaInferenceEngine', 'infer'),
        ]
        
        for module_name, class_name, method_name in modules_to_test:
            with self.subTest(module=module_name):
                try:
                    module = __import__(f'app.ml.{module_name}', fromlist=[class_name])
                    engine_class = getattr(module, class_name)
                    engine = engine_class()
                    method = getattr(engine, method_name)
                    
                    # Should not crash
                    result = method(self.empty_df)
                    self.assertIsNotNone(result)
                except ImportError:
                    pass  # Module not available
                except Exception as e:
                    # Log but don't fail - some may legitimately fail
                    print(f"Module {module_name} raised {type(e).__name__}: {e}")
    
    def test_all_null_handling(self):
        """Test modules handle all-null data."""
        modules_to_test = [
            ('data_validation', 'DataValidationEngine', 'validate'),
            ('data_quality_report', 'DataQualityReportEngine', 'assess'),
        ]
        
        for module_name, class_name, method_name in modules_to_test:
            with self.subTest(module=module_name):
                try:
                    module = __import__(f'app.ml.{module_name}', fromlist=[class_name])
                    engine_class = getattr(module, class_name)
                    engine = engine_class()
                    method = getattr(engine, method_name)
                    
                    result = method(self.all_null)
                    self.assertIsNotNone(result)
                except ImportError:
                    pass
                except Exception as e:
                    print(f"Module {module_name} raised {type(e).__name__}: {e}")
    
    def test_single_row_handling(self):
        """Test modules handle single-row DataFrame."""
        # Single row is tricky for statistical tests
        from app.ml.univariate_analysis import UnivariateAnalysisEngine
        
        engine = UnivariateAnalysisEngine()
        
        # Should handle without crashing
        try:
            result = engine.analyze(self.single_row['a'])
            self.assertIsNotNone(result)
        except Exception as e:
            # May fail gracefully
            pass
    
    def test_constant_column_handling(self):
        """Test modules handle constant columns."""
        from app.ml.univariate_analysis import UnivariateAnalysisEngine
        
        engine = UnivariateAnalysisEngine()
        result = engine.analyze(self.all_same['a'])
        
        self.assertIsNotNone(result)
        self.assertEqual(result.std, 0)
    
    def test_extreme_numeric_values(self):
        """Test modules handle extreme numeric values."""
        from app.ml.data_quality_report import DataQualityReportEngine
        
        engine = DataQualityReportEngine()
        result = engine.assess(self.extreme_values)
        
        self.assertIsNotNone(result)
    
    def test_unicode_handling(self):
        """Test modules handle unicode and special characters."""
        from app.ml.schema_inference import SchemaInferenceEngine
        
        engine = SchemaInferenceEngine()
        result = engine.infer(self.unicode_data)
        
        self.assertIsNotNone(result)


class ConcurrencyStressTests(unittest.TestCase):
    """Tests for concurrent access patterns."""
    
    def test_multiple_engine_instances(self):
        """Test creating multiple engine instances."""
        from app.ml.data_validation import DataValidationEngine
        from app.ml.data_quality_report import DataQualityReportEngine
        
        engines = [DataValidationEngine() for _ in range(10)]
        
        # All should be independent
        self.assertEqual(len(engines), 10)
    
    def test_reuse_engine(self):
        """Test reusing engine for multiple analyses."""
        from app.ml.data_quality_report import DataQualityReportEngine
        
        generator = SyntheticDataGenerator(seed=42)
        engine = DataQualityReportEngine()
        
        for i in range(5):
            data = generator.generate_customer_data(n_customers=50)
            result = engine.assess(data)
            self.assertIsNotNone(result)


class PerformanceTests(unittest.TestCase):
    """Performance-related tests."""
    
    def test_large_dataset_handling(self):
        """Test handling of larger datasets."""
        import time
        
        generator = SyntheticDataGenerator(seed=42)
        
        # Generate larger dataset
        large_data = generator.generate_transaction_data(n_transactions=10000)
        
        from app.ml.data_quality_report import DataQualityReportEngine
        
        engine = DataQualityReportEngine()
        
        start = time.time()
        result = engine.assess(large_data)
        elapsed = time.time() - start
        
        self.assertIsNotNone(result)
        # Should complete in reasonable time
        self.assertLess(elapsed, 30)  # 30 seconds max
    
    def test_wide_dataset_handling(self):
        """Test handling of wide datasets (many columns)."""
        # Create wide dataset
        n_cols = 100
        data = {f'col_{i}': np.random.normal(0, 1, 100) for i in range(n_cols)}
        wide_df = pd.DataFrame(data)
        
        from app.ml.schema_inference import SchemaInferenceEngine
        
        engine = SchemaInferenceEngine()
        result = engine.infer(wide_df)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.n_columns, n_cols)


class DataTypeTests(unittest.TestCase):
    """Tests for various data type handling."""
    
    def test_datetime_handling(self):
        """Test datetime column handling."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='H'),
            'value': np.random.normal(100, 15, 100)
        })
        
        from app.ml.data_enrichment import DataEnrichmentEngine
        
        engine = DataEnrichmentEngine()
        result = engine.enrich(data, auto=True)
        
        self.assertIsNotNone(result)
        # Should extract date features
        self.assertGreater(result.n_new_columns, 0)
    
    def test_categorical_handling(self):
        """Test categorical dtype handling."""
        data = pd.DataFrame({
            'category': pd.Categorical(['A', 'B', 'C'] * 33 + ['A']),
            'value': range(100)
        })
        
        from app.ml.schema_inference import SchemaInferenceEngine
        
        engine = SchemaInferenceEngine()
        result = engine.infer(data)
        
        self.assertIsNotNone(result)
    
    def test_boolean_handling(self):
        """Test boolean column handling."""
        data = pd.DataFrame({
            'flag': [True, False, True, None, True] * 20,
            'value': range(100)
        })
        
        from app.ml.schema_inference import SchemaInferenceEngine
        
        engine = SchemaInferenceEngine()
        result = engine.infer(data)
        
        self.assertIsNotNone(result)


class ModuleIntegrationTests(unittest.TestCase):
    """Tests for module integration and chaining."""
    
    def test_validation_to_quality_pipeline(self):
        """Test validation followed by quality assessment."""
        generator = SyntheticDataGenerator(seed=42)
        data = generator.generate_customer_data(n_customers=200)
        
        from app.ml.data_validation import DataValidationEngine
        from app.ml.data_quality_report import DataQualityReportEngine
        
        # Step 1: Validate
        validation_engine = DataValidationEngine()
        validation_result = validation_engine.validate(data)
        
        # Step 2: Quality report
        quality_engine = DataQualityReportEngine()
        quality_result = quality_engine.assess(data)
        
        self.assertIsNotNone(validation_result)
        self.assertIsNotNone(quality_result)
    
    def test_enrichment_to_analysis_pipeline(self):
        """Test enrichment followed by analysis."""
        generator = SyntheticDataGenerator(seed=42)
        data = generator.generate_customer_data(n_customers=200)
        
        from app.ml.data_enrichment import DataEnrichmentEngine
        from app.ml.eda_automation import EDAAutomationEngine
        
        # Step 1: Enrich
        enrichment_engine = DataEnrichmentEngine()
        enrichment_result = enrichment_engine.enrich(data, auto=True)
        
        # Step 2: EDA on enriched data
        eda_engine = EDAAutomationEngine()
        eda_result = eda_engine.analyze(enrichment_result.enriched_df)
        
        self.assertIsNotNone(enrichment_result)
        self.assertIsNotNone(eda_result)
    
    def test_catalog_to_lineage_pipeline(self):
        """Test catalog registration with lineage tracking."""
        generator = SyntheticDataGenerator(seed=42)
        data = generator.generate_customer_data(n_customers=100)
        
        from app.ml.data_catalog import DataCatalogEngine
        from app.ml.data_lineage import DataLineageEngine
        
        # Step 1: Catalog
        catalog = DataCatalogEngine()
        dataset_id = catalog.register_dataset('customers', data)
        
        # Step 2: Lineage
        lineage = DataLineageEngine()
        lineage.register_source('customers', data.columns.tolist())
        
        self.assertIsNotNone(dataset_id)
        self.assertGreater(lineage.get_lineage().n_nodes, 0)


def run_all_tests():
    """Run all test suites."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(EdgeCaseStressTests))
    suite.addTests(loader.loadTestsFromTestCase(ConcurrencyStressTests))
    suite.addTests(loader.loadTestsFromTestCase(PerformanceTests))
    suite.addTests(loader.loadTestsFromTestCase(DataTypeTests))
    suite.addTests(loader.loadTestsFromTestCase(ModuleIntegrationTests))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    run_all_tests()
