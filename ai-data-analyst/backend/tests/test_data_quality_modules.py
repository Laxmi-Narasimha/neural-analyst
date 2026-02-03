# AI Enterprise Data Analyst - Integration Tests: Data Quality Modules
# Comprehensive tests for data quality, validation, and cleaning modules

import sys
import os
import unittest
from datetime import datetime
from typing import Dict, Any

import numpy as np
import pandas as pd

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tests.synthetic_data_generator import SyntheticDataGenerator


class TestDataValidation(unittest.TestCase):
    """Tests for data_validation module."""
    
    @classmethod
    def setUpClass(cls):
        cls.generator = SyntheticDataGenerator(seed=42)
        cls.customer_data = cls.generator.generate_customer_data(n_customers=500)
        cls.transaction_data = cls.generator.generate_transaction_data(n_transactions=2000)
    
    def test_validation_with_nulls(self):
        """Test validation handles null values correctly."""
        from app.ml.data_validation import DataValidationEngine
        
        engine = DataValidationEngine()
        # Add null check rules for columns we know have nulls
        engine.require_not_null('email')
        engine.require_not_null('phone')
        
        result = engine.validate(self.customer_data)
        
        self.assertIsNotNone(result)
        self.assertGreater(len(result.issues), 0)  # Should detect null issues
        self.assertLessEqual(result.quality_score, 100)
    
    def test_validation_with_empty_df(self):
        """Test validation handles empty DataFrame."""
        from app.ml.data_validation import DataValidationEngine
        
        engine = DataValidationEngine()
        empty_df = self.generator.generate_empty_dataframe()
        
        # Should not crash
        try:
            result = engine.validate(empty_df)
            self.assertIsNotNone(result)
        except Exception as e:
            self.fail(f"Validation crashed on empty DataFrame: {e}")
    
    def test_validation_with_single_row(self):
        """Test validation handles single row DataFrame."""
        from app.ml.data_validation import DataValidationEngine
        
        engine = DataValidationEngine()
        single_row = self.generator.generate_single_row_dataframe()
        
        result = engine.validate(single_row)
        self.assertIsNotNone(result)
    
    def test_regex_patterns(self):
        """Test regex pattern validation."""
        from app.ml.data_validation import DataValidationEngine
        
        engine = DataValidationEngine()
        
        # Add email validation rule using require_pattern
        engine.require_pattern('email', r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        
        result = engine.validate(self.customer_data)
        
        # Should detect invalid emails (those that don't match the pattern or are null)
        email_issues = [i for i in result.issues if i.column == 'email']
        self.assertGreater(len(email_issues), 0)


class TestDataMasking(unittest.TestCase):
    """Tests for data_masking module."""
    
    @classmethod
    def setUpClass(cls):
        cls.generator = SyntheticDataGenerator(seed=42)
        cls.customer_data = cls.generator.generate_customer_data(n_customers=200)
    
    def test_pii_detection(self):
        """Test PII detection finds emails and phones."""
        from app.ml.data_masking import DataMaskingEngine
        
        engine = DataMaskingEngine()
        detections = engine.detect_pii(self.customer_data)
        
        self.assertIsNotNone(detections)
        
        # Should detect email column
        detected_types = [d.pii_type.value for d in detections]
        self.assertIn('email', detected_types)
    
    def test_partial_masking(self):
        """Test partial masking strategy."""
        from app.ml.data_masking import DataMaskingEngine, MaskingStrategy
        
        engine = DataMaskingEngine()
        result = engine.mask(self.customer_data, strategy=MaskingStrategy.PARTIAL)
        
        self.assertIsNotNone(result.masked_df)
        self.assertGreater(result.n_values_masked, 0)
    
    def test_hash_masking(self):
        """Test hash masking strategy."""
        from app.ml.data_masking import DataMaskingEngine, MaskingStrategy
        
        engine = DataMaskingEngine()
        result = engine.mask(
            self.customer_data, 
            columns=['email'],
            strategy=MaskingStrategy.HASH
        )
        
        # Masked values should be different from original
        if 'email' in result.masked_df.columns:
            original_emails = self.customer_data['email'].dropna()
            masked_emails = result.masked_df['email'].dropna()
            
            if len(original_emails) > 0 and len(masked_emails) > 0:
                self.assertNotEqual(
                    original_emails.iloc[0], 
                    masked_emails.iloc[0]
                )
    
    def test_masking_all_null_column(self):
        """Test masking handles all-null columns."""
        from app.ml.data_masking import DataMaskingEngine, MaskingStrategy
        
        df = pd.DataFrame({'email': [None] * 100})
        
        engine = DataMaskingEngine()
        result = engine.mask(df, strategy=MaskingStrategy.REDACT)
        
        self.assertIsNotNone(result.masked_df)


class TestDataQualityReport(unittest.TestCase):
    """Tests for data_quality_report module."""
    
    @classmethod
    def setUpClass(cls):
        cls.generator = SyntheticDataGenerator(seed=42)
        cls.customer_data = cls.generator.generate_customer_data(n_customers=500)
        cls.all_null_data = cls.generator.generate_all_null_dataframe(100)
    
    def test_quality_assessment(self):
        """Test quality assessment produces valid scores."""
        from app.ml.data_quality_report import DataQualityReportEngine
        
        engine = DataQualityReportEngine()
        result = engine.assess(self.customer_data)
        
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.overall_score, 0)
        self.assertLessEqual(result.overall_score, 100)
        self.assertIn(result.grade, ['A', 'B', 'C', 'D', 'F'])
    
    def test_quality_with_all_nulls(self):
        """Test quality assessment with all-null data."""
        from app.ml.data_quality_report import DataQualityReportEngine
        
        engine = DataQualityReportEngine()
        result = engine.assess(self.all_null_data)
        
        # Should produce low score but not crash
        self.assertIsNotNone(result)
        self.assertLess(result.completeness_score, 50)
    
    def test_quality_dimensions(self):
        """Test all quality dimensions are calculated."""
        from app.ml.data_quality_report import DataQualityReportEngine
        
        engine = DataQualityReportEngine()
        result = engine.assess(self.customer_data)
        
        # All dimensions should have scores
        self.assertGreaterEqual(result.completeness_score, 0)
        self.assertGreaterEqual(result.uniqueness_score, 0)
        self.assertGreaterEqual(result.validity_score, 0)
        self.assertGreaterEqual(result.consistency_score, 0)
    
    def test_issue_detection(self):
        """Test issues are detected and categorized."""
        from app.ml.data_quality_report import DataQualityReportEngine
        
        engine = DataQualityReportEngine()
        result = engine.assess(self.customer_data)
        
        # Should detect some issues in edge-case-rich data
        total_issues = (
            result.critical_issues + 
            result.high_issues + 
            result.medium_issues + 
            result.low_issues
        )
        self.assertGreater(total_issues, 0)


class TestOutlierTreatment(unittest.TestCase):
    """Tests for outlier_treatment module."""
    
    @classmethod
    def setUpClass(cls):
        cls.generator = SyntheticDataGenerator(seed=42)
        cls.transaction_data = cls.generator.generate_transaction_data(n_transactions=1000)
        
        # Create data with known outliers
        cls.outlier_data = pd.DataFrame({
            'value': [10, 12, 11, 13, 9, 100, 10, 11, -50, 12]  # 100 and -50 are outliers
        })
    
    def test_iqr_detection(self):
        """Test IQR outlier detection."""
        from app.ml.outlier_treatment import OutlierTreatmentEngine, OutlierMethod
        
        engine = OutlierTreatmentEngine()
        result = engine.detect_only(self.outlier_data, columns=['value'], method=OutlierMethod.IQR)
        
        self.assertIsNotNone(result)
        self.assertGreater(result.total_outliers, 0)
    
    def test_zscore_detection(self):
        """Test Z-score outlier detection."""
        from app.ml.outlier_treatment import OutlierTreatmentEngine, OutlierMethod
        
        engine = OutlierTreatmentEngine()
        result = engine.detect_only(self.outlier_data, columns=['value'], method=OutlierMethod.ZSCORE)
        
        self.assertIsNotNone(result)
    
    def test_treatment_cap(self):
        """Test capping treatment."""
        from app.ml.outlier_treatment import OutlierTreatmentEngine, TreatmentStrategy
        
        engine = OutlierTreatmentEngine()
        result = engine.detect_and_treat(
            self.outlier_data, 
            columns=['value'], 
            strategy=TreatmentStrategy.CAP
        )
        
        self.assertIsNotNone(result.treated_df)
        
        # Extreme values should be capped
        treated_values = result.treated_df['value']
        self.assertLess(treated_values.max(), 100)
    
    def test_outlier_detection_with_nulls(self):
        """Test outlier detection handles nulls."""
        from app.ml.outlier_treatment import OutlierTreatmentEngine
        
        df = pd.DataFrame({
            'value': [10, None, 12, np.nan, 100, 11, None]
        })
        
        engine = OutlierTreatmentEngine()
        result = engine.detect_only(df, columns=['value'])
        
        self.assertIsNotNone(result)


class TestDataSampling(unittest.TestCase):
    """Tests for data_sampling module."""
    
    @classmethod
    def setUpClass(cls):
        cls.generator = SyntheticDataGenerator(seed=42)
        cls.customer_data = cls.generator.generate_customer_data(n_customers=1000)
    
    def test_random_sampling(self):
        """Test random sampling."""
        from app.ml.data_sampling import DataSamplingEngine, SamplingMethod
        
        engine = DataSamplingEngine()
        result = engine.sample(
            self.customer_data,
            method=SamplingMethod.RANDOM,
            n=100
        )
        
        self.assertIsNotNone(result.sampled_df)
        self.assertEqual(len(result.sampled_df), 100)
    
    def test_stratified_sampling(self):
        """Test stratified sampling."""
        from app.ml.data_sampling import DataSamplingEngine, SamplingMethod
        
        engine = DataSamplingEngine()
        result = engine.sample(
            self.customer_data,
            method=SamplingMethod.STRATIFIED,
            stratify_col='segment',
            n=200
        )
        
        self.assertIsNotNone(result.sampled_df)
    
    def test_sampling_larger_than_data(self):
        """Test sampling when requested size > data size."""
        from app.ml.data_sampling import DataSamplingEngine, SamplingMethod
        
        small_df = self.customer_data.head(50)
        
        engine = DataSamplingEngine()
        result = engine.sample(
            small_df,
            method=SamplingMethod.RANDOM,
            n=100,  # Larger than data
            replace=True  # Allow replacement for larger sample
        )
        
        # Should handle gracefully
        self.assertIsNotNone(result.sampled_df)


class TestSchemaInference(unittest.TestCase):
    """Tests for schema_inference module."""
    
    @classmethod
    def setUpClass(cls):
        cls.generator = SyntheticDataGenerator(seed=42)
        cls.customer_data = cls.generator.generate_customer_data(n_customers=200)
        cls.transaction_data = cls.generator.generate_transaction_data(n_transactions=500)
    
    def test_basic_inference(self):
        """Test basic schema inference."""
        from app.ml.schema_inference import SchemaInferenceEngine
        
        engine = SchemaInferenceEngine()
        result = engine.infer(self.customer_data)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.n_columns, len(self.customer_data.columns))
        self.assertGreater(result.type_coverage, 0)
    
    def test_semantic_type_detection(self):
        """Test semantic type detection."""
        from app.ml.schema_inference import SchemaInferenceEngine, SemanticType
        
        engine = SchemaInferenceEngine()
        result = engine.infer(self.customer_data)
        
        # Should detect email column
        email_col = next((c for c in result.columns if c.column_name == 'email'), None)
        if email_col:
            self.assertEqual(email_col.semantic_type, SemanticType.EMAIL)
    
    def test_inference_empty_df(self):
        """Test inference handles empty DataFrame."""
        from app.ml.schema_inference import SchemaInferenceEngine
        
        engine = SchemaInferenceEngine()
        empty_df = pd.DataFrame()
        
        result = engine.infer(empty_df)
        self.assertIsNotNone(result)
        self.assertEqual(result.n_columns, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
