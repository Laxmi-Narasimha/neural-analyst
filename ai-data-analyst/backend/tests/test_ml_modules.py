# AI Enterprise Data Analyst - Integration Tests: ML/AutoML Modules
# Comprehensive tests for machine learning, AutoML, and forecasting modules

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


class TestAutoMLEngine(unittest.TestCase):
    """Tests for auto_ml_engine module."""
    
    @classmethod
    def setUpClass(cls):
        cls.generator = SyntheticDataGenerator(seed=42)
        cls.X, cls.y = cls.generator.generate_ml_dataset(
            n_samples=500, n_features=10, task='classification'
        )
        cls.X_reg, cls.y_reg = cls.generator.generate_ml_dataset(
            n_samples=500, n_features=10, task='regression'
        )
    
    def test_classification_task(self):
        """Test AutoML for classification."""
        from app.ml.auto_ml_engine import AutoMLEngine
        
        # Clean data
        X_clean = self.X.select_dtypes(include=[np.number]).dropna(axis=1)
        valid_idx = X_clean.dropna().index
        X_clean = X_clean.loc[valid_idx]
        y_clean = self.y.loc[valid_idx]
        
        if len(X_clean) > 50:
            engine = AutoMLEngine()
            result = engine.train(X_clean, y_clean, task='classification')
            
            self.assertIsNotNone(result)
            self.assertIsNotNone(result.best_model)
    
    def test_regression_task(self):
        """Test AutoML for regression."""
        from app.ml.auto_ml_engine import AutoMLEngine
        
        # Clean data
        X_clean = self.X_reg.select_dtypes(include=[np.number]).dropna(axis=1)
        valid_idx = X_clean.dropna().index
        X_clean = X_clean.loc[valid_idx]
        y_clean = self.y_reg.loc[valid_idx]
        
        if len(X_clean) > 50:
            engine = AutoMLEngine()
            result = engine.train(X_clean, y_clean, task='regression')
            
            self.assertIsNotNone(result)
    
    def test_automl_with_imbalanced_data(self):
        """Test AutoML handles imbalanced classification."""
        from app.ml.auto_ml_engine import AutoMLEngine
        
        # Data from generator is imbalanced (90/10 split)
        X_clean = self.X.select_dtypes(include=[np.number]).dropna(axis=1)
        valid_idx = X_clean.dropna().index
        X_clean = X_clean.loc[valid_idx]
        y_clean = self.y.loc[valid_idx]
        
        if len(X_clean) > 50:
            engine = AutoMLEngine()
            result = engine.train(X_clean, y_clean, task='classification')
            
            self.assertIsNotNone(result)


class TestModelComparison(unittest.TestCase):
    """Tests for model_comparison module."""
    
    @classmethod
    def setUpClass(cls):
        cls.generator = SyntheticDataGenerator(seed=42)
        cls.X, cls.y = cls.generator.generate_ml_dataset(
            n_samples=300, n_features=5, task='classification'
        )
        
        # Clean data
        cls.X_clean = cls.X.select_dtypes(include=[np.number]).dropna(axis=1)
        valid_idx = cls.X_clean.dropna().index
        cls.X_clean = cls.X_clean.loc[valid_idx]
        cls.y_clean = cls.y.loc[valid_idx]
    
    def test_model_comparison(self):
        """Test model comparison."""
        from app.ml.model_comparison import ModelComparisonEngine
        
        if len(self.X_clean) < 50:
            self.skipTest("Not enough clean data")
        
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.tree import DecisionTreeClassifier
            
            models = {
                'logistic': LogisticRegression(random_state=42, max_iter=1000),
                'tree': DecisionTreeClassifier(random_state=42)
            }
            
            engine = ModelComparisonEngine()
            result = engine.compare(
                models,
                self.X_clean.head(200),
                self.y_clean.head(200)
            )
            
            self.assertIsNotNone(result)
            self.assertGreater(len(result.rankings), 0)
        except ImportError:
            self.skipTest("sklearn not available")


class TestClusteringQuality(unittest.TestCase):
    """Tests for clustering_quality module."""
    
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        
        # Generate clusterable data
        cls.cluster_data = pd.DataFrame({
            'x': np.concatenate([
                np.random.normal(0, 1, 100),
                np.random.normal(5, 1, 100),
                np.random.normal(10, 1, 100)
            ]),
            'y': np.concatenate([
                np.random.normal(0, 1, 100),
                np.random.normal(5, 1, 100),
                np.random.normal(0, 1, 100)
            ])
        })
        
        cls.labels = np.array([0] * 100 + [1] * 100 + [2] * 100)
    
    def test_silhouette_score(self):
        """Test silhouette score calculation."""
        from app.ml.clustering_quality import ClusteringQualityEngine
        
        engine = ClusteringQualityEngine()
        result = engine.assess(self.cluster_data, self.labels)
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.silhouette_score)
        
        # Well-separated clusters should have high silhouette
        self.assertGreater(result.silhouette_score, 0.3)
    
    def test_quality_rating(self):
        """Test quality rating assignment."""
        from app.ml.clustering_quality import ClusteringQualityEngine
        
        engine = ClusteringQualityEngine()
        result = engine.assess(self.cluster_data, self.labels)
        
        self.assertIn(result.quality_rating, ['excellent', 'good', 'fair', 'poor'])
    
    def test_cluster_balance(self):
        """Test cluster balance calculation."""
        from app.ml.clustering_quality import ClusteringQualityEngine
        
        engine = ClusteringQualityEngine()
        result = engine.assess(self.cluster_data, self.labels)
        
        # Perfectly balanced clusters
        self.assertEqual(result.size_balance, 1.0)


class TestAdvancedForecasting(unittest.TestCase):
    """Tests for advanced_forecasting module."""
    
    @classmethod
    def setUpClass(cls):
        cls.generator = SyntheticDataGenerator(seed=42)
        cls.ts_data = cls.generator.generate_time_series_data(n_points=365)
    
    def test_basic_forecast(self):
        """Test basic forecasting."""
        from app.ml.advanced_forecasting import AdvancedForecastingEngine
        
        # Clean data
        clean_data = self.ts_data.dropna(subset=['date', 'value'])
        clean_data = clean_data[clean_data['value'].apply(lambda x: np.isfinite(x) if isinstance(x, (int, float)) else False)]
        
        if len(clean_data) > 30:
            engine = AdvancedForecastingEngine()
            result = engine.forecast(
                clean_data,
                date_col='date',
                value_col='value',
                horizon=30
            )
            
            self.assertIsNotNone(result)
            self.assertGreater(len(result.predictions), 0)
    
    def test_forecast_with_gaps(self):
        """Test forecasting handles date gaps."""
        from app.ml.advanced_forecasting import AdvancedForecastingEngine
        
        # Data has gaps from generator
        engine = AdvancedForecastingEngine()
        
        # Clean data
        clean_data = self.ts_data.dropna(subset=['date', 'value'])
        clean_data = clean_data[clean_data['value'].apply(lambda x: np.isfinite(x) if isinstance(x, (int, float)) else False)]
        
        if len(clean_data) > 30:
            result = engine.forecast(
                clean_data,
                date_col='date',
                value_col='value',
                horizon=14
            )
            
            self.assertIsNotNone(result)


class TestAnomalyDetection(unittest.TestCase):
    """Tests for advanced_anomaly_detection module."""
    
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        
        # Normal data with injected anomalies
        cls.normal_data = np.random.normal(100, 15, 500)
        
        # Inject anomalies
        anomaly_indices = [50, 150, 250, 350, 450]
        for idx in anomaly_indices:
            cls.normal_data[idx] = cls.normal_data[idx] * 3
        
        cls.data = pd.DataFrame({
            'value': cls.normal_data,
            'id': range(500)
        })
    
    def test_anomaly_detection(self):
        """Test anomaly detection."""
        from app.ml.advanced_anomaly_detection import AdvancedAnomalyDetectionEngine
        
        engine = AdvancedAnomalyDetectionEngine()
        result = engine.detect(self.data, columns=['value'])
        
        self.assertIsNotNone(result)
        self.assertGreater(result.n_anomalies, 0)
    
    def test_detection_methods(self):
        """Test different detection methods."""
        from app.ml.advanced_anomaly_detection import AdvancedAnomalyDetectionEngine, DetectionMethod
        
        engine = AdvancedAnomalyDetectionEngine()
        
        for method in [DetectionMethod.IQR, DetectionMethod.ZSCORE]:
            result = engine.detect(
                self.data,
                columns=['value'],
                method=method
            )
            self.assertIsNotNone(result)


class TestSegmentation(unittest.TestCase):
    """Tests for advanced_segmentation module."""
    
    @classmethod
    def setUpClass(cls):
        cls.generator = SyntheticDataGenerator(seed=42)
        cls.customer_data = cls.generator.generate_customer_data(n_customers=300)
    
    def test_customer_segmentation(self):
        """Test customer segmentation."""
        from app.ml.advanced_segmentation import AdvancedSegmentationEngine
        
        # Use numeric columns
        numeric_cols = ['age', 'lifetime_value']
        clean_data = self.customer_data[numeric_cols].dropna()
        
        if len(clean_data) > 50:
            engine = AdvancedSegmentationEngine()
            result = engine.segment(clean_data, n_clusters=3)
            
            self.assertIsNotNone(result)
            self.assertIsNotNone(result.labels)
    
    def test_optimal_k_selection(self):
        """Test optimal K selection."""
        from app.ml.advanced_segmentation import AdvancedSegmentationEngine
        
        numeric_cols = ['age', 'lifetime_value']
        clean_data = self.customer_data[numeric_cols].dropna()
        
        if len(clean_data) > 50:
            engine = AdvancedSegmentationEngine()
            result = engine.segment(clean_data, auto_k=True, max_k=5)
            
            self.assertIsNotNone(result)


class TestTextModules(unittest.TestCase):
    """Tests for text processing modules."""
    
    @classmethod
    def setUpClass(cls):
        cls.generator = SyntheticDataGenerator(seed=42)
        cls.text_data = cls.generator.generate_text_data(n_records=200)
    
    def test_keyword_extraction(self):
        """Test keyword extraction."""
        from app.ml.keyword_extraction import KeywordExtractionEngine
        
        engine = KeywordExtractionEngine()
        
        # Get valid texts
        valid_texts = self.text_data['text'].dropna()
        valid_texts = valid_texts[valid_texts.str.len() > 10]
        
        if len(valid_texts) > 10:
            result = engine.extract(valid_texts.tolist())
            self.assertIsNotNone(result)
            self.assertGreater(len(result.keywords), 0)
    
    def test_text_summarization(self):
        """Test text summarization."""
        from app.ml.text_summarization import TextSummarizationEngine
        
        engine = TextSummarizationEngine()
        
        long_text = """
        This is a long text that needs to be summarized. 
        It contains multiple sentences about various topics.
        The summarization algorithm should pick the most important sentences.
        Testing the robustness of the summarization engine.
        We need to ensure it handles different types of content properly.
        """
        
        result = engine.summarize(long_text, ratio=0.4)
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.summary)
        self.assertLess(len(result.summary), len(long_text))
    
    def test_summarization_empty_text(self):
        """Test summarization handles empty text."""
        from app.ml.text_summarization import TextSummarizationEngine
        
        engine = TextSummarizationEngine()
        
        result = engine.summarize("")
        self.assertIsNotNone(result)
    
    def test_sentiment_analysis(self):
        """Test sentiment analysis."""
        from app.ml.sentiment_analysis import SentimentAnalysisEngine
        
        engine = SentimentAnalysisEngine()
        
        texts = [
            "This product is amazing! I love it!",
            "Terrible experience, would not recommend.",
            "It's okay, nothing special."
        ]
        
        for text in texts:
            result = engine.analyze(text)
            self.assertIsNotNone(result)
            self.assertIn(result.sentiment.lower(), ['positive', 'negative', 'neutral'])


class TestFeatureEngineering(unittest.TestCase):
    """Tests for feature engineering modules."""
    
    @classmethod
    def setUpClass(cls):
        cls.generator = SyntheticDataGenerator(seed=42)
        cls.customer_data = cls.generator.generate_customer_data(n_customers=200)
        cls.transaction_data = cls.generator.generate_transaction_data(n_transactions=1000)
    
    def test_data_enrichment(self):
        """Test data enrichment."""
        from app.ml.data_enrichment import DataEnrichmentEngine
        
        engine = DataEnrichmentEngine()
        result = engine.enrich(self.customer_data, auto=True)
        
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.n_new_columns, 0)
    
    def test_date_feature_extraction(self):
        """Test date feature extraction."""
        from app.ml.data_enrichment import DataEnrichmentEngine
        
        engine = DataEnrichmentEngine()
        engine.add_date_features('registration_date')
        
        result = engine.enrich(self.customer_data, auto=False)
        
        self.assertIsNotNone(result)


class TestDataTransformation(unittest.TestCase):
    """Tests for data_transformation module."""
    
    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.numeric_data = pd.DataFrame({
            'value': np.random.exponential(10, 200),
            'normal': np.random.normal(100, 15, 200)
        })
    
    def test_log_transform(self):
        """Test log transformation."""
        from app.ml.data_transformation import DataTransformationEngine, TransformationType
        
        engine = DataTransformationEngine()
        result = engine.transform(
            self.numeric_data,
            column='value',
            transform_type=TransformationType.LOG
        )
        
        self.assertIsNotNone(result.transformed_df)
    
    def test_standardize(self):
        """Test standardization."""
        from app.ml.data_transformation import DataTransformationEngine, TransformationType
        
        engine = DataTransformationEngine()
        result = engine.transform(
            self.numeric_data,
            column='normal',
            transform_type=TransformationType.STANDARDIZE
        )
        
        self.assertIsNotNone(result.transformed_df)
        
        # Standardized should have mean ~0, std ~1
        transformed = result.transformed_df['normal_standardized']
        self.assertAlmostEqual(transformed.mean(), 0, places=1)
        self.assertAlmostEqual(transformed.std(), 1, places=1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
