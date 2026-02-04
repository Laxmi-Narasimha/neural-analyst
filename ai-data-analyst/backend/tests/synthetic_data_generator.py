# AI Enterprise Data Analyst - Synthetic Data Generator
# Production-grade synthetic data for testing all modules
# Generates: complex, edge-case-rich datasets

from __future__ import annotations

import random
import string
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class SyntheticDataGenerator:
    """
    Production-grade Synthetic Data Generator for testing.
    
    Generates complex datasets with:
    - Missing values (various patterns)
    - Outliers (extreme values)
    - Edge cases (empty, single value, duplicates)
    - Mixed types and inconsistencies
    - Time series with gaps and anomalies
    - Categorical imbalance
    - Unicode and special characters
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
    
    # =========================================================================
    # Core Data Generators
    # =========================================================================
    
    def generate_customer_data(
        self,
        n_customers: int = 1000,
        include_edge_cases: bool = True
    ) -> pd.DataFrame:
        """
        Generate complex customer dataset with edge cases.
        
        Edge cases:
        - Missing emails, phones
        - Invalid formats
        - Unicode names
        - Duplicate IDs
        - Extreme ages
        - Future/past dates
        """
        data = {
            'customer_id': [],
            'first_name': [],
            'last_name': [],
            'email': [],
            'phone': [],
            'age': [],
            'gender': [],
            'registration_date': [],
            'country': [],
            'city': [],
            'segment': [],
            'lifetime_value': [],
            'is_active': []
        }
        
        names_first = ['John', 'Jane', 'Alice', 'Bob', 'Carlos', 'María', '李明', 'Müller', 'François', 'Александр']
        names_last = ['Smith', 'Johnson', 'García', 'Müller', '王', 'O\'Brien', 'Björk', 'López', 'Иванов', 'Nakamura']
        countries = ['USA', 'UK', 'Germany', 'Japan', 'Brazil', 'India', 'Australia', None, '', 'Unknown']
        cities = ['New York', 'London', 'Berlin', 'Tokyo', None, '', 'N/A', '城市', 'São Paulo']
        genders = ['M', 'F', 'Other', None, '', 'Unknown', 'N/A']
        segments = ['Premium', 'Standard', 'Basic', 'VIP', None, 'Unknown']
        
        for i in range(n_customers):
            # Customer ID with some duplicates
            if include_edge_cases and i > 0 and random.random() < 0.02:
                cust_id = data['customer_id'][random.randint(0, len(data['customer_id'])-1)]
            else:
                cust_id = f"CUST_{i:06d}"
            
            data['customer_id'].append(cust_id)
            
            # Names with Unicode
            data['first_name'].append(random.choice(names_first) if random.random() > 0.05 else None)
            data['last_name'].append(random.choice(names_last) if random.random() > 0.05 else None)
            
            # Email with various edge cases
            if random.random() < 0.85:
                email = f"user{i}@email.com"
            elif random.random() < 0.5:
                email = None
            elif random.random() < 0.5:
                email = "invalid-email"  # Invalid format
            else:
                email = ""
            data['email'].append(email)
            
            # Phone with various formats
            phone_formats = [
                f"+1-555-{random.randint(100,999)}-{random.randint(1000,9999)}",
                f"555{random.randint(1000000,9999999)}",
                f"(555) 123-{random.randint(1000,9999)}",
                None,
                "",
                "invalid"
            ]
            data['phone'].append(random.choice(phone_formats))
            
            # Age with outliers
            if include_edge_cases and random.random() < 0.05:
                age = random.choice([-5, 0, 150, 200, None, np.nan])
            else:
                age = random.randint(18, 80)
            data['age'].append(age)
            
            data['gender'].append(random.choice(genders))
            
            # Registration date with edge cases
            if include_edge_cases and random.random() < 0.05:
                reg_date = random.choice([
                    datetime(1900, 1, 1),  # Very old
                    datetime(2099, 12, 31),  # Future
                    None
                ])
            else:
                days_ago = random.randint(1, 1000)
                reg_date = datetime.now() - timedelta(days=days_ago)
            data['registration_date'].append(reg_date)
            
            data['country'].append(random.choice(countries))
            data['city'].append(random.choice(cities))
            data['segment'].append(random.choice(segments))
            
            # Lifetime value with outliers
            if include_edge_cases and random.random() < 0.05:
                ltv = random.choice([-100, 0, 1000000, None, np.nan])
            else:
                ltv = round(random.uniform(10, 5000), 2)
            data['lifetime_value'].append(ltv)
            
            data['is_active'].append(random.choice([True, False, None, 1, 0, 'Yes', 'No']))
        
        return pd.DataFrame(data)
    
    def generate_transaction_data(
        self,
        n_transactions: int = 5000,
        n_customers: int = 500,
        include_edge_cases: bool = True
    ) -> pd.DataFrame:
        """
        Generate complex transaction dataset.
        
        Edge cases:
        - Negative amounts
        - Zero amounts
        - Extremely large amounts
        - Future dates
        - Missing values
        - Duplicate transaction IDs
        """
        data = {
            'transaction_id': [],
            'customer_id': [],
            'transaction_date': [],
            'amount': [],
            'quantity': [],
            'product_id': [],
            'product_category': [],
            'payment_method': [],
            'status': [],
            'discount_pct': [],
            'tax_amount': [],
            'currency': []
        }
        
        categories = ['Electronics', 'Clothing', 'Food', 'Books', 'Home', None, '', 'Üñíçödé']
        payment_methods = ['Credit Card', 'Debit Card', 'PayPal', 'Cash', 'Crypto', None, '']
        statuses = ['Completed', 'Pending', 'Failed', 'Refunded', 'Cancelled', None]
        currencies = ['USD', 'EUR', 'GBP', 'JPY', None, '', 'INVALID']
        
        customer_ids = [f"CUST_{i:06d}" for i in range(n_customers)]
        
        for i in range(n_transactions):
            # Transaction ID with duplicates
            if include_edge_cases and i > 0 and random.random() < 0.01:
                txn_id = data['transaction_id'][random.randint(0, len(data['transaction_id'])-1)]
            else:
                txn_id = f"TXN_{i:08d}"
            data['transaction_id'].append(txn_id)
            
            data['customer_id'].append(random.choice(customer_ids))
            
            # Date with gaps and anomalies
            if include_edge_cases and random.random() < 0.05:
                txn_date = random.choice([
                    datetime(1990, 1, 1),
                    datetime(2050, 1, 1),
                    None
                ])
            else:
                days_ago = random.randint(0, 365)
                txn_date = datetime.now() - timedelta(days=days_ago, hours=random.randint(0, 23))
            data['transaction_date'].append(txn_date)
            
            # Amount with edge cases
            if include_edge_cases and random.random() < 0.05:
                amount = random.choice([-500, 0, 1000000, np.inf, -np.inf, np.nan, None])
            else:
                amount = round(random.uniform(1, 1000), 2)
            data['amount'].append(amount)
            
            # Quantity with edge cases
            if include_edge_cases and random.random() < 0.05:
                qty = random.choice([-1, 0, 10000, None])
            else:
                qty = random.randint(1, 10)
            data['quantity'].append(qty)
            
            data['product_id'].append(f"PROD_{random.randint(1, 100):04d}")
            data['product_category'].append(random.choice(categories))
            data['payment_method'].append(random.choice(payment_methods))
            data['status'].append(random.choice(statuses))
            
            # Discount with edge cases
            if include_edge_cases and random.random() < 0.05:
                discount = random.choice([-10, 150, None])
            else:
                discount = random.choice([0, 5, 10, 15, 20, 25, 50])
            data['discount_pct'].append(discount)
            
            data['tax_amount'].append(round(random.uniform(0, 100), 2) if random.random() > 0.1 else None)
            data['currency'].append(random.choice(currencies))
        
        return pd.DataFrame(data)
    
    def generate_time_series_data(
        self,
        n_points: int = 365,
        include_edge_cases: bool = True
    ) -> pd.DataFrame:
        """
        Generate complex time series with anomalies.
        
        Edge cases:
        - Missing dates
        - Duplicates
        - Extreme spikes/drops
        - Seasonal patterns
        - Trend changes
        """
        base_date = datetime.now() - timedelta(days=n_points)
        dates = []
        values = []
        
        # Generate trend + seasonality + noise
        trend = np.linspace(100, 200, n_points)
        seasonal = 20 * np.sin(np.linspace(0, 4*np.pi, n_points))
        noise = np.random.normal(0, 5, n_points)
        
        base_values = trend + seasonal + noise
        
        for i in range(n_points):
            current_date = base_date + timedelta(days=i)
            
            # Skip some dates (gaps)
            if include_edge_cases and random.random() < 0.03:
                continue
            
            # Duplicate some dates
            if include_edge_cases and random.random() < 0.02:
                dates.append(current_date)
                values.append(base_values[i] + random.uniform(-2, 2))
            
            dates.append(current_date)
            
            # Add anomalies
            if include_edge_cases and random.random() < 0.02:
                value = base_values[i] * random.choice([3, 0.2, -1])  # Spike, drop, negative
            elif include_edge_cases and random.random() < 0.02:
                value = random.choice([None, np.nan, np.inf, -np.inf])
            else:
                value = base_values[i]
            
            values.append(value)
        
        df = pd.DataFrame({
            'date': dates,
            'value': values,
            'metric_name': ['sales'] * len(dates),
            'region': [random.choice(['North', 'South', 'East', 'West']) for _ in range(len(dates))]
        })
        
        return df
    
    def generate_text_data(
        self,
        n_records: int = 500,
        include_edge_cases: bool = True
    ) -> pd.DataFrame:
        """
        Generate complex text data.
        
        Edge cases:
        - Empty strings
        - Very long text
        - Unicode/emojis
        - HTML tags
        - Special characters
        """
        texts = []
        sentiments = []
        categories = []
        word_counts = []
        
        sample_texts = [
            "This product is amazing! I love it.",
            "Terrible experience, would not recommend.",
            "Decent quality for the price.",
            "五星好评！非常满意",
            "Отличный продукт! Рекомендую",
            "Très bon produit, livraison rapide",
            "😍🎉 Best purchase ever! 💯",
            "<script>alert('xss')</script>",
            "Normal review with some details about the product.",
            "   ",  # Whitespace only
            "",  # Empty
            None,  # Null
        ]
        
        for i in range(n_records):
            if include_edge_cases and random.random() < 0.05:
                # Edge cases
                text = random.choice([
                    None,
                    "",
                    "   ",
                    "a" * 10000,  # Very long
                    "Line1\nLine2\nLine3",  # Newlines
                    "Tab\there",  # Tabs
                    "<b>Bold</b> and <i>italic</i>",  # HTML
                    "email@test.com and https://url.com",  # URLs/emails
                    "Special: @#$%^&*()!",  # Special chars
                ])
            else:
                text = random.choice(sample_texts[:10])
            
            texts.append(text)
            sentiments.append(random.choice(['positive', 'negative', 'neutral', None, '']))
            categories.append(random.choice(['review', 'feedback', 'complaint', 'inquiry', None]))
            word_counts.append(len(str(text).split()) if text else 0)
        
        return pd.DataFrame({
            'id': range(n_records),
            'text': texts,
            'sentiment': sentiments,
            'category': categories,
            'word_count': word_counts,
            'timestamp': [datetime.now() - timedelta(hours=random.randint(0, 1000)) for _ in range(n_records)]
        })
    
    def generate_ml_dataset(
        self,
        n_samples: int = 1000,
        n_features: int = 20,
        task: str = 'classification',
        include_edge_cases: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate ML dataset with challenging characteristics.
        
        Edge cases:
        - Highly correlated features
        - Constant features
        - Feature with all nulls
        - Class imbalance
        - Outliers
        """
        # Generate features
        features = {}
        
        # Normal numeric features.
        # Ensure at least feature_0 exists so edge-case features (e.g. correlated) can be built
        # even when n_features is small (some tests use n_features=5).
        base_feature_count = max(1, n_features - 5)
        for i in range(base_feature_count):
            features[f'feature_{i}'] = np.random.randn(n_samples)
        
        if include_edge_cases:
            # Highly correlated feature
            features['feature_correlated'] = features['feature_0'] * 0.95 + np.random.randn(n_samples) * 0.1
            
            # Constant feature
            features['feature_constant'] = [42.0] * n_samples
            
            # Feature with all nulls
            features['feature_null'] = [None] * n_samples
            
            # Feature with outliers
            outlier_feature = np.random.randn(n_samples)
            outlier_idx = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
            outlier_feature[outlier_idx] = np.random.uniform(10, 100, size=len(outlier_idx))
            features['feature_outliers'] = outlier_feature
            
            # Categorical feature
            features['feature_categorical'] = [random.choice(['A', 'B', 'C', None]) for _ in range(n_samples)]
        
        X = pd.DataFrame(features)
        
        # Generate target
        if task == 'classification':
            # Imbalanced classification
            if include_edge_cases:
                y = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
            else:
                y = np.random.choice([0, 1], size=n_samples)
        else:
            # Regression with noise
            y = 3 * X['feature_0'].values + 2 * X.get('feature_1', pd.Series(np.zeros(n_samples))).values
            y = y + np.random.randn(n_samples) * 0.5
            
            if include_edge_cases:
                # Add some extreme values
                outlier_idx = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)
                y[outlier_idx] = np.random.uniform(-100, 100, size=len(outlier_idx))
        
        return X, pd.Series(y, name='target')
    
    def generate_financial_data(
        self,
        n_records: int = 200,
        include_edge_cases: bool = True
    ) -> pd.DataFrame:
        """
        Generate financial/accounting data.
        
        Edge cases:
        - Negative balances
        - Zero divisions
        - Missing financials
        - Extreme ratios
        """
        data = []
        
        for i in range(n_records):
            record = {
                'company_id': f"COMP_{i:04d}",
                'period': f"Q{random.randint(1,4)}-{random.randint(2020, 2024)}",
            }
            
            # Revenue/costs with edge cases
            if include_edge_cases and random.random() < 0.05:
                revenue = random.choice([0, -1000, None, np.nan])
                costs = random.choice([0, -500, None])
            else:
                revenue = round(random.uniform(100000, 10000000), 2)
                costs = round(revenue * random.uniform(0.3, 0.9), 2)
            
            record['revenue'] = revenue
            record['cost_of_goods'] = costs
            record['gross_profit'] = revenue - costs if revenue and costs else None
            
            # Assets/liabilities
            assets = round(random.uniform(500000, 50000000), 2)
            liabilities = round(assets * random.uniform(0.2, 1.5), 2)
            
            if include_edge_cases and random.random() < 0.05:
                assets = random.choice([0, None])
                liabilities = random.choice([0, None])
            
            record['total_assets'] = assets
            record['total_liabilities'] = liabilities
            record['equity'] = assets - liabilities if assets and liabilities else None
            
            # Other financials
            record['cash'] = round(random.uniform(10000, 1000000), 2)
            record['inventory'] = round(random.uniform(5000, 500000), 2)
            record['accounts_receivable'] = round(random.uniform(10000, 1000000), 2)
            record['accounts_payable'] = round(random.uniform(5000, 500000), 2)
            
            data.append(record)
        
        return pd.DataFrame(data)
    
    def generate_user_session_data(
        self,
        n_events: int = 10000,
        n_users: int = 500,
        include_edge_cases: bool = True
    ) -> pd.DataFrame:
        """
        Generate user session/clickstream data.
        
        Edge cases:
        - Very short sessions
        - Very long sessions
        - Missing timestamps
        - Duplicate events
        """
        events = []
        pages = ['/home', '/products', '/cart', '/checkout', '/thank-you', '/about', '/contact', None, '']
        actions = ['view', 'click', 'scroll', 'submit', None, '']
        
        user_ids = [f"user_{i:05d}" for i in range(n_users)]
        
        for i in range(n_events):
            user = random.choice(user_ids)
            
            if include_edge_cases and random.random() < 0.02:
                timestamp = random.choice([None, datetime(1970, 1, 1)])
            else:
                timestamp = datetime.now() - timedelta(
                    days=random.randint(0, 30),
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59),
                    seconds=random.randint(0, 59)
                )
            
            event = {
                'event_id': f"evt_{i:08d}",
                'user_id': user,
                'session_id': f"sess_{user}_{random.randint(1, 10)}",
                'timestamp': timestamp,
                'page': random.choice(pages),
                'action': random.choice(actions),
                'duration_sec': random.choice([1, 5, 30, 120, 600, 3600, None, -1]) if include_edge_cases else random.randint(1, 300),
                'device': random.choice(['desktop', 'mobile', 'tablet', None, '']),
                'browser': random.choice(['Chrome', 'Firefox', 'Safari', 'Edge', None]),
                'is_conversion': random.choice([True, False, None, 1, 0])
            }
            
            events.append(event)
        
        return pd.DataFrame(events)
    
    def generate_inventory_data(
        self,
        n_products: int = 500,
        include_edge_cases: bool = True
    ) -> pd.DataFrame:
        """
        Generate inventory data.
        
        Edge cases:
        - Negative stock
        - Zero cost
        - Missing categories
        """
        products = []
        
        categories = ['Electronics', 'Clothing', 'Food', 'Furniture', 'Sports', None, '']
        suppliers = [f"SUPP_{i:03d}" for i in range(20)] + [None, '']
        
        for i in range(n_products):
            if include_edge_cases and random.random() < 0.05:
                stock = random.choice([-10, 0, 100000, None])
                cost = random.choice([0, -50, None])
            else:
                stock = random.randint(0, 1000)
                cost = round(random.uniform(1, 500), 2)
            
            products.append({
                'product_id': f"PROD_{i:05d}",
                'product_name': f"Product {i}" if random.random() > 0.05 else None,
                'category': random.choice(categories),
                'stock_quantity': stock,
                'unit_cost': cost,
                'unit_price': cost * random.uniform(1.2, 3.0) if cost and cost > 0 else None,
                'supplier_id': random.choice(suppliers),
                'reorder_point': random.randint(10, 100),
                'lead_time_days': random.randint(1, 30),
                'last_restock_date': datetime.now() - timedelta(days=random.randint(0, 90))
            })
        
        return pd.DataFrame(products)
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def generate_empty_dataframe(self) -> pd.DataFrame:
        """Generate empty DataFrame for edge case testing."""
        return pd.DataFrame()
    
    def generate_single_row_dataframe(self) -> pd.DataFrame:
        """Generate single-row DataFrame for edge case testing."""
        return pd.DataFrame({'a': [1], 'b': ['text'], 'c': [datetime.now()]})
    
    def generate_all_null_dataframe(self, n_rows: int = 100) -> pd.DataFrame:
        """Generate DataFrame with all null values."""
        return pd.DataFrame({
            'col_a': [None] * n_rows,
            'col_b': [np.nan] * n_rows,
            'col_c': [None] * n_rows
        })
    
    def generate_mixed_types_column(self, n_rows: int = 100) -> pd.Series:
        """Generate column with mixed types."""
        values = []
        for _ in range(n_rows):
            values.append(random.choice([
                42,
                3.14,
                'text',
                None,
                True,
                datetime.now(),
                [1, 2, 3],
                {'key': 'value'}
            ]))
        return pd.Series(values)


# Factory function
def get_synthetic_generator(seed: int = 42) -> SyntheticDataGenerator:
    """Get synthetic data generator."""
    return SyntheticDataGenerator(seed=seed)
