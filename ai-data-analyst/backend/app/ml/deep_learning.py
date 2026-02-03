# AI Enterprise Data Analyst - Deep Learning Module
# Neural network architectures for tabular data, time series, and NLP

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Callable, Union
import warnings

import numpy as np
import pandas as pd

from app.core.logging import get_logger
try:
    from app.core.exceptions import ValidationException
except ImportError:
    class ValidationException(Exception): pass

logger = get_logger(__name__)

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Deep Learning Types
# ============================================================================

class NeuralNetType(str, Enum):
    """Neural network architecture types."""
    MLP = "mlp"
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    CNN_1D = "cnn_1d"
    AUTOENCODER = "autoencoder"
    VAE = "variational_autoencoder"
    TABNET = "tabnet"


class TaskType(str, Enum):
    """ML task types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    FORECASTING = "forecasting"
    ANOMALY = "anomaly_detection"
    EMBEDDING = "embedding"


@dataclass
class DeepLearningConfig:
    """Configuration for deep learning models."""
    
    architecture: NeuralNetType = NeuralNetType.MLP
    task: TaskType = TaskType.REGRESSION
    
    # Architecture params
    hidden_layers: list[int] = field(default_factory=lambda: [128, 64, 32])
    dropout_rate: float = 0.3
    activation: str = "relu"
    
    # Training params
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    
    # Regularization
    l1_reg: float = 0.0
    l2_reg: float = 0.01
    
    # LSTM/GRU specific
    sequence_length: int = 10
    bidirectional: bool = False
    
    # Transformer specific
    n_heads: int = 4
    ff_dim: int = 128
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture.value,
            "task": self.task.value,
            "hidden_layers": self.hidden_layers,
            "dropout_rate": self.dropout_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate
        }


@dataclass
class TrainingHistory:
    """Training history and metrics."""
    
    epochs_completed: int = 0
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_metrics: dict[str, list[float]] = field(default_factory=dict)
    val_metrics: dict[str, list[float]] = field(default_factory=dict)
    best_epoch: int = 0
    best_val_loss: float = float('inf')
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs_completed": self.epochs_completed,
            "best_epoch": self.best_epoch,
            "best_val_loss": round(self.best_val_loss, 6),
            "final_train_loss": round(self.train_loss[-1], 6) if self.train_loss else None,
            "final_val_loss": round(self.val_loss[-1], 6) if self.val_loss else None
        }


# ============================================================================
# Base Neural Network
# ============================================================================

class BaseNeuralNetwork(ABC):
    """Abstract base for neural network architectures."""
    
    def __init__(self, config: DeepLearningConfig):
        self.config = config
        self.model = None
        self.history: Optional[TrainingHistory] = None
        self._fitted = False
    
    @abstractmethod
    def build(self, input_shape: tuple, output_shape: int) -> None:
        """Build the neural network architecture."""
        pass
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> TrainingHistory:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    def get_embeddings(self, X: np.ndarray, layer_name: str = None) -> np.ndarray:
        """Extract embeddings from intermediate layer."""
        raise NotImplementedError("Embedding extraction not implemented")


# ============================================================================
# MLP (Multi-Layer Perceptron)
# ============================================================================

class MLPNetwork(BaseNeuralNetwork):
    """Multi-Layer Perceptron for tabular data."""
    
    def build(self, input_shape: tuple, output_shape: int) -> None:
        """Build MLP architecture."""
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers, regularizers
            
            inputs = keras.Input(shape=input_shape)
            x = inputs
            
            # Hidden layers
            for units in self.config.hidden_layers:
                x = layers.Dense(
                    units,
                    activation=self.config.activation,
                    kernel_regularizer=regularizers.l2(self.config.l2_reg)
                )(x)
                x = layers.BatchNormalization()(x)
                x = layers.Dropout(self.config.dropout_rate)(x)
            
            # Output layer
            if self.config.task == TaskType.CLASSIFICATION:
                if output_shape == 1:
                    outputs = layers.Dense(1, activation='sigmoid')(x)
                else:
                    outputs = layers.Dense(output_shape, activation='softmax')(x)
            else:
                outputs = layers.Dense(output_shape, activation='linear')(x)
            
            self.model = keras.Model(inputs, outputs)
            
            # Compile
            if self.config.task == TaskType.CLASSIFICATION:
                loss = 'binary_crossentropy' if output_shape == 1 else 'sparse_categorical_crossentropy'
                metrics = ['accuracy']
            else:
                loss = 'mse'
                metrics = ['mae']
            
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
                loss=loss,
                metrics=metrics
            )
            
        except ImportError:
            logger.warning("TensorFlow not available, using sklearn fallback")
            self._use_sklearn_fallback(input_shape, output_shape)
    
    def _use_sklearn_fallback(self, input_shape: tuple, output_shape: int) -> None:
        """Fallback to sklearn MLPClassifier/Regressor."""
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        
        hidden_layer_sizes = tuple(self.config.hidden_layers)
        
        if self.config.task == TaskType.CLASSIFICATION:
            self.model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation='relu',
                max_iter=self.config.epochs,
                early_stopping=True,
                validation_fraction=self.config.validation_split,
                random_state=42
            )
        else:
            self.model = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                activation='relu',
                max_iter=self.config.epochs,
                early_stopping=True,
                validation_fraction=self.config.validation_split,
                random_state=42
            )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> TrainingHistory:
        """Train MLP."""
        self.history = TrainingHistory()
        
        try:
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
            
            callbacks = [
                EarlyStopping(
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(factor=0.5, patience=5)
            ]
            
            history = self.model.fit(
                X, y,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                validation_split=self.config.validation_split,
                callbacks=callbacks,
                verbose=0
            )
            
            self.history.epochs_completed = len(history.history['loss'])
            self.history.train_loss = history.history['loss']
            self.history.val_loss = history.history.get('val_loss', [])
            self.history.best_epoch = np.argmin(self.history.val_loss) if self.history.val_loss else 0
            self.history.best_val_loss = min(self.history.val_loss) if self.history.val_loss else 0
            
        except (ImportError, AttributeError):
            # sklearn fallback
            self.model.fit(X, y)
            self.history.epochs_completed = self.model.n_iter_
        
        self._fitted = True
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self._fitted:
            raise ValidationException("Model not trained")
        return self.model.predict(X)


# ============================================================================
# LSTM Network
# ============================================================================

class LSTMNetwork(BaseNeuralNetwork):
    """LSTM network for sequence/time series data."""
    
    def build(self, input_shape: tuple, output_shape: int) -> None:
        """Build LSTM architecture."""
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
            
            inputs = keras.Input(shape=input_shape)
            x = inputs
            
            # LSTM layers
            for i, units in enumerate(self.config.hidden_layers):
                return_sequences = i < len(self.config.hidden_layers) - 1
                
                if self.config.bidirectional:
                    x = layers.Bidirectional(
                        layers.LSTM(units, return_sequences=return_sequences)
                    )(x)
                else:
                    x = layers.LSTM(units, return_sequences=return_sequences)(x)
                
                x = layers.Dropout(self.config.dropout_rate)(x)
            
            # Output
            if self.config.task == TaskType.CLASSIFICATION:
                outputs = layers.Dense(output_shape, activation='softmax')(x)
            else:
                outputs = layers.Dense(output_shape, activation='linear')(x)
            
            self.model = keras.Model(inputs, outputs)
            
            loss = 'sparse_categorical_crossentropy' if self.config.task == TaskType.CLASSIFICATION else 'mse'
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
                loss=loss,
                metrics=['mae'] if self.config.task != TaskType.CLASSIFICATION else ['accuracy']
            )
            
        except ImportError:
            raise ValidationException("TensorFlow required for LSTM")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> TrainingHistory:
        """Train LSTM."""
        from tensorflow.keras.callbacks import EarlyStopping
        
        self.history = TrainingHistory()
        
        callbacks = [
            EarlyStopping(patience=self.config.early_stopping_patience, restore_best_weights=True)
        ]
        
        history = self.model.fit(
            X, y,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_split=self.config.validation_split,
            callbacks=callbacks,
            verbose=0
        )
        
        self.history.train_loss = history.history['loss']
        self.history.val_loss = history.history.get('val_loss', [])
        self.history.epochs_completed = len(self.history.train_loss)
        self._fitted = True
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)


# ============================================================================
# Transformer Network
# ============================================================================

class TransformerNetwork(BaseNeuralNetwork):
    """Transformer architecture for sequences (Google pattern)."""
    
    def build(self, input_shape: tuple, output_shape: int) -> None:
        """Build Transformer architecture."""
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
            
            inputs = keras.Input(shape=input_shape)
            
            # Positional encoding
            x = self._positional_encoding(inputs, input_shape[0], input_shape[1])
            
            # Transformer blocks
            for _ in range(len(self.config.hidden_layers)):
                x = self._transformer_block(x)
            
            # Global pooling
            x = layers.GlobalAveragePooling1D()(x)
            
            # Dense layers
            x = layers.Dense(self.config.ff_dim, activation='relu')(x)
            x = layers.Dropout(self.config.dropout_rate)(x)
            
            # Output
            outputs = layers.Dense(output_shape, activation='linear')(x)
            
            self.model = keras.Model(inputs, outputs)
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
                loss='mse',
                metrics=['mae']
            )
            
        except ImportError:
            raise ValidationException("TensorFlow required for Transformer")
    
    def _positional_encoding(self, x, seq_len: int, d_model: int):
        """Add positional encoding."""
        from tensorflow.keras import layers
        
        positions = np.arange(seq_len)[:, np.newaxis]
        dims = np.arange(d_model)[np.newaxis, :]
        
        angles = positions / np.power(10000, (2 * (dims // 2)) / d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        
        pos_encoding = angles[np.newaxis, ...]
        return x + pos_encoding.astype('float32')
    
    def _transformer_block(self, x):
        """Single transformer block."""
        from tensorflow.keras import layers
        
        # Multi-head attention
        attn_output = layers.MultiHeadAttention(
            num_heads=self.config.n_heads,
            key_dim=x.shape[-1] // self.config.n_heads
        )(x, x)
        
        attn_output = layers.Dropout(self.config.dropout_rate)(attn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
        
        # Feed-forward
        ff_output = layers.Dense(self.config.ff_dim, activation='relu')(x)
        ff_output = layers.Dense(x.shape[-1])(ff_output)
        ff_output = layers.Dropout(self.config.dropout_rate)(ff_output)
        
        return layers.LayerNormalization(epsilon=1e-6)(x + ff_output)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> TrainingHistory:
        """Train Transformer."""
        from tensorflow.keras.callbacks import EarlyStopping
        
        self.history = TrainingHistory()
        
        history = self.model.fit(
            X, y,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_split=self.config.validation_split,
            callbacks=[EarlyStopping(patience=self.config.early_stopping_patience)],
            verbose=0
        )
        
        self.history.train_loss = history.history['loss']
        self.history.val_loss = history.history.get('val_loss', [])
        self._fitted = True
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


# ============================================================================
# Autoencoder
# ============================================================================

class Autoencoder(BaseNeuralNetwork):
    """Autoencoder for anomaly detection and dimensionality reduction."""
    
    def __init__(self, config: DeepLearningConfig):
        super().__init__(config)
        self.encoder = None
        self.decoder = None
        self.threshold = None
    
    def build(self, input_shape: tuple, output_shape: int = None) -> None:
        """Build Autoencoder architecture."""
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
            
            input_dim = input_shape[0] if isinstance(input_shape, tuple) else input_shape
            
            # Encoder
            inputs = keras.Input(shape=(input_dim,))
            x = inputs
            
            encoding_layers = self.config.hidden_layers
            for units in encoding_layers:
                x = layers.Dense(units, activation='relu')(x)
                x = layers.BatchNormalization()(x)
            
            # Bottleneck (latent space)
            latent_dim = encoding_layers[-1] // 2
            encoded = layers.Dense(latent_dim, activation='relu', name='latent')(x)
            
            # Decoder (mirror of encoder)
            x = encoded
            for units in reversed(encoding_layers):
                x = layers.Dense(units, activation='relu')(x)
                x = layers.BatchNormalization()(x)
            
            # Output reconstruction
            outputs = layers.Dense(input_dim, activation='linear')(x)
            
            # Full autoencoder
            self.model = keras.Model(inputs, outputs)
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
                loss='mse'
            )
            
            # Encoder only (for embeddings)
            self.encoder = keras.Model(inputs, encoded)
            
        except ImportError:
            raise ValidationException("TensorFlow required for Autoencoder")
    
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> TrainingHistory:
        """Train Autoencoder (unsupervised - reconstructs input)."""
        from tensorflow.keras.callbacks import EarlyStopping
        
        self.history = TrainingHistory()
        
        history = self.model.fit(
            X, X,  # Reconstruction task
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_split=self.config.validation_split,
            callbacks=[EarlyStopping(patience=self.config.early_stopping_patience)],
            verbose=0
        )
        
        self.history.train_loss = history.history['loss']
        self.history.val_loss = history.history.get('val_loss', [])
        
        # Calculate anomaly threshold (e.g., 95th percentile of reconstruction error)
        reconstructed = self.model.predict(X)
        mse = np.mean(np.power(X - reconstructed, 2), axis=1)
        self.threshold = np.percentile(mse, 95)
        
        self._fitted = True
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct input."""
        return self.model.predict(X)
    
    def detect_anomalies(self, X: np.ndarray) -> np.ndarray:
        """Detect anomalies based on reconstruction error."""
        reconstructed = self.model.predict(X)
        mse = np.mean(np.power(X - reconstructed, 2), axis=1)
        return mse > self.threshold
    
    def get_embeddings(self, X: np.ndarray, layer_name: str = None) -> np.ndarray:
        """Get latent space embeddings."""
        return self.encoder.predict(X)


# ============================================================================
# Deep Learning Engine
# ============================================================================

class DeepLearningEngine:
    """
    Unified deep learning engine.
    
    Supports:
    - MLP for tabular data
    - LSTM/GRU for sequences
    - Transformer for advanced sequences
    - Autoencoder for anomaly/embedding
    """
    
    def __init__(self):
        self._models: dict[str, BaseNeuralNetwork] = {}
    
    def create_model(
        self,
        name: str,
        config: DeepLearningConfig,
        input_shape: tuple,
        output_shape: int = 1
    ) -> BaseNeuralNetwork:
        """Create and register a neural network model."""
        if config.architecture == NeuralNetType.MLP:
            model = MLPNetwork(config)
        elif config.architecture == NeuralNetType.LSTM:
            model = LSTMNetwork(config)
        elif config.architecture == NeuralNetType.TRANSFORMER:
            model = TransformerNetwork(config)
        elif config.architecture == NeuralNetType.AUTOENCODER:
            model = Autoencoder(config)
        else:
            model = MLPNetwork(config)
        
        model.build(input_shape, output_shape)
        self._models[name] = model
        
        return model
    
    def train(
        self,
        name: str,
        X: np.ndarray,
        y: np.ndarray = None
    ) -> TrainingHistory:
        """Train a registered model."""
        if name not in self._models:
            raise ValidationException(f"Model '{name}' not found")
        
        return self._models[name].fit(X, y)
    
    def predict(self, name: str, X: np.ndarray) -> np.ndarray:
        """Make predictions with a model."""
        if name not in self._models:
            raise ValidationException(f"Model '{name}' not found")
        
        return self._models[name].predict(X)
    
    def get_model(self, name: str) -> BaseNeuralNetwork:
        """Get a registered model."""
        return self._models.get(name)
    
    def prepare_sequences(
        self,
        data: np.ndarray,
        sequence_length: int,
        target_column: int = -1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare data for sequence models (LSTM, Transformer)."""
        X, y = [], []
        
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length, target_column])
        
        return np.array(X), np.array(y)


# Factory function
def get_deep_learning_engine() -> DeepLearningEngine:
    """Get deep learning engine instance."""
    return DeepLearningEngine()
