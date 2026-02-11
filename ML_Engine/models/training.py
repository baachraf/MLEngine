"""
Model Training and Hyperparameter Tuning
========================================

Functions for training ML models, hyperparameter tuning, and model management.
Supports traditional ML, PyCaret AutoML, and H2O AutoML (optional).
"""

import pickle
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Union
from datetime import datetime
import os
import yaml
from ..utils.logger import get_logger
from ..evaluation import metrics as eval_metrics

logger = get_logger(__name__)

# Core sklearn imports
try:
    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split, cross_val_score
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        mean_squared_error, mean_absolute_error, r2_score,
        make_scorer
    )
    from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
    from sklearn.base import BaseEstimator
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning('scikit-learn not installed. Core training functions will not work.')

# Optional imports for hyperparameter tuning
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning('Optuna not installed. Optuna-based hyperparameter tuning will not work.')

try:
    from scipy.stats import uniform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning('SciPy not installed. RandomizedSearchCV with uniform distributions will not work.')


class CategoricalModelWrapper:
    """
    Wrapper for models trained with categorical encoding.
    Applies the same categorical encoding during prediction.
    """
    
    def __init__(self, base_model, categorical_mappings, categorical_columns):
        """
        Initialize wrapper.
        
        Parameters
        ----------
        base_model : object
            Trained model
        categorical_mappings : dict
            Dictionary of categorical column mappings
        categorical_columns : list
            List of categorical column names
        """
        self.base_model = base_model
        self.categorical_mappings = categorical_mappings
        self.categorical_columns = categorical_columns
        
        # Copy important attributes from base model
        self.__dict__.update(base_model.__dict__)
    
    def _encode_data(self, X):
        """Encode categorical columns in data."""
        import pandas as pd
        from ..data.transformation import apply_saved_label_encoding
        
        if isinstance(X, pd.DataFrame) and self.categorical_columns:
            # Make sure we only encode columns that exist in the data
            cols_to_encode = [col for col in self.categorical_columns if col in X.columns]
            if cols_to_encode:
                try:
                    X = apply_saved_label_encoding(X, cols_to_encode, self.categorical_mappings)
                except ValueError as e:
                    # If there are unknown values, we need to handle them
                    # For now, raise the error
                    raise ValueError(f"Error encoding categorical data: {e}")
        return X
    
    def predict(self, X, **kwargs):
        """Predict with encoding."""
        X_encoded = self._encode_data(X)
        return self.base_model.predict(X_encoded, **kwargs)
    
    def predict_proba(self, X, **kwargs):
        """Predict probabilities with encoding."""
        X_encoded = self._encode_data(X)
        return self.base_model.predict_proba(X_encoded, **kwargs)
    
    def score(self, X, y, **kwargs):
        """Score with encoding."""
        X_encoded = self._encode_data(X)
        return self.base_model.score(X_encoded, y, **kwargs)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return self.base_model.get_params(deep)
    
    def set_params(self, **params):
        """Set parameters for this estimator."""
        return self.base_model.set_params(**params)
    
    def __getattr__(self, name):
        """Forward other attributes to base model."""
        return getattr(self.base_model, name)

# Optional imports for AutoML
try:
    from pycaret import classification as pycaret_cl
    from pycaret import regression as pycaret_rg
    PYCARET_AVAILABLE = True
except (ImportError, RuntimeError):
    PYCARET_AVAILABLE = False
    logger.warning('PyCaret not available. PyCaret AutoML will not work.')

try:
    import h2o
    from h2o.sklearn import H2OAutoMLClassifier, H2OAutoMLRegressor
    H2O_AVAILABLE = True
except ImportError:
    H2O_AVAILABLE = False
    logger.warning('H2O not installed. H2O AutoML will not work.')

# Optional imports for uncertainty quantification
try:
    from mapie.regression import MapieRegressor
    from mapie.classification import MapieClassifier
    MAPIE_AVAILABLE = True
except ImportError:
    MAPIE_AVAILABLE = False
    logger.warning('MAPIE not installed. Uncertainty quantification will not work.')


class ModelTrainer:
    """
    Trainer for ML models with support for hyperparameter tuning and multi-output.
    """
    
    def __init__(
        self,
        problem_type: str,
        model_class: Any,
        model_config: Dict[str, Any],
        model_params: Dict[str, Any],
        class_weight=None,
        tuning_config: Optional[Dict[str, Any]] = None,
        tuning_metric: Optional[str] = None
    ):
        """
        Initialize ModelTrainer.
        
        Parameters
        ----------
        problem_type : {'classification', 'regression'}
            Type of problem
        model_class : class
            Model class (e.g., RandomForestClassifier)
        model_config : dict
            Model configuration from model_configs.py
        model_params : dict
            Model parameters for instantiation
        class_weight : dict or 'balanced', optional
            Class weights for classification
        tuning_config : dict, optional
            Hyperparameter tuning configuration
        tuning_metric : str, optional
            Metric to optimize during tuning
        """
        self.problem_type = problem_type
        self.model_class = model_class
        self.model_config = model_config
        self.model_params = model_params
        self.class_weight = class_weight
        self.tuning_config = tuning_config or {}
        self.tuning_metric = tuning_metric
        
        # Define available metrics
        self.classification_metrics = {
            'Accuracy': accuracy_score,
            'Precision': precision_score,
            'Recall': recall_score,
            'F1': f1_score
        }
        
        self.regression_metrics = {
            'RMSE': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            'MSE': mean_squared_error,
            'MAE': mean_absolute_error,
            'R2': r2_score
        }
        
        self.all_metrics = {**self.classification_metrics, **self.regression_metrics}
        
        # Categorical encoding storage
        self.categorical_mappings = {}
        self.categorical_columns = []
    
    def _handle_categorical_encoding(self, X, X_val=None):
        """
        Handle categorical encoding for input data.
        
        Parameters
        ----------
        X : array-like or pd.DataFrame
            Training data
        X_val : array-like or pd.DataFrame, optional
            Validation data
            
        Returns
        -------
        X_encoded : array-like
            Encoded training data
        X_val_encoded : array-like or None
            Encoded validation data if provided
        """
        import pandas as pd
        from ..data.transformation import encode_categorical
        
        # Reset encoder storage
        self.categorical_mappings = {}
        self.categorical_columns = []
        
        X_encoded = X
        X_val_encoded = X_val
        
        if isinstance(X, pd.DataFrame):
            # Identify categorical columns
            cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            if cat_cols:
                # Encode categorical columns using label encoding
                X_encoded, mappings = encode_categorical(X, cat_cols, method='label')
                self.categorical_mappings = mappings
                self.categorical_columns = cat_cols
                
                # Encode validation data if provided
                if X_val is not None and isinstance(X_val, pd.DataFrame):
                    from ..data.transformation import apply_saved_label_encoding
                    X_val_encoded = apply_saved_label_encoding(X_val, cat_cols, mappings)
        
        return X_encoded, X_val_encoded
    
    def _reshape_target(self, y: np.ndarray) -> np.ndarray:
        """Reshape target if needed (for single column 2D arrays)."""
        if y is None:
            return None
        if len(y.shape) == 2 and y.shape[1] == 1:
            return y.ravel()
        return y
    
    def _is_multi_output(self, y: np.ndarray) -> bool:
        """Check if target is multi-output."""
        return len(y.shape) > 1 and y.shape[1] > 1
    
    def _create_base_model(self, params: Optional[Dict[str, Any]] = None) -> Any:
        """Create model instance with given parameters."""
        if params is None:
            params = self.model_params.copy()
        
        # Add class weights if provided
        if self.class_weight is not None and 'class_weight' in self.model_class.__init__.__annotations__:
            params['class_weight'] = self.class_weight
        
        return self.model_class(**params)
    
    def _create_model_wrapper(self, base_model: Any, is_multi_output: bool) -> Any:
        """Wrap model for multi-output if needed."""
        if not is_multi_output or self.model_config.get('multi_output', False):
            return base_model
        
        if self.problem_type == 'classification':
            return MultiOutputClassifier(base_model)
        else:
            return MultiOutputRegressor(base_model)
    
    def _get_metric_function(self, metric_name: Optional[str] = None):
        """Get metric function by name."""
        if metric_name is None:
            metric_name = self.tuning_metric
        
        if metric_name in self.all_metrics:
            return self.all_metrics[metric_name]
        elif self.problem_type == 'classification' and metric_name in self.classification_metrics:
            return self.classification_metrics[metric_name]
        elif self.problem_type == 'regression' and metric_name in self.regression_metrics:
            return self.regression_metrics[metric_name]
        else:
            raise ValueError(f'Unknown metric: {metric_name}')
    
    def _get_optimization_direction(self, metric_name: str) -> str:
        """Determine optimization direction for a metric."""
        minimize_metrics = ['RMSE', 'MSE', 'MAE']
        maximize_metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'R2']
        
        if metric_name in minimize_metrics:
            return 'minimize'
        elif metric_name in maximize_metrics:
            return 'maximize'
        else:
            # Default to maximize for classification, minimize for regression
            return 'minimize' if self.problem_type == 'regression' else 'maximize'
    
    def train_basic(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **fit_params
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train model with basic settings (no hyperparameter tuning).
        
        Returns
        -------
        model : trained model
        history : dict with training information
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError('scikit-learn is required for training')
        
        # Handle categorical encoding
        X_train, X_val = self._handle_categorical_encoding(X_train, X_val)
        
        # Store encoder info for history
        encoder_info = {}
        if self.categorical_mappings:
            encoder_info['categorical_mappings'] = self.categorical_mappings
            encoder_info['categorical_columns'] = self.categorical_columns
        
        y_train = self._reshape_target(y_train)
        is_multi_output = self._is_multi_output(y_train)
        
        # Create and train model
        base_model = self._create_base_model()
        model = self._create_model_wrapper(base_model, is_multi_output)
        
        # Prepare fit parameters
        fit_kwargs = fit_params.copy()
        if sample_weight is not None:
            fit_kwargs['sample_weight'] = sample_weight
        
        # Add validation set for early stopping if supported
        early_stopping_rounds = self.model_params.get('early_stopping_rounds', 0)
        if early_stopping_rounds > 0 and X_val is not None and y_val is not None:
            if hasattr(model, 'fit') and hasattr(model, 'eval_set'):
                fit_kwargs['eval_set'] = [(X_val, y_val)]
        
        # Train model
        model.fit(X_train, y_train, **fit_kwargs)
        
        # Wrap model for categorical encoding if needed
        if self.categorical_mappings:
            model = CategoricalModelWrapper(model, self.categorical_mappings, self.categorical_columns)
        
        history = {
            'problem_type': self.problem_type,
            'model_class': str(self.model_class),
            'model_params': self.model_params,
            'training_samples': len(X_train),
            'multi_output': is_multi_output
        }
        
        # Add encoder info if categorical columns were encoded
        if encoder_info:
            history['categorical_encoding'] = encoder_info
        
        return model, history
    
    def train_with_randomized_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_trials: int = 100,
        cv: int = 5,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
        random_state: int = 42,
        **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train model with RandomizedSearchCV hyperparameter tuning.
        
        Returns
        -------
        best_model : best model from search
        search_results : dict with search results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError('scikit-learn is required for RandomizedSearchCV')
        if not SCIPY_AVAILABLE:
            logger.warning('SciPy not available for uniform distributions')
        
        # Handle categorical encoding
        X_train, X_val = self._handle_categorical_encoding(X_train, X_val)
        
        y_train = self._reshape_target(y_train)
        is_multi_output = self._is_multi_output(y_train)
        
        # Create base model
        base_model = self._create_base_model({})  # Empty params, tuned by RandomizedSearchCV
        model = self._create_model_wrapper(base_model, is_multi_output)
        
        # Prepare parameter distributions
        param_distributions = self._convert_tuning_config_to_distributions()
        
        # Setup scoring
        metric_func = self._get_metric_function(self.tuning_metric)
        direction = self._get_optimization_direction(self.tuning_metric)
        
        # Create appropriate scorer
        if hasattr(model, 'predict_proba') and self.problem_type == 'classification':
            scorer = make_scorer(metric_func, needs_proba=True)
        else:
            scorer = make_scorer(metric_func, greater_is_better=(direction == 'maximize'))
        
        # Configure RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=n_trials,
            scoring=scorer,
            cv=cv,
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )
        
        # Prepare fit parameters
        fit_params = {}
        if sample_weight is not None:
            fit_params['sample_weight'] = sample_weight
        
        # Fit RandomizedSearchCV
        random_search.fit(X_train, y_train, **fit_params)
        
        # Prepare results
        search_results = {
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'best_index': random_search.best_index_,
            'cv_results': random_search.cv_results_,
            'n_trials': n_trials,
            'cv_folds': cv
        }
        
        # Wrap best estimator for categorical encoding if needed
        best_estimator = random_search.best_estimator_
        if self.categorical_mappings:
            best_estimator = CategoricalModelWrapper(best_estimator, self.categorical_mappings, self.categorical_columns)
        
        return best_estimator, search_results
    
    def train_with_grid_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv: int = 5,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
        random_state: int = 42, # Unused but kept for API consistency
        **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train model with GridSearchCV hyperparameter tuning.
        
        Returns
        -------
        best_model : best model from search
        search_results : dict with search results
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError('scikit-learn is required for GridSearchCV')
        
        # Handle categorical encoding
        X_train, X_val = self._handle_categorical_encoding(X_train, X_val)
        
        y_train = self._reshape_target(y_train)
        is_multi_output = self._is_multi_output(y_train)
        
        # Create base model
        base_model = self._create_base_model({})  # Empty params, tuned by GridSearchCV
        model = self._create_model_wrapper(base_model, is_multi_output)
        
        # Prepare parameter grid
        param_grid = self._convert_tuning_config_to_grid()
        
        # Setup scoring
        metric_func = self._get_metric_function(self.tuning_metric)
        direction = self._get_optimization_direction(self.tuning_metric)
        
        # Create appropriate scorer
        if hasattr(model, 'predict_proba') and self.problem_type == 'classification':
            scorer = make_scorer(metric_func, needs_proba=True)
        else:
            scorer = make_scorer(metric_func, greater_is_better=(direction == 'maximize'))
        
        # Configure GridSearchCV
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=scorer,
            cv=cv,
            n_jobs=-1,
            verbose=0
        )
        
        # Prepare fit parameters
        fit_params = {}
        if sample_weight is not None:
            fit_params['sample_weight'] = sample_weight
        
        # Fit GridSearchCV
        grid_search.fit(X_train, y_train, **fit_params)
        
        # Prepare results
        search_results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_index': grid_search.best_index_,
            'cv_results': grid_search.cv_results_,
            'cv_folds': cv
        }
        
        # Wrap best estimator for categorical encoding if needed
        best_estimator = grid_search.best_estimator_
        if self.categorical_mappings:
            best_estimator = CategoricalModelWrapper(best_estimator, self.categorical_mappings, self.categorical_columns)
        
        return best_estimator, search_results

    def train_with_optuna(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_trials: int = 100,
        cv: int = 5,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
        random_state: int = 42,
        **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train model with Optuna hyperparameter tuning.
        
        Returns
        -------
        best_model : best model from Optuna optimization
        study : Optuna study object
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError('Optuna is required for Optuna-based tuning')
        
        # Handle categorical encoding
        X_train, X_val = self._handle_categorical_encoding(X_train, X_val)
        
        y_train = self._reshape_target(y_train)
        is_multi_output = self._is_multi_output(y_train)
        
        # Define objective function for Optuna
        def objective(trial):
            # Sample parameters from tuning config
            trial_params = self._sample_params_from_trial(trial)
            
            # Create model with trial parameters
            base_model = self._create_base_model(trial_params)
            model = self._create_model_wrapper(base_model, is_multi_output)
            
            # Prepare fit parameters
            fit_params = {}
            if sample_weight is not None:
                fit_params['sample_weight'] = sample_weight
            
            # Perform cross-validation
            scores = cross_val_score(
                model, X_train, y_train,
                scoring=self.tuning_metric.lower() if self.tuning_metric else None,
                cv=cv,
                fit_params=fit_params,
                n_jobs=-1
            )
            
            return scores.mean()
        
        # Create and run Optuna study
        direction = self._get_optimization_direction(self.tuning_metric)
        study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=random_state))
        study.optimize(objective, n_trials=n_trials)
        
        # Train final model with best parameters
        best_params = study.best_params
        base_model = self._create_base_model(best_params)
        best_model = self._create_model_wrapper(base_model, is_multi_output)
        
        fit_params = {}
        if sample_weight is not None:
            fit_params['sample_weight'] = sample_weight
        
        best_model.fit(X_train, y_train, **fit_params)
        
        # Wrap model for categorical encoding if needed
        if self.categorical_mappings:
            best_model = CategoricalModelWrapper(best_model, self.categorical_mappings, self.categorical_columns)
        
        return best_model, {'study': study, 'best_params': best_params, 'best_value': study.best_value}
    
    def _convert_tuning_config_to_distributions(self) -> Dict[str, Any]:
        """Convert tuning configuration to parameter distributions for RandomizedSearchCV."""
        if not self.tuning_config:
            return {}
        
        param_distributions = {}
        for param_name, param_config in self.tuning_config.items():
            param_type = param_config.get('type', 'float')
            param_values = param_config.get('value', [0, 1])
            
            if param_type == 'int':
                param_distributions[param_name] = list(range(int(param_values[0]), int(param_values[1]) + 1))
            elif param_type == 'float':
                param_distributions[param_name] = uniform(loc=float(param_values[0]), scale=float(param_values[1]) - float(param_values[0]))
            elif param_type == 'cat':
                param_distributions[param_name] = param_values
            else:
                logger.warning(f'Unknown parameter type {param_type} for parameter {param_name}')
        
        return param_distributions
    
    def _convert_tuning_config_to_grid(self, n_grid_points: int = 5) -> Dict[str, Any]:
        """Convert tuning configuration to parameter grid for GridSearchCV."""
        if not self.tuning_config:
            return {}
        
        param_grid = {}
        for param_name, param_config in self.tuning_config.items():
            param_type = param_config.get('type', 'float')
            param_values = param_config.get('value', [0, 1])
            
            if param_type == 'int':
                # Create linear space of integers
                start, end = int(param_values[0]), int(param_values[1])
                # Ensure we don't create more points than integers in range
                num = min(n_grid_points, end - start + 1)
                param_grid[param_name] = np.linspace(start, end, num=num, dtype=int).tolist()
            elif param_type == 'float':
                # Create linear space of floats
                start, end = float(param_values[0]), float(param_values[1])
                param_grid[param_name] = np.linspace(start, end, num=n_grid_points).tolist()
            elif param_type == 'cat':
                param_grid[param_name] = param_values
            else:
                logger.warning(f'Unknown parameter type {param_type} for parameter {param_name}')
        
        return param_grid

    def _sample_params_from_trial(self, trial) -> Dict[str, Any]:
        """Sample parameters from Optuna trial based on tuning configuration."""
        if not self.tuning_config:
            return {}
        
        trial_params = {}
        for param_name, param_config in self.tuning_config.items():
            param_type = param_config.get('type', 'float')
            param_values = param_config.get('value', [0, 1])
            
            if param_type == 'int':
                trial_params[param_name] = trial.suggest_int(
                    param_name, int(param_values[0]), int(param_values[1])
                )
            elif param_type == 'float':
                trial_params[param_name] = trial.suggest_float(
                    param_name, float(param_values[0]), float(param_values[1])
                )
            elif param_type == 'cat':
                trial_params[param_name] = trial.suggest_categorical(param_name, param_values)
            else:
                logger.warning(f'Unknown parameter type {param_type} for parameter {param_name}')
        
        return trial_params


def train_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    problem_type: str = 'classification',
    model_params: Optional[Dict[str, Any]] = None,
    tuning_method: Optional[str] = None,
    tuning_config: Optional[Dict[str, Any]] = None,
    tuning_metric: Optional[str] = None,
    n_trials: int = 100,
    cv: int = 5,
    sample_weight: Optional[np.ndarray] = None,
    class_weight = None,
    **kwargs
) -> Tuple[Any, Dict[str, Any]]:
    """
    High-level function to train a model with optional hyperparameter tuning.
    
    Parameters
    ----------
    model_name : str
        Name of model (must be in model_configs)
    X_train, y_train : array-like
        Training data
    problem_type : {'classification', 'regression'}
    model_params : dict, optional
        Model-specific parameters
    tuning_method : {None, 'randomized_search', 'grid_search', 'optuna'}
        Hyperparameter tuning method
    tuning_config : dict, optional
        Tuning configuration
    tuning_metric : str, optional
        Metric to optimize
    n_trials : int
        Number of trials for tuning
    cv : int
        Cross-validation folds
    sample_weight : array-like, optional
        Sample weights
    class_weight : dict or 'balanced', optional
        Class weights for classification
    **kwargs : dict
        Additional arguments passed to the training method.
    
    Returns
    -------
    model : trained model
    training_info : dict with training details
    """
    from . import configs
    
    # Handle tuning_config_path from kwargs
    if 'tuning_config_path' in kwargs:
        tuning_config_path = kwargs.pop('tuning_config_path')
        if tuning_config_path is not None and isinstance(tuning_config_path, str):
            import yaml
            with open(tuning_config_path, 'r') as f:
                tuning_config = yaml.safe_load(f)
    # Ensure tuning_config_path is not in kwargs (safety)
    kwargs.pop('tuning_config_path', None)
    # Also handle tuning_config being a string path
    if isinstance(tuning_config, str):
        import yaml
        with open(tuning_config, 'r') as f:
            tuning_config = yaml.safe_load(f)
    
    # Extract model-specific tuning config if tuning_config is nested by model name
    if tuning_config is not None and isinstance(tuning_config, dict) and model_name in tuning_config:
        tuning_config = tuning_config[model_name]

    # Get model configuration
    try:
        config = configs.get_model_config(model_name, problem_type)
    except ValueError as e:
        raise ValueError(f'Model {model_name} not found for {problem_type}: {e}')
    
    # Default model parameters
    if model_params is None:
        model_params = {}
    
    # Default tuning metric
    if tuning_metric is None:
        tuning_metric = 'Accuracy' if problem_type == 'classification' else 'R2'
    
    # Create trainer
    trainer = ModelTrainer(
        problem_type=problem_type,
        model_class=config['class'],
        model_config=config,
        model_params=model_params,
        class_weight=class_weight,
        tuning_config=tuning_config,
        tuning_metric=tuning_metric
    )
    
    # Train based on method
    if tuning_method == 'randomized_search':
        model, info = trainer.train_with_randomized_search(
            X_train, y_train,
            n_trials=n_trials,
            cv=cv,
            sample_weight=sample_weight,
            **kwargs
        )
    elif tuning_method == 'grid_search':
        model, info = trainer.train_with_grid_search(
            X_train, y_train,
            cv=cv,
            sample_weight=sample_weight,
            **kwargs
        )
    elif tuning_method == 'optuna':
        if not OPTUNA_AVAILABLE:
            raise ImportError('Optuna required for optuna tuning method')
        model, info = trainer.train_with_optuna(
            X_train, y_train,
            n_trials=n_trials,
            cv=cv,
            sample_weight=sample_weight,
            **kwargs
        )
    else:
        # Basic training
        model, info = trainer.train_basic(
            X_train, y_train,
            sample_weight=sample_weight,
            **kwargs
        )
    
    # Add model name to info
    info['model_name'] = model_name
    info['problem_type'] = problem_type
    info['tuning_method'] = tuning_method
    
    return model, info


def tune_hyperparameters(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    problem_type: str = 'classification',
    n_trials: int = 100,
    cv: int = 5,
    random_state: int = 42,
    **kwargs
) -> Tuple[Dict[str, Any], Any]:
    """
    High-level function for hyperparameter tuning using Optuna.
    
    Parameters
    ----------
    model_name : str
        Name of model (must be in model_configs)
    X_train, y_train : array-like
        Training data
    problem_type : {'classification', 'regression'}
    n_trials : int
        Number of Optuna trials
    cv : int
        Cross-validation folds
    random_state : int
        Random seed for Optuna sampler
    **kwargs
        Additional arguments passed to train_model
        
    Returns
    -------
    best_params : dict
        Best hyperparameters found
    study : optuna.study.Study
        Optuna study object (or None if Optuna not available)
    """
    # Use train_model with optuna tuning
    model, info = train_model(
        model_name=model_name,
        X_train=X_train,
        y_train=y_train,
        problem_type=problem_type,
        tuning_method='optuna',
        n_trials=n_trials,
        cv=cv,
        random_state=random_state,
        **kwargs
    )
    # Extract best_params and study from info
    best_params = info.get('best_params', {})
    study = info.get('study', None)
    return best_params, study


def train_and_compare_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    problem_type: str,
    models_to_compare: Optional[List[str]] = None,
    sort_by: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Train and compare multiple scikit-learn models.

    Parameters
    ----------
    X_train, y_train, X_test, y_test : array-like
        Training and testing data.
    problem_type : {'classification', 'regression'}
        The type of machine learning problem.
    models_to_compare : list of str, optional
        A list of model names to compare. If None, all models for the problem type will be used.
    sort_by : str, optional
        The metric to sort the results by. Defaults to 'Accuracy' for classification and 'R2' for regression.
    **kwargs
        Additional arguments passed to the `train_model` function.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the performance metrics for each model, sorted by the specified metric.
    """
    from . import configs
    
    if models_to_compare is None:
        models_to_compare = configs.list_models(problem_type)
        
    if sort_by is None:
        sort_by = 'Accuracy' if problem_type == 'classification' else 'R2'

    results_list = []
    
    for model_name in models_to_compare:
        logger.info(f"Training {model_name}...")
        try:
            model, _ = train_model(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                problem_type=problem_type,
                **kwargs
            )
            
            y_pred = model.predict(X_test)
            
            # Determine metrics to calculate
            if problem_type == 'classification':
                metrics_to_calc = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
                target_cols = y_test.columns if isinstance(y_test, pd.DataFrame) else ['target']
            else:
                metrics_to_calc = ['R2', 'MSE', 'MAE']
                target_cols = y_test.columns if isinstance(y_test, pd.DataFrame) else ['target']

            result = eval_metrics.get_metric_result(
                y_true=y_test,
                y_pred=y_pred,
                selected_metrics=metrics_to_calc,
                problem_type=problem_type,
                selected_targets=target_cols
            )
            
            # Flatten the results for the DataFrame
            flat_result = {'Model': model_name}
            for target_name, metrics in result.items():
                for metric_name, value in metrics.items():
                    # Add target name to metric if there are multiple targets
                    key = f"{target_name}_{metric_name}" if len(target_cols) > 1 else metric_name
                    flat_result[key] = value
            
            results_list.append(flat_result)
            
        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")

    if not results_list:
        return pd.DataFrame()

    results_df = pd.DataFrame(results_list)
    
    # Sort the results
    if sort_by in results_df.columns:
        ascending = sort_by in ['MSE', 'MAE']
        results_df = results_df.sort_values(by=sort_by, ascending=ascending)
        
    return results_df.reset_index(drop=True)


def run_experiment(
    experiment_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config_path: str = None,
    **kwargs
) -> pd.DataFrame:
    """
    Run a pre-defined experiment from a YAML configuration file.

    Parameters
    ----------
    experiment_name : str
        The name of the experiment to run (must be a key in the YAML file).
    X_train, y_train, X_test, y_test : array-like
        Training and testing data.
    config_path : str, optional
        Path to the experiments YAML file. Defaults to `configs/experiments.yml`.
    **kwargs
        Additional arguments passed to the `train_model` function (e.g., cv, sample_weight).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the performance metrics for each model in the experiment.
    """
    if config_path is None:
        # Try finding it relative to the module (dev mode)
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        config_path = os.path.join(base_dir, 'configs', 'experiments.yml')
        
        if not os.path.exists(config_path):
            # Try finding it relative to current working directory
            cwd = os.getcwd()
            potential_path = os.path.join(cwd, '..', 'configs', 'experiments.yml')
            if os.path.exists(potential_path):
                config_path = potential_path
            else:
                 potential_path = os.path.join(cwd, 'configs', 'experiments.yml')
                 if os.path.exists(potential_path):
                     config_path = potential_path

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Experiments configuration file not found at {config_path}")

    with open(config_path, 'r') as f:
        experiments = yaml.safe_load(f)

    if experiment_name not in experiments:
        raise ValueError(f"Experiment '{experiment_name}' not found in {config_path}")

    exp_config = experiments[experiment_name]
    logger.info(f"Running experiment: {exp_config.get('description', experiment_name)}")

    problem_type = exp_config['problem_type']
    sort_by = exp_config.get('sort_by')
    
    results_list = []

    for model_spec in exp_config['models']:
        model_name = model_spec['name']
        model_params = model_spec.get('params', {})
        
        logger.info(f"Training {model_name} with custom parameters...")
        try:
            model, _ = train_model(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                problem_type=problem_type,
                model_params=model_params,
                **kwargs
            )
            
            y_pred = model.predict(X_test)
            
            if problem_type == 'classification':
                metrics_to_calc = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
                target_cols = y_test.columns if isinstance(y_test, pd.DataFrame) else ['target']
            else:
                metrics_to_calc = ['R2', 'MSE', 'MAE']
                target_cols = y_test.columns if isinstance(y_test, pd.DataFrame) else ['target']

            result = eval_metrics.get_metric_result(
                y_true=y_test,
                y_pred=y_pred,
                selected_metrics=metrics_to_calc,
                problem_type=problem_type,
                selected_targets=target_cols
            )
            
            flat_result = {'Model': model_name}
            for target_name, metrics in result.items():
                for metric_name, value in metrics.items():
                    key = f"{target_name}_{metric_name}" if len(target_cols) > 1 else metric_name
                    flat_result[key] = value
            
            results_list.append(flat_result)
            
        except Exception as e:
            logger.error(f"Failed to train {model_name} in experiment '{experiment_name}': {e}")

    if not results_list:
        return pd.DataFrame()

    results_df = pd.DataFrame(results_list)
    
    if sort_by and sort_by in results_df.columns:
        ascending = sort_by in ['MSE', 'MAE']
        results_df = results_df.sort_values(by=sort_by, ascending=ascending)
        
    return results_df.reset_index(drop=True)


from .automl import run_pycaret_automl, run_h2o_automl, get_available_automl_backends


def save_model(
    model: Any,
    filepath: str,
    metadata: Optional[Dict[str, Any]] = None,
    save_format: str = 'pickle'
) -> str:
    """
    Save trained model to file.
    
    Parameters
    ----------
    model : trained model
    filepath : str
        Path to save model
    metadata : dict, optional
        Additional metadata to save
    save_format : {'pickle', 'joblib'}
        Format to save model
    
    Returns
    -------
    str : path where model was saved
    """
    if metadata is None:
        metadata = {}
    
    # Add timestamp
    metadata['saved_at'] = datetime.now().isoformat()
    
    # Create model info dictionary
    model_info = {
        'model': model,
        'metadata': metadata
    }
    
    # Save based on format
    if save_format == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(model_info, f)
    elif save_format == 'joblib':
        try:
            from joblib import dump
            dump(model_info, filepath)
        except ImportError:
            logger.warning('joblib not available, falling back to pickle')
            with open(filepath, 'wb') as f:
                pickle.dump(model_info, f)
    else:
        raise ValueError(f'Unsupported save_format: {save_format}')
    
    return filepath


def load_model(filepath: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Load trained model from file.
    
    Returns
    -------
    model : loaded model
    metadata : model metadata
    """
    # Try joblib first, then pickle
    try:
        from joblib import load
        model_info = load(filepath)
    except:
        with open(filepath, 'rb') as f:
            model_info = pickle.load(f)
    
    return model_info['model'], model_info['metadata']


def get_available_tuning_methods() -> List[str]:
    """Get list of available hyperparameter tuning methods."""
    methods = ['basic']
    
    if SKLEARN_AVAILABLE:
        methods.append('grid_search')

    if SKLEARN_AVAILABLE and SCIPY_AVAILABLE:
        methods.append('randomized_search')
    
    if OPTUNA_AVAILABLE:
        methods.append('optuna')
    
    return methods


def run_model_comparison_experiment(X, y, problem_type, models_to_compare=None, **kwargs):
    """
    Alias for train_and_compare_models for backward compatibility.
    
    Parameters
    ----------
    X : array-like
        Feature data
    y : array-like
        Target data
    problem_type : {'classification', 'regression'}
        Problem type
    models_to_compare : list of str, optional
        Models to compare
    **kwargs : dict
        Additional arguments passed to train_and_compare_models
    
    Returns
    -------
    dict
        Dictionary with model comparison results (different format than train_and_compare_models)
    """
    from sklearn.model_selection import train_test_split
    
    # Split data for compatibility with train_and_compare_models signature
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Call the actual function
    results_df = train_and_compare_models(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        problem_type=problem_type,
        models_to_compare=models_to_compare,
        **kwargs
    )
    
    # Convert to dictionary format for backward compatibility
    results_dict = {}
    for _, row in results_df.iterrows():
        model_name = row['Model']
        results_dict[model_name] = {
            metric: row[metric] for metric in results_df.columns if metric != 'Model'
        }
    
    return results_dict
