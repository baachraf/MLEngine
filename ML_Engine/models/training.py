"""
Model Training and Hyperparameter Tuning
========================================

Functions for training ML models, hyperparameter tuning, and model management.
Supports traditional ML, PyCaret AutoML, and H2O AutoML (optional).
"""

import warnings
import pickle
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Union
from datetime import datetime
import os

# Core sklearn imports
try:
    from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score
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
    warnings.warn('scikit-learn not installed. Core training functions will not work.')

# Optional imports for hyperparameter tuning
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn('Optuna not installed. Optuna-based hyperparameter tuning will not work.')

try:
    from scipy.stats import uniform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn('SciPy not installed. RandomizedSearchCV with uniform distributions will not work.')

# Optional imports for AutoML
try:
    from pycaret import classification as pycaret_cl
    from pycaret import regression as pycaret_rg
    PYCARET_AVAILABLE = True
except ImportError:
    PYCARET_AVAILABLE = False
    warnings.warn('PyCaret not installed. PyCaret AutoML will not work.')

try:
    import h2o
    from h2o.sklearn import H2OAutoMLClassifier, H2OAutoMLRegressor
    H2O_AVAILABLE = True
except ImportError:
    H2O_AVAILABLE = False
    warnings.warn('H2O not installed. H2O AutoML will not work.')

# Optional imports for uncertainty quantification
try:
    from mapie.regression import MapieRegressor
    from mapie.classification import MapieClassifier
    MAPIE_AVAILABLE = True
except ImportError:
    MAPIE_AVAILABLE = False
    warnings.warn('MAPIE not installed. Uncertainty quantification will not work.')


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
        
        history = {
            'problem_type': self.problem_type,
            'model_class': str(self.model_class),
            'model_params': self.model_params,
            'training_samples': len(X_train),
            'multi_output': is_multi_output
        }
        
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
        random_state: int = 42
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
            warnings.warn('SciPy not available for uniform distributions')
        
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
        
        return random_search.best_estimator_, search_results
    
    def train_with_optuna(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_trials: int = 100,
        cv: int = 5,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
        random_state: int = 42
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
                warnings.warn(f'Unknown parameter type {param_type} for parameter {param_name}')
        
        return param_distributions
    
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
                warnings.warn(f'Unknown parameter type {param_type} for parameter {param_name}')
        
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
    tuning_method : {None, 'randomized_search', 'optuna'}
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
    
    Returns
    -------
    model : trained model
    training_info : dict with training details
    """
    from . import configs
    
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


def run_pycaret_automl(
    train_df: pd.DataFrame,
    target_column: str,
    problem_type: str = 'classification',
    sort_by: Optional[str] = None,
    remove_outliers: bool = False,
    enable_optimization: bool = False,
    **pycaret_kwargs
) -> Tuple[Any, Dict[str, Any]]:
    """
    Run PyCaret AutoML.
    
    Returns
    -------
    best_model : best model from PyCaret
    pycaret_results : dict with PyCaret results
    """
    if not PYCARET_AVAILABLE:
        raise ImportError('PyCaret is required for PyCaret AutoML')
    
    # Setup PyCaret
    if problem_type == 'classification':
        pycaret_module = pycaret_cl
        if sort_by is None:
            sort_by = 'Accuracy'
    else:
        pycaret_module = pycaret_rg
        if sort_by is None:
            sort_by = 'R2'
    
    # Setup experiment
    exp = pycaret_module.setup(
        data=train_df,
        target=target_column,
        remove_outliers=remove_outliers,
        silent=True,
        verbose=False,
        **pycaret_kwargs
    )
    
    # Compare models
    best_model = pycaret_module.compare_models(sort=sort_by, verbose=False)
    
    # Optimize if requested
    if enable_optimization:
        best_model = pycaret_module.tune_model(best_model, verbose=False)
    
    # Finalize model
    final_model = pycaret_module.finalize_model(best_model)
    
    # Get results
    results = pycaret_module.pull()
    
    return final_model, {
        'pycaret_results': results.to_dict(),
        'sort_by': sort_by,
        'remove_outliers': remove_outliers,
        'enable_optimization': enable_optimization
    }


def run_h2o_automl(
    X_train: np.ndarray,
    y_train: np.ndarray,
    problem_type: str = 'classification',
    max_runtime_secs: int = 60,
    max_models: int = 10,
    **h2o_kwargs
) -> Tuple[Any, Dict[str, Any]]:
    """
    Run H2O AutoML.
    
    Returns
    -------
    best_model : best model from H2O AutoML
    h2o_results : dict with H2O results
    """
    if not H2O_AVAILABLE:
        raise ImportError('H2O is required for H2O AutoML')
    
    # Initialize H2O
    h2o.init()
    
    # Prepare H2O frame
    train_data = pd.DataFrame(X_train)
    train_data['target'] = y_train
    
    h2o_train = h2o.H2OFrame(train_data)
    
    # Run AutoML
    if problem_type == 'classification':
        aml = h2o.H2OAutoML(
            max_runtime_secs=max_runtime_secs,
            max_models=max_models,
            **h2o_kwargs
        )
        aml.train(y='target', training_frame=h2o_train)
    else:
        aml = h2o.H2OAutoML(
            max_runtime_secs=max_runtime_secs,
            max_models=max_models,
            **h2o_kwargs
        )
        aml.train(y='target', training_frame=h2o_train)
    
    # Get leaderboard
    leaderboard = aml.leaderboard.as_data_frame()
    
    return aml.leader, {
        'leaderboard': leaderboard.to_dict(),
        'max_runtime_secs': max_runtime_secs,
        'max_models': max_models,
        'problem_type': problem_type
    }


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
            import joblib
            joblib.dump(model_info, filepath)
        except ImportError:
            warnings.warn('joblib not available, falling back to pickle')
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
        import joblib
        model_info = joblib.load(filepath)
    except:
        with open(filepath, 'rb') as f:
            model_info = pickle.load(f)
    
    return model_info['model'], model_info['metadata']


def get_available_tuning_methods() -> List[str]:
    """Get list of available hyperparameter tuning methods."""
    methods = ['basic']
    
    if SKLEARN_AVAILABLE and SCIPY_AVAILABLE:
        methods.append('randomized_search')
    
    if OPTUNA_AVAILABLE:
        methods.append('optuna')
    
    return methods


def get_available_automl_backends() -> List[str]:
    """Get list of available AutoML backends."""
    backends = []
    
    if PYCARET_AVAILABLE:
        backends.append('pycaret')
    
    if H2O_AVAILABLE:
        backends.append('h2o')
    
    return backends
