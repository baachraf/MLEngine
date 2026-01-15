"""
AutoML Module
=============

Functions for AutoML integration with PyCaret and H2O.
"""

import warnings
from typing import Any, Dict, Tuple, Optional

import numpy as np
import pandas as pd

# Optional imports
PYCARET_AVAILABLE = False
H2O_AVAILABLE = False
pycaret_cl = None
pycaret_rg = None
h2o = None

try:
    from pycaret import classification as pycaret_cl
    from pycaret import regression as pycaret_rg
    PYCARET_AVAILABLE = True
except ImportError:
    warnings.warn('PyCaret is not installed. PyCaret AutoML will not be available.')

try:
    import h2o
    from h2o.sklearn import H2OAutoMLClassifier, H2OAutoMLRegressor
    H2O_AVAILABLE = True
except ImportError:
    warnings.warn('H2O is not installed. H2O AutoML will not be available.')


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
    
    Parameters
    ----------
    train_df : pd.DataFrame
        Training data
    target_column : str
        Name of target column
    problem_type : {'classification', 'regression'}
        Type of problem
    sort_by : str, optional
        Metric to sort models by (default: 'Accuracy' for classification, 'R2' for regression)
    remove_outliers : bool
        Whether to remove outliers in PyCaret setup
    enable_optimization : bool
        Whether to tune the best model
    **pycaret_kwargs : additional keyword arguments
        Passed to PyCaret setup
    
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
    
    Parameters
    ----------
    X_train : np.ndarray
        Feature matrix
    y_train : np.ndarray
        Target vector
    problem_type : {'classification', 'regression'}
        Type of problem
    max_runtime_secs : int
        Maximum runtime in seconds
    max_models : int
        Maximum number of models to train
    **h2o_kwargs : additional keyword arguments
        Passed to H2O AutoML
    
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


def get_available_automl_backends() -> list:
    """Get list of available AutoML backends."""
    backends = []
    
    if PYCARET_AVAILABLE:
        backends.append('pycaret')
    
    if H2O_AVAILABLE:
        backends.append('h2o')
    
    return backends
