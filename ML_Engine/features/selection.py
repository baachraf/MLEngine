"""
Feature Selection and Engineering
==================================

Functions for feature selection, engineering, and dimensionality reduction.
"""

import warnings
import pandas as pd
import numpy as np
import yaml
import os
from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif, mutual_info_regression,
    f_classif, f_regression, chi2, RFE, VarianceThreshold
)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Optional imports for advanced feature selection methods
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    xgb = None
    XGBOOST_AVAILABLE = False
    logger.warning('xgboost not installed. Some feature selection methods may not be available.')

try:
    from boruta import BorutaPy
    BORUTA_AVAILABLE = True
except ImportError:
    BorutaPy = None
    BORUTA_AVAILABLE = False
    logger.warning('boruta not installed. Boruta feature selection will not be available.')


def variance_feature_selection(X, threshold=0.0):
    """
    Select features based on variance threshold.
    
    Parameters
    ----------
    X : array-like or pandas.DataFrame
        Feature data
    threshold : float, default=0.0
        Variance threshold
    
    Returns
    -------
    selected_features : array-like
        Indices of selected features
    """
    from ..data.transformation import encode_categorical
    
    # Handle categorical columns in DataFrames
    if isinstance(X, pd.DataFrame):
        # Identify categorical columns
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            # Encode categorical columns using label encoding
            X_encoded, _ = encode_categorical(X, cat_cols, method='label')
            X = X_encoded
    
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)
    return selector.get_support(indices=True)

def select_k_best_features(X, y, k=10, score_func='mutual_info', problem_type='classification'):
    """
    Select k best features using scoring function.
    
    Parameters
    ----------
    X : array-like or pandas.DataFrame
        Feature data
    y : array-like
        Target data
    k : int, default=10
        Number of features to select
    score_func : {'mutual_info', 'f_classif', 'f_regression', 'chi2'}
        Scoring function
    problem_type : {'classification', 'regression'}
        Problem type
    
    Returns
    -------
    selected_features : array-like
        Indices of selected features
    scores : array-like
        Feature scores
    """
    from ..data.transformation import encode_categorical
    
    # Handle categorical columns in DataFrames
    if isinstance(X, pd.DataFrame):
        # Identify categorical columns
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            # Encode categorical columns using label encoding
            X_encoded, _ = encode_categorical(X, cat_cols, method='label')
            X = X_encoded
    
    # Map score function names to actual functions
    score_funcs = {
        'mutual_info': mutual_info_classif if problem_type == 'classification' else mutual_info_regression,
        'f_classif': f_classif,
        'f_regression': f_regression,
        'chi2': chi2
    }
    
    if score_func not in score_funcs:
        raise ValueError(f"Unknown score function: {score_func}")
    
    score_function = score_funcs[score_func]
    
    # Chi2 requires non-negative values
    if score_func == 'chi2':
        if (X < 0).any().any():
            raise ValueError("Chi-squared test requires non-negative values")
    
    selector = SelectKBest(score_func=score_function, k=k)
    selector.fit(X, y)
    
    return selector.get_support(indices=True), selector.scores_


def recursive_feature_elimination(X, y, estimator=None, n_features_to_select=None, 
                                  step=1, problem_type='classification', **estimator_params):
    """
    Recursive Feature Elimination.
    
    Parameters
    ----------
    X : array-like or pandas.DataFrame
        Feature data
    y : array-like
        Target data
    estimator : object, optional
        Estimator to use. If None, defaults based on problem_type.
    n_features_to_select : int, optional
        Number of features to select. If None, half of features.
    step : int or float, default=1
        Number of features to remove at each iteration
    problem_type : {'classification', 'regression'}
        Problem type
    **estimator_params : dict
        Parameters for estimator
    
    Returns
    -------
    selected_features : array-like
        Indices of selected features
    rankings : array-like
        Feature rankings (1 = best)
    """
    from ..data.transformation import encode_categorical
    
    # Handle categorical columns in DataFrames
    if isinstance(X, pd.DataFrame):
        # Identify categorical columns
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            # Encode categorical columns using label encoding
            X_encoded, _ = encode_categorical(X, cat_cols, method='label')
            X = X_encoded
    
    if estimator is None:
        if problem_type == 'classification':
            estimator = RandomForestClassifier(n_estimators=100, random_state=42, **estimator_params)
        else:
            estimator = RandomForestRegressor(n_estimators=100, random_state=42, **estimator_params)
    
    if n_features_to_select is None:
        n_features_to_select = X.shape[1] // 2
    
    selector = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=step)
    selector.fit(X, y)
    
    return selector.get_support(indices=True), selector.ranking_


def get_rfe_result(estimator, estimator_params, k, step, X_train, y_train, feature_names):
    """
    Perform Recursive Feature Elimination (RFE) and return selected features.
    
    Parameters
    ----------
    estimator : sklearn estimator class
        Estimator to use (e.g., RandomForestClassifier)
    estimator_params : dict
        Parameters for the estimator
    k : int
        Number of features to select
    step : int or float
        Number/percentage of features to remove at each iteration
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    feature_names : list
        Names of features
    
    Returns
    -------
    selected_features : list
        Names of selected features
    selector : RFE
        Fitted RFE selector
    """
    from sklearn.feature_selection import RFE
    
    estimator_model = estimator(**estimator_params)
    selector = RFE(estimator=estimator_model, n_features_to_select=k, step=step)
    selector.fit(X_train, y_train)
    
    selected_feature_mask = selector.get_support()
    selected_features = [feature_names[i] for i in range(len(feature_names)) 
                         if selected_feature_mask[i]]
    
    return selected_features, selector


def boruta_feature_selection(X, y, problem_type='classification', max_iter=100, random_state=42):
    """
    Boruta all-relevant feature selection.
    
    Parameters
    ----------
    X : array-like or pandas.DataFrame
        Feature data
    y : array-like
        Target data
    problem_type : {'classification', 'regression'}
        Problem type
    max_iter : int, default=100
        Maximum number of iterations
    random_state : int, default=42
        Random seed
    
    Returns
    -------
    selected_features : array-like
        Indices of selected features
    feature_importances : array-like
        Feature importances
    
    Raises
    ------
    ImportError
        If boruta package is not installed
    """
    from ..data.transformation import encode_categorical
    
    if not BORUTA_AVAILABLE:
        raise ImportError('boruta package is required for boruta_feature_selection')
    
    # Handle categorical columns in DataFrames
    if isinstance(X, pd.DataFrame):
        # Identify categorical columns
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            # Encode categorical columns using label encoding
            X_encoded, _ = encode_categorical(X, cat_cols, method='label')
            X = X_encoded
    
    # Convert to numpy arrays if needed
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    y_arr = y.values if isinstance(y, pd.Series) else y
    
    # Choose estimator based on problem type
    if problem_type == 'classification':
        estimator = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    else:
        estimator = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    
    # Initialize Boruta
    boruta = BorutaPy(
        estimator=estimator,
        n_estimators='auto',
        max_iter=max_iter,
        random_state=random_state,
        verbose=0
    )
    
    # Fit Boruta
    boruta.fit(X_arr, y_arr)
    
    # Compute mean importance across iterations
    importances = boruta.importance_history_.mean(axis=0) if hasattr(boruta, 'importance_history_') else boruta.ranking_
    return boruta.support_, importances


def run_feature_selection_experiment(
    experiment_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    problem_type: str,
    config_path: str = None
) -> dict:
    """
    Run a feature selection experiment from a YAML configuration file.

    Parameters
    ----------
    experiment_name : str
        The name of the experiment to run.
    X : pd.DataFrame
        Feature data.
    y : pd.Series
        Target data.
    problem_type : {'classification', 'regression'}
        The type of machine learning problem.
    config_path : str, optional
        Path to the feature selection experiments YAML file.

    Returns
    -------
    dict
        A dictionary where keys are method names and values are lists of selected feature names.
    """
    if config_path is None:
        # Try finding it relative to the module (dev mode)
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        config_path = os.path.join(base_dir, 'configs', 'feature_selection_experiments.yml')
        
        if not os.path.exists(config_path):
            # Try finding it relative to current working directory
            cwd = os.getcwd()
            potential_path = os.path.join(cwd, '..', 'configs', 'feature_selection_experiments.yml')
            if os.path.exists(potential_path):
                config_path = potential_path
            else:
                 potential_path = os.path.join(cwd, 'configs', 'feature_selection_experiments.yml')
                 if os.path.exists(potential_path):
                     config_path = potential_path

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Feature selection configuration file not found at {config_path}")

    with open(config_path, 'r') as f:
        experiments = yaml.safe_load(f)

    if experiment_name not in experiments:
        raise ValueError(f"Experiment '{experiment_name}' not found in {config_path}")

    exp_config = experiments[experiment_name]
    logger.info(f"Running feature selection experiment: {exp_config.get('description', experiment_name)}")

    # Normalize problem_type to lowercase for case-insensitive matching
    problem_type = problem_type.lower()
    
    results = {}
    
    for method_spec in exp_config['methods']:
        method_name = method_spec['name']
        params = method_spec.get('params', {})
        logger.info(f"Running method: {method_name} with params: {params}")
        
        try:
            if method_name == 'variance':
                indices = variance_feature_selection(X, **params)
                results[method_name] = X.columns[indices].tolist()
            
            elif method_name == 'kbest':
                indices, _ = select_k_best_features(X, y, problem_type=problem_type, **params)
                results[method_name] = X.columns[indices].tolist()

            elif method_name == 'rfe':
                estimator_name = params.pop('estimator', 'RandomForestClassifier')
                if estimator_name == 'RandomForestClassifier':
                    estimator = RandomForestClassifier()
                elif estimator_name == 'RandomForestRegressor':
                    estimator = RandomForestRegressor()
                else:
                    raise ValueError(f"Unsupported estimator for RFE: {estimator_name}")
                
                indices, _ = recursive_feature_elimination(X, y, estimator=estimator, problem_type=problem_type, **params)
                results[method_name] = X.columns[indices].tolist()

            elif method_name == 'boruta':
                indices, _ = boruta_feature_selection(X, y, problem_type=problem_type, **params)
                results[method_name] = X.columns[indices].tolist()

        except Exception as e:
            logger.error(f"Failed to run method '{method_name}': {e}")
            results[method_name] = f"Error: {e}"
            
    return results

def apply_pca(X, n_components=None, variance=0.95, should_encode_categorical=True):
    """
    Apply Principal Component Analysis (PCA).
    Wrapper for reduction.apply_pca to ensure availability in this module.
    
    Parameters
    ----------
    X : array-like or pandas.DataFrame
        Feature data
    n_components : int, optional
        Number of components
    variance : float, default=0.95
        Variance to retain
    should_encode_categorical : bool, default=True
        Whether to encode categorical columns using label encoding
    
    Returns
    -------
    X_pca : array-like
        Transformed data
    pca : PCA
        Fitted PCA object
    """
    # Import locally to avoid circular imports if reduction imports selection
    from .reduction import apply_pca as _apply_pca
    return _apply_pca(X, n_components, variance, should_encode_categorical)
