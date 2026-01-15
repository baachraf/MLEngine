"""
Feature Selection and Engineering
==================================

Functions for feature selection, engineering, and dimensionality reduction.
"""

import warnings
import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif, mutual_info_regression,
    f_classif, f_regression, chi2, RFE, VarianceThreshold
)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
# Optional imports for advanced feature selection methods
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    xgb = None
    XGBOOST_AVAILABLE = False
    warnings.warn('xgboost not installed. Some feature selection methods may not be available.')

try:
    from boruta import BorutaPy
    BORUTA_AVAILABLE = True
except ImportError:
    BorutaPy = None
    BORUTA_AVAILABLE = False
    warnings.warn('boruta not installed. Boruta feature selection will not be available.')


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
    
    selector = SelectKBest(score_function=score_function, k=k)
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
    if not BORUTA_AVAILABLE:
        raise ImportError('boruta package is required for boruta_feature_selection')
    
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
    
    return boruta.support_, boruta.importances_


def calculate_polynomial_features_count(n_features, degree, interaction_only=False, include_bias=False):
    """
    Calculate number of polynomial features.
    
    Parameters
    ----------
    n_features : int
        Number of input features
    degree : int
        Polynomial degree
    interaction_only : bool, default=False
        Whether to include only interaction terms
    include_bias : bool, default=False
        Whether to include bias term
    
    Returns
    -------
    int
        Number of polynomial features
    """
    # Simple calculation - exact formula depends on implementation
    # This is approximate
    if interaction_only:
        # Only interaction terms
        count = 0
        for d in range(1, degree + 1):
            if d == 1:
                count += n_features
            else:
                # Combinations of features for interaction
                # This is simplified
                count += np.math.comb(n_features, d)
    else:
        # All polynomial terms
        count = np.math.comb(n_features + degree, degree) - (0 if include_bias else 1)
    
    return count


def create_polynomial_features(X, degree=2, interaction_only=False, include_bias=False, 
                               return_poly=False):
    """
    Create polynomial and interaction features.
    
    Parameters
    ----------
    X : array-like or pandas.DataFrame
        Feature data
    degree : int, default=2
        Polynomial degree
    interaction_only : bool, default=False
        Whether to include only interaction terms
    include_bias : bool, default=False
        Whether to include bias term
    return_poly : bool, default=False
        Whether to return the PolynomialFeatures object
    
    Returns
    -------
    X_poly : array-like or pandas.DataFrame
        Polynomial features
    poly : PolynomialFeatures, optional
        Fitted PolynomialFeatures object (only if return_poly=True)
    """
    poly = PolynomialFeatures(
        degree=degree,
        interaction_only=interaction_only,
        include_bias=include_bias
    )
    
    X_poly = poly.fit_transform(X)
    
    # Create column names if X is a DataFrame
    if isinstance(X, pd.DataFrame):
        feature_names = poly.get_feature_names_out(X.columns)
        X_poly = pd.DataFrame(X_poly, columns=feature_names, index=X.index)
    
    if return_poly:
        return X_poly, poly
    else:
        return X_poly


def apply_polynomial_features(df, degree, include_bias, interaction_only, 
                              selected_cols, poly=None):
    """
    Apply polynomial feature transformation to selected columns.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame
    degree : int
        Polynomial degree
    include_bias : bool
        Whether to include bias term
    interaction_only : bool
        Whether to include only interaction terms
    selected_cols : list
        Columns to transform
    poly : PolynomialFeatures, optional
        Pre-fitted PolynomialFeatures object
    
    Returns
    -------
    transformed_df : pandas.DataFrame
        DataFrame with polynomial features
    poly : PolynomialFeatures
        Fitted PolynomialFeatures object
    """
    X = df[selected_cols]
    
    if poly is None:
        poly = PolynomialFeatures(
            degree=degree,
            include_bias=include_bias,
            interaction_only=interaction_only
        )
        X_poly = poly.fit_transform(X)
    else:
        X_poly = poly.transform(X)
    
    # Create column names
    feature_names = poly.get_feature_names_out(selected_cols)
    
    # Create new DataFrame with polynomial features
    poly_df = pd.DataFrame(X_poly, columns=feature_names, index=df.index)
    
    # Drop original columns and concatenate polynomial features
    transformed_df = df.drop(columns=selected_cols)
    transformed_df = pd.concat([transformed_df, poly_df], axis=1)
    
    return transformed_df, poly


def select_features_by_method(X, y, method='variance', problem_type='classification', **kwargs):
    """
    Select features using specified method.
    
    Parameters
    ----------
    X : array-like or pandas.DataFrame
        Feature data
    y : array-like
        Target data
    method : {'variance', 'kbest', 'rfe', 'boruta', 'pca'}
        Feature selection method
    problem_type : {'classification', 'regression'}
        Problem type
    **kwargs : dict
        Method-specific parameters
    
    Returns
    -------
    selected_indices : array-like
        Indices of selected features
    additional_info : dict
        Method-specific additional information
    """
    if method == 'variance':
        threshold = kwargs.get('threshold', 0.0)
        selected_indices = variance_feature_selection(X, threshold)
        additional_info = {'method': 'variance', 'threshold': threshold}
    
    elif method == 'kbest':
        k = kwargs.get('k', 10)
        score_func = kwargs.get('score_func', 'mutual_info')
        selected_indices, scores = select_k_best_features(
            X, y, k=k, score_func=score_func, problem_type=problem_type
        )
        additional_info = {
            'method': 'kbest',
            'k': k,
            'score_func': score_func,
            'scores': scores
        }
    
    elif method == 'rfe':
        estimator = kwargs.get('estimator')
        n_features = kwargs.get('n_features_to_select')
        step = kwargs.get('step', 1)
        selected_indices, rankings = recursive_feature_elimination(
            X, y, estimator=estimator, n_features_to_select=n_features,
            step=step, problem_type=problem_type
        )
        additional_info = {
            'method': 'rfe',
            'n_features_to_select': n_features,
            'step': step,
            'rankings': rankings
        }
    
    elif method == 'boruta':
        max_iter = kwargs.get('max_iter', 100)
        random_state = kwargs.get('random_state', 42)
        selected_indices, importances = boruta_feature_selection(
            X, y, problem_type=problem_type, max_iter=max_iter, random_state=random_state
        )
        additional_info = {
            'method': 'boruta',
            'max_iter': max_iter,
            'importances': importances
        }
    
    elif method == 'pca':
        n_components = kwargs.get('n_components', None)
        variance = kwargs.get('variance', 0.95)
        selected_indices, pca_info = apply_pca(
            X, n_components=n_components, variance=variance, return_indices=True
        )
        additional_info = {
            'method': 'pca',
            'n_components': n_components,
            'variance': variance,
            'pca_info': pca_info
        }
    
    else:
        raise ValueError(f"Unknown feature selection method: {method}")
    
    return selected_indices, additional_info


def apply_pca(X, n_components=None, variance=0.95, return_indices=False):
    """
    Apply Principal Component Analysis.
    
    Parameters
    ----------
    X : array-like or pandas.DataFrame
        Feature data
    n_components : int, optional
        Number of components. If None, use variance threshold.
    variance : float, default=0.95
        Variance to retain (used if n_components is None)
    return_indices : bool, default=False
        Whether to return original feature indices (not applicable for PCA)
    
    Returns
    -------
    X_pca : array-like
        Transformed data
    pca : PCA
        Fitted PCA object
    """
    pca = PCA(n_components=n_components)
    
    if n_components is None:
        # Fit PCA to determine number of components for variance threshold
        pca.fit(X)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= variance) + 1
        pca = PCA(n_components=n_components)
    
    X_pca = pca.fit_transform(X)
    
    if return_indices:
        # For PCA, we don't have original feature indices
        # Return component indices instead
        indices = np.arange(n_components)
        pca_info = {
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'components': pca.components_,
            'n_components': n_components,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)
        }
        return indices, pca_info
    else:
        return X_pca, pca


def generate_x_train_y_train_test(train_df, test_df, selected_features, selected_targets):
    """
    Generate X_train, X_test, y_train, y_test from train and test dataframes.
    
    Parameters
    ----------
    train_df : pandas.DataFrame
        Training dataframe
    test_df : pandas.DataFrame
        Test dataframe
    selected_features : list
        Names of selected feature columns
    selected_targets : list
        Names of target columns
    
    Returns
    -------
    dict
        Dictionary with keys: X_train, X_test, y_train, y_test, train_df, test_df
    """
    # Filter columns that exist in dataframes
    train_features = [f for f in selected_features if f in train_df.columns]
    test_features = [f for f in selected_features if f in test_df.columns]
    train_targets = [t for t in selected_targets if t in train_df.columns]
    test_targets = [t for t in selected_targets if t in test_df.columns]
    
    X_train = train_df[train_features]
    X_test = test_df[test_features]
    y_train = train_df[train_targets]
    y_test = test_df[test_targets]
    
    train_df_filtered = train_df[train_features + train_targets]
    test_df_filtered = test_df[test_features + test_targets]
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'train_df': train_df_filtered,
        'test_df': test_df_filtered,
        'selected_features': train_features,
        'selected_targets': train_targets
    }
