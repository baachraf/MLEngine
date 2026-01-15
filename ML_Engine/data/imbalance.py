"""
Imbalance Handling
==================

Functions for handling imbalanced datasets for classification and regression.
"""

import warnings
import pandas as pd
import numpy as np
from sklearn.utils import class_weight


# Try to import imblearn modules
try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    warnings.warn("imblearn not installed. Some imbalance handling functions will not be available.")

# Try to import ImbalancedLearningRegression
try:
    import ImbalancedLearningRegression as iblr
    IBLR_AVAILABLE = True
except ImportError:
    IBLR_AVAILABLE = False
    warnings.warn("ImbalancedLearningRegression not installed. Regression imbalance functions will not be available.")


def handle_classification_imbalance(X, y, method='oversample', technique='RandomOverSampler', random_state=42):
    """
    Handle imbalance for classification problems.
    
    Parameters
    ----------
    X : array-like or pandas.DataFrame
        Feature data
    y : array-like
        Target labels
    method : {'oversample', 'undersample', 'class_weights'}
        Imbalance handling method
    technique : str
        Specific technique to use
    random_state : int, default=42
        Random seed
    
    Returns
    -------
    X_resampled : array-like or pandas.DataFrame
        Resampled features
    y_resampled : array-like
        Resampled labels
    sample_weights : array-like, optional
        Sample weights (only for class_weights method)
    """
    if not IMBLEARN_AVAILABLE and method in ['oversample', 'undersample']:
        raise ImportError("imblearn is required for over/under sampling")
    
    if method == 'oversample':
        if technique == 'RandomOverSampler':
            sampler = RandomOverSampler(random_state=random_state)
        elif technique == 'SMOTE':
            sampler = SMOTE(random_state=random_state)
        elif technique == 'ADASYN':
            sampler = ADASYN(random_state=random_state)
        else:
            raise ValueError(f"Unknown oversampling technique: {technique}")
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        return X_resampled, y_resampled
    
    elif method == 'undersample':
        if technique == 'RandomUnderSampler':
            sampler = RandomUnderSampler(random_state=random_state)
        elif technique == 'TomekLinks':
            sampler = TomekLinks()
        elif technique == 'EditedNearestNeighbours':
            sampler = EditedNearestNeighbours()
        else:
            raise ValueError(f"Unknown undersampling technique: {technique}")
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        return X_resampled, y_resampled
    
    elif method == 'class_weights':
        # Compute class weights
        classes = np.unique(y)
        weights = class_weight.compute_class_weight('balanced', classes=classes, y=y)
        weight_dict = dict(zip(classes, weights))
        sample_weights = np.array([weight_dict[label] for label in y])
        
        return X, y, sample_weights
    
    else:
        raise ValueError(f"Unknown imbalance handling method: {method}")


def handle_regression_imbalance(df, target_column, method='lds', technique=None, **kwargs):
    """
    Handle imbalance for regression problems.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing features and target
    target_column : str
        Name of target column
    method : {'lds', 'fds', 'oversample', 'undersample'}
        Imbalance handling method
    technique : str, optional
        Specific technique for over/under sampling
    **kwargs : dict
        Additional parameters for specific methods
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with imbalance handling applied
    """
    if method in ['lds', 'fds'] and not IBLR_AVAILABLE:
        raise ImportError("ImbalancedLearningRegression is required for LDS/FDS methods")
    
    if method == 'lds':
        # Label Distribution Smoothing
        return iblr.lds(df, y=target_column, **kwargs)
    
    elif method == 'fds':
        # Feature Distribution Smoothing
        return iblr.fds(df, y=target_column, **kwargs)
    
    elif method == 'oversample':
        if technique == 'Random Over-sampling':
            return iblr.random_over_sampling(df, y=target_column, **kwargs)
        elif technique == 'SMOTE':
            return iblr.smote(df, y=target_column, **kwargs)
        elif technique == 'Introduction of Gaussian Noise':
            return iblr.gn(df, y=target_column, **kwargs)
        elif technique == 'ADASYN':
            return iblr.adasyn(df, y=target_column, **kwargs)
        else:
            raise ValueError(f"Unknown regression oversampling technique: {technique}")
    
    elif method == 'undersample':
        if technique == 'Random Under-sampling':
            return iblr.random_under_sampling(df, y=target_column, **kwargs)
        elif technique == 'Condensed Nearest Neighbor':
            return iblr.cnn(df, y=target_column, **kwargs)
        elif technique == 'TomekLinks':
            return iblr.tomeklinks(df, y=target_column, **kwargs)
        elif technique == 'Edited Nearest Neighbor':
            return iblr.enn(df, y=target_column, **kwargs)
        else:
            raise ValueError(f"Unknown regression undersampling technique: {technique}")
    
    else:
        raise ValueError(f"Unknown regression imbalance handling method: {method}")


def get_available_classification_techniques():
    """
    Get available imbalance handling techniques for classification.
    
    Returns
    -------
    dict
        Dictionary mapping methods to available techniques
    """
    techniques = {
        'oversample': ['RandomOverSampler', 'SMOTE', 'ADASYN'],
        'undersample': ['RandomUnderSampler', 'TomekLinks', 'EditedNearestNeighbours'],
        'class_weights': ['balanced']
    }
    return techniques


def get_available_regression_techniques():
    """
    Get available imbalance handling techniques for regression.
    
    Returns
    -------
    dict
        Dictionary mapping methods to available techniques
    """
    techniques = {
        'lds': ['Label Distribution Smoothing'],
        'fds': ['Feature Distribution Smoothing'],
        'oversample': [
            'Random Over-sampling',
            'SMOTE',
            'Introduction of Gaussian Noise',
            'ADASYN'
        ],
        'undersample': [
            'Random Under-sampling',
            'Condensed Nearest Neighbor',
            'TomekLinks',
            'Edited Nearest Neighbor'
        ]
    }
    return techniques


def compute_class_weights(y, method='balanced'):
    """
    Compute class weights for classification.
    
    Parameters
    ----------
    y : array-like
        Target labels
    method : {'balanced', 'custom'}
        Weight computation method
    
    Returns
    -------
    dict
        Dictionary mapping class to weight
    """
    classes = np.unique(y)
    
    if method == 'balanced':
        weights = class_weight.compute_class_weight('balanced', classes=classes, y=y)
    else:
        # Equal weights
        weights = np.ones(len(classes))
    
    return dict(zip(classes, weights))


def check_imbalance_ratio(y, threshold=0.5):
    """
    Check if dataset is imbalanced.
    
    Parameters
    ----------
    y : array-like
        Target labels
    threshold : float, default=0.5
        Threshold for imbalance ratio (minority class proportion)
    
    Returns
    -------
    bool
        True if imbalanced
    float
        Imbalance ratio (minority class proportion)
    """
    class_counts = pd.Series(y).value_counts()
    minority_proportion = class_counts.min() / len(y)
    is_imbalanced = minority_proportion < threshold
    
    return is_imbalanced, minority_proportion


def prepare_lds_weights(y, **kwargs):
    """
    Prepare weights for Label Distribution Smoothing.
    
    Parameters
    ----------
    y : array-like
        Target values
    **kwargs : dict
        Additional parameters for LDS
    
    Returns
    -------
    array-like
        Sample weights
    """
    if not IBLR_AVAILABLE:
        raise ImportError("ImbalancedLearningRegression is required for LDS")
    
    # This is a simplified version
    # In practice, you would use the actual LDS implementation
    y_series = pd.Series(y)
    kernel = kwargs.get('kernel', 'gaussian')
    bandwidth = kwargs.get('bandwidth', 1.0)
    
    # Simple kernel density estimation
    from scipy import stats
    kde = stats.gaussian_kde(y_series, bw_method=bandwidth)
    weights = 1.0 / kde(y_series)
    weights = weights / weights.mean()  # Normalize
    
    return weights
