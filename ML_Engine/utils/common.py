"""
Utilities Module
================

Common utility functions used across the ML Core modules.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Any, Union, List, Dict, Optional, Callable


# ----------------------------------------------------------------------
# Functions imported from original util.utils (stubs/implementations)
# ----------------------------------------------------------------------

def prepare_lds_weights(y: np.ndarray, **kwargs) -> np.ndarray:
    """
    Compute Label Distribution Smoothing (LDS) weights for imbalanced data.
    
    Parameters
    ----------
    y : np.ndarray
        Target labels
    
    Returns
    -------
    np.ndarray
        Sample weights
    """
    # Default implementation: return uniform weights
    warnings.warn("prepare_lds_weights is not fully implemented; returning uniform weights")
    return np.ones_like(y, dtype=float)


class CustomModel:
    """Placeholder for custom model class from original code."""
    pass


def train_main(model, X, y, **kwargs):
    """Placeholder for train_main function."""
    warnings.warn("train_main is not implemented")
    return model


def get_torch_device():
    """Get PyTorch device (CPU/GPU)."""
    try:
        import torch
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    except ImportError:
        warnings.warn("PyTorch not installed")
        return None


def get_class_weights(y: np.ndarray, **kwargs) -> np.ndarray:
    """
    Compute class weights for imbalanced classification.
    
    Parameters
    ----------
    y : np.ndarray
        Class labels
    
    Returns
    -------
    np.ndarray
        Weight for each class
    """
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes, weights))


def can_convert_to_float(value: Any) -> bool:
    """
    Check if a value can be converted to float.
    
    Parameters
    ----------
    value : any
        Value to test
    
    Returns
    -------
    bool
        True if convertible to float
    """
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


# ----------------------------------------------------------------------
# General utility functions
# ----------------------------------------------------------------------

def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None) -> bool:
    """
    Validate dataframe structure.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to validate
    required_columns : list of str, optional
        Columns that must be present
    
    Returns
    -------
    bool
        True if valid
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    return True


def safe_divide(a: Union[float, np.ndarray], b: Union[float, np.ndarray],
                default: float = 0.0) -> Union[float, np.ndarray]:
    """
    Safe division with zero denominator handling.
    
    Parameters
    ----------
    a : numerator
    b : denominator
    default : value to return when denominator is zero
    
    Returns
    -------
    a / b or default
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(a, b)
        if isinstance(result, np.ndarray):
            result = np.where(np.isfinite(result), result, default)
        else:
            result = result if np.isfinite(result) else default
    return result


def dict_to_json_serializable(d: Dict) -> Dict:
    """
    Convert dictionary values to JSON serializable types.
    
    Parameters
    ----------
    d : dict
    
    Returns
    -------
    dict
        JSON serializable dictionary
    """
    import json
    def convert(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        else:
            return obj
    
    return json.loads(json.dumps(d, default=convert))


def log_message(message: str, level: str = 'INFO'):
    """
    Simple logging utility.
    
    Parameters
    ----------
    message : str
        Message to log
    level : {'INFO', 'WARNING', 'ERROR'}
        Log level
    """
    print(f"[{level}] {message}")


def get_default_preprocessing_json():
    """
    Return default preprocessing JSON configuration dictionary.
    
    Returns
    -------
    dict
        Default preprocessing configuration
    """
    return {
        "undersampling_regression_used": None,
        "oversampling_regression_used": None,
        "second_imbalance_technique_regression": None,
        "use_second_imbalance_technique_regression": False,
        "transformation_types_column": {},
        "transformation_columns_Yeo_Johnson": [],
        "use_filtered_df": False,
        "filter_selected_column_type": None,
        "filter_selected_column": None,
        "filter_value": None,
        "filter_operation": None,
        "use_group_shuffle": False,
        "group_shuffle_column": None,
        "column_droped": [],
        "drop_inf": False,
        "custom_split": False,
        "selected_column": None,
        "drop_null": False,
        "drop_dupplicated": False,
        "label_encoded_columns": [],
        "one_hot_encoded_columns": [],
        "select_all_features": False,
        "prefix_features": False,
        "select_all_features_tr": False,
        "prefix_features_tr": False,
        "selected_family": [],
        "update_each_selected_family": False,
        "choose_selected_family_update": [],
        "family_updated": {},
        "choose_DSType": False,
        "selected_dstype_value": [],
        "org_selected_features": [],
        "selected_features": [],
        "selected_targets": [],
        "normalization_technique": 'None',
        "n_components": 1,
        "apply_pca": False,
        "pca_graph": False,
        "use_pca_variance": False,
        "variance": 0.9,
        "number_features_pca": 1,
        "test_size": 0.2,
        "problem_type": "Classification",
        "imbalance_technique_classification": "Over-Sampling",
        "imbalance": False,
        "oversample": "RandomOverSampler",
        "undersample": "RandomUnderSampler",
        "imbalance_technique_regression": []
    }


def init_preprocessing_config(updates=None):
    """
    Initialize preprocessing configuration with optional updates.
    
    Parameters
    ----------
    updates : dict, optional
        Key-value pairs to update in default configuration
    
    Returns
    -------
    dict
        Configuration dictionary
    """
    config = get_default_preprocessing_json()
    if updates:
        config.update(updates)
    return config
