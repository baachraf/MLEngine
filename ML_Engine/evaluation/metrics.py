"""
Evaluation Metrics
==================

Functions for model evaluation and metrics calculation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Default metric dictionaries
CLASSIFICATION_METRICS = {
    "Accuracy": accuracy_score,
    "Precision": precision_score,
    "Recall": recall_score,
    "F1 Score": f1_score,
}

REGRESSION_METRICS = {
    "Mean Squared Error": mean_squared_error,
    "Mean Absolute Error": mean_absolute_error,
    "R^2 Score": r2_score,
}

def get_metric_result(y_true, y_pred, problem_type,
                      selected_metrics=None, selected_targets=None,
                      classification_metrics=None, regression_metrics=None):
    """
    Compute evaluation metrics for each target.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_targets)
        True target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_targets)
        Predicted target values.
    problem_type : {'Classification', 'Regression'}
        Type of problem.
    selected_metrics : list of str, optional
        Names of metrics to compute. If None, uses all default metrics for the problem type.
    selected_targets : list of str, optional
        Names of target columns. If None and y_true is 1D, defaults to ["target"].
    classification_metrics : dict, optional
        Custom classification metrics.
    regression_metrics : dict, optional
        Custom regression metrics.

    Returns
    -------
    dict
        Nested dictionary {target: {metric_name: metric_value}}.
    """
    if classification_metrics is None:
        classification_metrics = CLASSIFICATION_METRICS
    if regression_metrics is None:
        regression_metrics = REGRESSION_METRICS

    metric_result = {}

    # Convert problem_type to title case for consistent comparison
    problem_type = problem_type.title()

    # Determine default selected_metrics if not provided
    if selected_metrics is None:
        if problem_type == "Classification":
            selected_metrics = list(classification_metrics.keys())
        elif problem_type == "Regression":
            selected_metrics = list(regression_metrics.keys())
        else:
            raise ValueError("problem_type must be 'Classification' or 'Regression'")

    # Determine selected_targets if not provided
    if selected_targets is None:
        if y_true.ndim == 1:
            selected_targets = ["target"]
        else:
            # For multi-output, if names aren't provided, create generic ones
            selected_targets = [f"target_{i}" for i in range(y_true.shape[1])]
    
    # Ensure y_true and y_pred have consistent dimensions for iteration
    # Convert to numpy arrays if they are pandas Series
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    if y_true.shape[1] != len(selected_targets):
        warnings.warn(f"Number of actual targets ({y_true.shape[1]}) does not match "
                      f"length of selected_targets ({len(selected_targets)}). "
                      f"Using generic target names.")
        selected_targets = [f"target_{i}" for i in range(y_true.shape[1])]


    if problem_type == "Classification":
        for i, target in enumerate(selected_targets):
            metric_result[target] = {}
            y_true_single = y_true[:, i]
            y_pred_single = y_pred[:, i]
            num_classes = len(np.unique(y_true_single))

            for metric_name in selected_metrics:
                metric_func = classification_metrics[metric_name]
                if metric_name in ["Precision", "Recall", "F1 Score"] and num_classes > 2:
                    # Handle cases where `average` parameter is needed for multi-class
                    value = metric_func(y_true_single, y_pred_single, average='weighted', zero_division=0)
                else:
                    value = metric_func(y_true_single, y_pred_single)
                metric_result[target][metric_name] = value

    elif problem_type == "Regression":
        for i, target in enumerate(selected_targets):
            metric_result[target] = {}
            y_true_single = y_true[:, i]
            y_pred_single = y_pred[:, i]

            for metric_name in selected_metrics:
                metric_func = regression_metrics[metric_name]
                value = metric_func(y_true_single, y_pred_single)
                metric_result[target][metric_name] = value
    else:
        raise ValueError("problem_type must be 'Classification' or 'Regression'")

    return metric_result

def show_metric_result(metric_result):
    """
    Log metric results.

    Parameters
    ----------
    metric_result : dict
        Result dictionary from get_metric_result.
    """
    for target, metrics in metric_result.items():
        logger.info(f"Model metrics for '{target}':")
        for name, value in metrics.items():
            logger.info(f"  {name}: {value:.4f}")
