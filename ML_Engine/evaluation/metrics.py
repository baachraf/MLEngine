"""
Evaluation Metrics
==================

Functions for model evaluation and metrics calculation.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

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

def get_metric_result(y_true, y_pred, selected_metrics, problem_type,
                      selected_targets, classification_metrics=None,
                      regression_metrics=None):
    """
    Compute evaluation metrics for each target.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_targets)
        True target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_targets)
        Predicted target values.
    selected_metrics : list of str
        Names of metrics to compute.
    problem_type : {'Classification', 'Regression'}
        Type of problem.
    selected_targets : list of str
        Names of target columns.
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

    if problem_type == "Classification":
        for i, target in enumerate(selected_targets):
            metric_result[target] = {}
            y_true_single = y_true[:, i] if y_true.ndim > 1 else y_true
            y_pred_single = y_pred[:, i] if y_pred.ndim > 1 else y_pred
            num_classes = len(np.unique(y_true_single))

            for metric_name in selected_metrics:
                metric_func = classification_metrics[metric_name]
                if metric_name in ["Precision", "Recall", "F1 Score"] and num_classes > 2:
                    value = metric_func(y_true_single, y_pred_single, average='weighted')
                else:
                    value = metric_func(y_true_single, y_pred_single)
                metric_result[target][metric_name] = value

    elif problem_type == "Regression":
        for i, target in enumerate(selected_targets):
            metric_result[target] = {}
            y_true_single = y_true[:, i] if y_true.ndim > 1 else y_true
            y_pred_single = y_pred[:, i] if y_pred.ndim > 1 else y_pred

            for metric_name in selected_metrics:
                metric_func = regression_metrics[metric_name]
                metric_result[target][metric_name] = metric_func(y_true_single, y_pred_single)
    else:
        raise ValueError("problem_type must be 'Classification' or 'Regression'")

    return metric_result

def show_metric_result(metric_result):
    """
    Print metric results to the console.

    Parameters
    ----------
    metric_result : dict
        Result dictionary from get_metric_result.
    """
    for target, metrics in metric_result.items():
        print(f"Model metrics for '{target}':")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")
