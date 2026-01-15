"""
Model Evaluation Plots
======================

Functions for visualizing model performance.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from typing import Optional, List, Union

def plot_confusion_matrix(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    labels: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = 'Confusion Matrix',
    cmap: str = 'Blues',
    **kwargs
) -> plt.Axes:
    """
    Plot a confusion matrix.
    
    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    labels : list of str, optional
        List of label names to use on the x and y axes.
    ax : plt.Axes, optional
        Matplotlib axes object.
    title : str, default='Confusion Matrix'
        Plot title.
    cmap : str, default='Blues'
        Colormap for the heatmap.
    **kwargs
        Additional keyword arguments passed to sns.heatmap.
        
    Returns
    -------
    plt.Axes
        The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()
        
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_pred)))
        
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                xticklabels=labels, yticklabels=labels, ax=ax, **kwargs)
    
    ax.set_title(title)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    return ax

def plot_roc_curve(
    y_true: Union[np.ndarray, pd.Series],
    y_prob: Union[np.ndarray, pd.Series],
    ax: Optional[plt.Axes] = None,
    title: str = 'ROC Curve',
    **kwargs
) -> plt.Axes:
    """
    Plot the Receiver Operating Characteristic (ROC) curve.
    
    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_prob : array-like
        Target scores, can either be probability estimates of the positive class or confidence values.
    ax : plt.Axes, optional
        Matplotlib axes object.
    title : str, default='ROC Curve'
        Plot title.
    **kwargs
        Additional keyword arguments passed to plt.plot.
        
    Returns
    -------
    plt.Axes
        The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()
        
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    
    ax.plot(fpr, tpr, **kwargs)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    
    return ax

def plot_precision_recall_curve(
    y_true: Union[np.ndarray, pd.Series],
    y_prob: Union[np.ndarray, pd.Series],
    ax: Optional[plt.Axes] = None,
    title: str = 'Precision-Recall Curve',
    **kwargs
) -> plt.Axes:
    """
    Plot the Precision-Recall curve.
    
    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_prob : array-like
        Target scores.
    ax : plt.Axes, optional
        Matplotlib axes object.
    title : str, default='Precision-Recall Curve'
        Plot title.
    **kwargs
        Additional keyword arguments passed to plt.plot.
        
    Returns
    -------
    plt.Axes
        The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()
        
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    ax.plot(recall, precision, **kwargs)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    
    return ax
