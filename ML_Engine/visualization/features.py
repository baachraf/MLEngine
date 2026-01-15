"""
Feature-related Plots
=====================

Functions for visualizing feature characteristics.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional

def plot_variance(
    data: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    title: str = 'Feature Variances',
    xlabel: str = 'Features',
    ylabel: str = 'Variance',
    show_values: bool = True,
    **kwargs
) -> plt.Axes:
    """
    Plot feature variances as a bar chart.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data containing the features.
    ax : plt.Axes, optional
        Matplotlib axes object.
    title : str, default='Feature Variances'
        Plot title.
    xlabel : str, default='Features'
        Label for the x-axis.
    ylabel : str, default='Variance'
        Label for the y-axis.
    show_values : bool, default=True
        Whether to show variance values on bars.
    **kwargs
        Additional keyword arguments passed to sns.barplot.
        
    Returns
    -------
    plt.Axes
        The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()
        
    variances = data.var().sort_values(ascending=False)
    
    sns.barplot(x=variances.index, y=variances.values, ax=ax, **kwargs)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', rotation=45)
    
    if show_values:
        for bar in ax.patches:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f'{bar.get_height():.3f}',
                ha='center',
                va='bottom'
            )
            
    return ax
