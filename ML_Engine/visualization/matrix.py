"""
Matrix Plots
============

Functions for visualizing matrix-like data, such as correlation matrices.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional

def plot_heatmap(
    data: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    annot: bool = True,
    cmap: str = 'coolwarm',
    **kwargs
) -> plt.Axes:
    """
    Plot a heatmap.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data, typically a correlation matrix.
    ax : plt.Axes, optional
        Matplotlib axes object.
    title : str, optional
        Plot title.
    annot : bool, default=True
        If True, write the data value in each cell.
    cmap : str, default='coolwarm'
        The mapping from data values to color space.
    **kwargs
        Additional keyword arguments passed to sns.heatmap.
        
    Returns
    -------
    plt.Axes
        The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()
        
    sns.heatmap(data, annot=annot, cmap=cmap, ax=ax, **kwargs)
    
    if title:
        ax.set_title(title)
        
    return ax
