"""
Distribution Plots
==================

Functions for visualizing the distribution of single variables.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional, Union, Tuple, Any

def plot_histogram(
    data: pd.DataFrame,
    x: str,
    hue: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    kde: bool = True,
    bins: Union[str, int] = 'auto',
    **kwargs
) -> plt.Axes:
    """
    Plot a histogram with optional KDE.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    x : str
        Column name for the x-axis.
    hue : str, optional
        Column name for color encoding.
    ax : plt.Axes, optional
        Matplotlib axes object to draw the plot onto.
    title : str, optional
        Plot title.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    kde : bool, default=True
        Whether to plot a gaussian kernel density estimate.
    bins : str or int, default='auto'
        Specification of hist bins.
    **kwargs
        Additional keyword arguments passed to sns.histplot.
        
    Returns
    -------
    plt.Axes
        The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    sns.histplot(data=data, x=x, hue=hue, kde=kde, bins=bins, ax=ax, **kwargs)
    
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
        
    return ax

def plot_kde(
    data: pd.DataFrame,
    x: str,
    hue: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    fill: bool = True,
    **kwargs
) -> plt.Axes:
    """
    Plot a Kernel Density Estimate (KDE).
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    x : str
        Column name for the x-axis.
    hue : str, optional
        Column name for color encoding.
    ax : plt.Axes, optional
        Matplotlib axes object to draw the plot onto.
    title : str, optional
        Plot title.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    fill : bool, default=True
        Whether to fill the area under the curve.
    **kwargs
        Additional keyword arguments passed to sns.kdeplot.
        
    Returns
    -------
    plt.Axes
        The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()
        
    sns.kdeplot(data=data, x=x, hue=hue, fill=fill, ax=ax, **kwargs)
    
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
        
    return ax

def plot_boxplot(
    data: pd.DataFrame,
    x: str,
    y: Optional[str] = None,
    hue: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    **kwargs
) -> plt.Axes:
    """
    Plot a box plot to show distribution quartiles and outliers.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    x : str
        Column name for the x-axis (can be categorical or numeric).
    y : str, optional
        Column name for the y-axis (usually numeric if x is categorical).
    hue : str, optional
        Column name for color encoding.
    ax : plt.Axes, optional
        Matplotlib axes object to draw the plot onto.
    title : str, optional
        Plot title.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    **kwargs
        Additional keyword arguments passed to sns.boxplot.
        
    Returns
    -------
    plt.Axes
        The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()
        
    sns.boxplot(data=data, x=x, y=y, hue=hue, ax=ax, **kwargs)
    
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
        
    return ax

def plot_qq(
    data: pd.DataFrame,
    x: str,
    dist: str = "norm",
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    **kwargs
) -> plt.Axes:
    """
    Plot a Q-Q plot to compare data distribution against a theoretical distribution.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    x : str
        Column name containing the data to check.
    dist : str, default="norm"
        Theoretical distribution to compare against (e.g., 'norm', 'expon').
    ax : plt.Axes, optional
        Matplotlib axes object to draw the plot onto.
    title : str, optional
        Plot title.
    **kwargs
        Additional keyword arguments passed to stats.probplot.
        
    Returns
    -------
    plt.Axes
        The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()
        
    # Drop NaNs for Q-Q plot
    series = data[x].dropna()
    
    stats.probplot(series, dist=dist, plot=ax, **kwargs)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Q-Q Plot - {x}")
        
    return ax

def plot_distribution_summary(
    data: pd.DataFrame,
    x: str,
    figsize: Tuple[int, int] = (12, 8),
    title: Optional[str] = None
) -> plt.Figure:
    """
    Create a summary figure with Histogram, Box Plot, and Q-Q Plot for a variable.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    x : str
        Column name to analyze.
    figsize : tuple, default=(12, 8)
        Size of the figure.
    title : str, optional
        Main title for the figure.
        
    Returns
    -------
    plt.Figure
        The created figure object.
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2)
    
    ax1 = fig.add_subplot(gs[0, 0]) # Histogram
    ax2 = fig.add_subplot(gs[0, 1]) # KDE
    ax3 = fig.add_subplot(gs[1, 0]) # Box Plot
    ax4 = fig.add_subplot(gs[1, 1]) # Q-Q Plot
    
    plot_histogram(data, x=x, ax=ax1, title="Histogram", kde=False)
    plot_kde(data, x=x, ax=ax2, title="KDE Plot", fill=True)
    plot_boxplot(data, x=x, ax=ax3, title="Box Plot")
    plot_qq(data, x=x, ax=ax4, title="Q-Q Plot")
    
    if title:
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle(f"Distribution Summary: {x}", fontsize=16)
        
    plt.tight_layout()
    return fig
