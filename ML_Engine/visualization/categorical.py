"""
Categorical Plots
=================

Functions for visualizing categorical data.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional

def plot_barplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    **kwargs
) -> plt.Axes:
    """
    Plot a bar plot.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    x : str
        Column name for the x-axis.
    y : str
        Column name for the y-axis.
    hue : str, optional
        Column name for color encoding.
    ax : plt.Axes, optional
        Matplotlib axes object.
    title : str, optional
        Plot title.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    **kwargs
        Additional keyword arguments passed to sns.barplot.
        
    Returns
    -------
    plt.Axes
        The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()
        
    sns.barplot(data=data, x=x, y=y, hue=hue, ax=ax, **kwargs)
    
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
        
    return ax

def plot_countplot(
    data: pd.DataFrame,
    x: str,
    hue: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    **kwargs
) -> plt.Axes:
    """
    Plot a count plot showing the counts of observations in each categorical bin.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    x : str
        Column name for the x-axis.
    hue : str, optional
        Column name for color encoding.
    ax : plt.Axes, optional
        Matplotlib axes object.
    title : str, optional
        Plot title.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    **kwargs
        Additional keyword arguments passed to sns.countplot.
        
    Returns
    -------
    plt.Axes
        The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()
        
    sns.countplot(data=data, x=x, hue=hue, ax=ax, **kwargs)
    
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
        
    return ax
