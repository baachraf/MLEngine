"""
Relationship Plots
==================

Functions for visualizing relationships between variables.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional, Union

def plot_scatter(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    style: Optional[str] = None,
    size: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    **kwargs
) -> plt.Axes:
    """
    Plot a scatter plot.
    
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
    style : str, optional
        Column name for marker style encoding.
    size : str, optional
        Column name for marker size encoding.
    ax : plt.Axes, optional
        Matplotlib axes object.
    title : str, optional
        Plot title.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    **kwargs
        Additional keyword arguments passed to sns.scatterplot.
        
    Returns
    -------
    plt.Axes
        The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()
        
    sns.scatterplot(data=data, x=x, y=y, hue=hue, style=style, size=size, ax=ax, **kwargs)
    
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
        
    return ax

def plot_line(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    style: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    **kwargs
) -> plt.Axes:
    """
    Plot a line plot.
    
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
    style : str, optional
        Column name for line style encoding.
    ax : plt.Axes, optional
        Matplotlib axes object.
    title : str, optional
        Plot title.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    **kwargs
        Additional keyword arguments passed to sns.lineplot.
        
    Returns
    -------
    plt.Axes
        The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()
        
    sns.lineplot(data=data, x=x, y=y, hue=hue, style=style, ax=ax, **kwargs)
    
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
        
    return ax

def plot_reg(
    data: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    **kwargs
) -> plt.Axes:
    """
    Plot a scatter plot with a linear regression line.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    x : str
        Column name for the x-axis.
    y : str
        Column name for the y-axis.
    color : str, optional
        Color for the plot elements.
    ax : plt.Axes, optional
        Matplotlib axes object.
    title : str, optional
        Plot title.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    **kwargs
        Additional keyword arguments passed to sns.regplot.
        
    Returns
    -------
    plt.Axes
        The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()
        
    sns.regplot(data=data, x=x, y=y, color=color, ax=ax, **kwargs)
    
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
        
    return ax
