import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Assuming the following modules exist and have the specified functions
from ML_Engine.visualization.distribution import plot_histogram
from ML_Engine.visualization.categorical import plot_countplot
from ML_Engine.visualization.matrix import plot_heatmap

def plot_dataset_overview(df: pd.DataFrame, target: str = None):
    """
    Automatically generates a grid of histograms for numeric columns and
    countplots for categorical columns to provide a visual overview of the dataset.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target (str, optional): The name of the target variable. If provided,
                                it will be excluded from the overview plots.

    Returns:
        matplotlib.figure.Figure: The figure object containing the plots.
    """
    cols = [col for col in df.columns if col != target]
    num_cols = len(cols)
    grid_size = math.ceil(math.sqrt(num_cols))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 5, grid_size * 5))
    # Ensure axes is always a flat array (handles single subplot case)
    if grid_size == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, col in enumerate(cols):
        if pd.api.types.is_numeric_dtype(df[col]):
            plot_histogram(df, col, ax=axes[i])
        else:
            plot_countplot(df, col, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    return fig

def plot_correlations(df: pd.DataFrame):
    """
    Computes the correlation matrix for the numeric columns of a DataFrame
    and renders it as a heatmap.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        matplotlib.figure.Figure: The figure object containing the heatmap.
    """
    numeric_df = df.select_dtypes(include=['number'])
    corr_matrix = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    plot_heatmap(corr_matrix, ax=ax, title='Correlation Matrix')
    return fig

def plot_pairplot(df: pd.DataFrame, hue: str = None, max_cols: int = 5):
    """
    Generates a pairplot for a subset of numeric columns in the DataFrame.
    Uses seaborn directly for this high-level plot.

    Args:
        df (pd.DataFrame): The input DataFrame.
        hue (str, optional): Variable in df to map plot aspects to different colors.
        max_cols (int, optional): The maximum number of numeric columns to include
                                  in the pairplot to avoid performance issues. Defaults to 5.

    Returns:
        seaborn.axisgrid.PairGrid: The PairGrid object.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > max_cols:
        print(f"Warning: More than {max_cols} numeric columns. Plotting first {max_cols}.")
        numeric_cols = numeric_cols[:max_cols]
        
    pair_grid = sns.pairplot(df, vars=numeric_cols, hue=hue)
    return pair_grid.fig

def plot_target_analysis(df: pd.DataFrame, target: str):
    """
    Provides a visual summary of the target variable, including its distribution,
    class balance (for classification), or outliers (for regression).

    Args:
        df (pd.DataFrame): The input DataFrame.
        target (str): The name of the target variable column.

    Returns:
        matplotlib.figure.Figure: The figure object containing the plot(s).
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Auto-detect problem type
    if pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 20:
        # Regression: plot histogram and a boxplot to show outliers
        plot_histogram(df, target, ax=ax)
        ax.set_title(f'Distribution and Outliers of {target}')
    else:
        # Classification: plot countplot
        plot_countplot(df, target, ax=ax)
        ax.set_title(f'Class Balance of {target}')
        
    plt.tight_layout()
    return fig
