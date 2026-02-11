"""
Data Cleaning Operations
========================

Functions for cleaning data: handling missing values, duplicates, infinite values.
"""

import pandas as pd
import numpy as np


def drop_null_values(df, inplace=False):
    """
    Drop rows with null values.
    
    Parameters
    ----------
    df : pandas.DataFrame
    inplace : bool, default=False
        Whether to modify the DataFrame in place
    
    Returns
    -------
    pandas.DataFrame or None
        If inplace=False, returns cleaned DataFrame
    """
    if inplace:
        df.dropna(inplace=True)
        return None
    else:
        return df.dropna()


def drop_duplicate_rows(df, inplace=False):
    """
    Drop duplicate rows.
    
    Parameters
    ----------
    df : pandas.DataFrame
    inplace : bool, default=False
        Whether to modify the DataFrame in place
    
    Returns
    -------
    pandas.DataFrame or None
        If inplace=False, returns cleaned DataFrame
    """
    if inplace:
        df.drop_duplicates(inplace=True)
        return None
    else:
        return df.drop_duplicates()


def drop_infinite_values(df, inplace=False):
    """
    Drop rows containing infinite values.
    
    Parameters
    ----------
    df : pandas.DataFrame
    inplace : bool, default=False
        Whether to modify the DataFrame in place
    
    Returns
    -------
    pandas.DataFrame or None
        If inplace=False, returns cleaned DataFrame
    """
    if inplace:
        df.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
        df.dropna(inplace=True)
        return None
    else:
        df_replaced = df.replace([float('inf'), float('-inf')], float('nan'))
        return df_replaced.dropna()


def drop_columns(df, columns_to_drop, inplace=False):
    """
    Drop specified columns.
    
    Parameters
    ----------
    df : pandas.DataFrame
    columns_to_drop : str or list
        Column(s) to drop
    inplace : bool, default=False
        Whether to modify the DataFrame in place
    
    Returns
    -------
    pandas.DataFrame or None
        If inplace=False, returns cleaned DataFrame
    """
    if isinstance(columns_to_drop, str):
        columns_to_drop = [columns_to_drop]
    
    if inplace:
        df.drop(columns=columns_to_drop, inplace=True)
        return None
    else:
        return df.drop(columns=columns_to_drop)


def remove_outliers(df, columns=None, factor=1.5, inplace=False):
    """
    Remove outliers using the Interquartile Range (IQR) method.
    
    Parameters
    ----------
    df : pandas.DataFrame
    columns : list, optional
        Columns to check for outliers. If None, all numeric columns are used.
    factor : float, default=1.5
        The IQR factor to use for determining outliers (typically 1.5 or 3.0).
    inplace : bool, default=False
        Whether to modify the DataFrame in place.
    
    Returns
    -------
    pandas.DataFrame or None
        If inplace=False, returns DataFrame with outliers removed.
    """
    if not inplace:
        df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            # Filter rows
            df.drop(df[(df[col] < lower_bound) | (df[col] > upper_bound)].index, inplace=True)
            
    if not inplace:
        return df


def count_null_values(df):
    """
    Count null values per column.
    
    Parameters
    ----------
    df : pandas.DataFrame
    
    Returns
    -------
    pandas.Series
        Count of null values for columns that have nulls
    """
    null_counts = df.isnull().sum()
    return null_counts[null_counts > 0]


def count_infinite_values(df):
    """
    Count infinite values per column.
    
    Parameters
    ----------
    df : pandas.DataFrame
    
    Returns
    -------
    pandas.Series
        Count of infinite values for columns that have infinities
    """
    inf_counts = (df == float('inf')).sum() + (df == float('-inf')).sum()
    return inf_counts[inf_counts > 0]


def count_duplicate_rows(df):
    """
    Count duplicate rows.
    
    Parameters
    ----------
    df : pandas.DataFrame
    
    Returns
    -------
    int
        Number of duplicate rows
    """
    return df.duplicated().sum()


def clean_data(df, drop_null=True, drop_duplicates=True, drop_inf=True, 
               remove_outlier=False, outlier_columns=None, outlier_factor=1.5,
               columns_to_drop=None, inplace=False):
    """
    Perform multiple cleaning operations.
    
    Parameters
    ----------
    df : pandas.DataFrame
    drop_null : bool, default=True
        Drop rows with null values
    drop_duplicates : bool, default=True
        Drop duplicate rows
    drop_inf : bool, default=True
        Drop rows with infinite values
    remove_outlier : bool, default=False
        Whether to remove outliers
    outlier_columns : list, optional
        Columns to check for outliers
    outlier_factor : float, default=1.5
        IQR factor for outliers
    columns_to_drop : list, optional
        Columns to drop
    inplace : bool, default=False
        Whether to modify the DataFrame in place
    
    Returns
    -------
    pandas.DataFrame or None
        Cleaned DataFrame if inplace=False, else None
    """
    if not inplace:
        df = df.copy()
    
    if drop_null:
        drop_null_values(df, inplace=True)
    
    if drop_duplicates:
        drop_duplicate_rows(df, inplace=True)
    
    if drop_inf:
        drop_infinite_values(df, inplace=True)
    
    if remove_outlier:
        remove_outliers(df, columns=outlier_columns, factor=outlier_factor, inplace=True)
    
    if columns_to_drop:
        drop_columns(df, columns_to_drop, inplace=True)
    
    if not inplace:
        return df


def get_data_description(df):
    """
    Get descriptive statistics of the data.
    
    Parameters
    ----------
    df : pandas.DataFrame
    
    Returns
    -------
    pandas.DataFrame
        Descriptive statistics
    """
    return df.describe()
