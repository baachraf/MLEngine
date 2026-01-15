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
    # Replace infinite values with NaN
    df_replaced = df.replace([float('inf'), float('-inf')], float('nan'))
    
    if inplace:
        df.dropna(inplace=True)
        return None
    else:
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
