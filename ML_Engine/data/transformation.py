"""
Data Transformation and Encoding
=================================

Functions for scaling, encoding, and transforming data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, Normalizer, RobustScaler,
    LabelEncoder, OneHotEncoder, PowerTransformer
)
from scipy import stats
from typing import List, Tuple, Any

def apply_feature_selection(df: pd.DataFrame, selected_features: List[str]) -> pd.DataFrame:
    """
    Filters a DataFrame to include only the selected features.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    selected_features : list of str
        A list of column names to keep.

    Returns
    -------
    pd.DataFrame
        A new DataFrame containing only the selected feature columns.
    """
    return df[selected_features].copy()

def fit_and_apply_scaling(data: pd.DataFrame, scaler_type: str = 'StandardScaler') -> Tuple[pd.DataFrame, Any]:
    """
    Fit a scaler and apply it to the data.

    Parameters
    ----------
    data : pd.DataFrame
        Data to scale.
    scaler_type : {'StandardScaler', 'MinMaxScaler', 'Normalizer', 'RobustScaler'}
        Type of scaler to use.

    Returns
    -------
    tuple
        A tuple containing the scaled data (as a DataFrame) and the fitted scaler object.
    """
    if scaler_type == "StandardScaler":
        scaler = StandardScaler()
    elif scaler_type == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif scaler_type == "Normalizer":
        scaler = Normalizer()
    elif scaler_type == "RobustScaler":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    scaled_data = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(scaled_data, index=data.index, columns=data.columns)
    
    return scaled_df, scaler

def apply_scaler(data: pd.DataFrame, scaler: Any) -> pd.DataFrame:
    """
    Apply a pre-fitted scaler to new data.

    Parameters
    ----------
    data : pd.DataFrame
        Data to transform.
    scaler : Any
        A pre-fitted scikit-learn scaler object.

    Returns
    -------
    pd.DataFrame
        The transformed data as a DataFrame.
    """
    transformed_data = scaler.transform(data)
    return pd.DataFrame(transformed_data, index=data.index, columns=data.columns)

def apply_label_encoding(df, columns):
    """
    Apply label encoding to categorical columns and return the mappings.
    
    Parameters
    ----------
    df : pandas.DataFrame
    columns : list
        Columns to encode
    
    Returns
    -------
    encoded_df : pandas.DataFrame
        DataFrame with encoded columns
    mappings : dict
        A dictionary where keys are column names and values are another
        dictionary mapping original labels to their encoded integer values.
        e.g., {'col_name': {'cat_A': 0, 'cat_B': 1}}
    """
    encoded_df = df.copy()
    mappings = {}
    
    for col in columns:
        le = LabelEncoder()
        encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
        
        # Create a mapping from original class names to encoded values
        mappings[col] = {cls: int(val) for cls, val in zip(le.classes_, le.transform(le.classes_))}
        
    return encoded_df, mappings


def apply_saved_label_encoding(df, columns, saved_mappings):
    """
    Apply saved label encoding mappings to categorical columns.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with categorical columns to encode
    columns : list
        Columns to encode
    saved_mappings : dict
        Dictionary mapping column names to label mappings.
        Each mapping should be a dict of {original_value: encoded_value}.
    
    Returns
    -------
    encoded_df : pandas.DataFrame
        DataFrame with encoded columns
    
    Raises
    ------
    ValueError
        If a column is not in the dataframe or if a value is not in the saved mapping
    """
    encoded_df = df.copy()
    
    for col in columns:
        if col not in encoded_df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")
        
        if col not in saved_mappings:
            raise ValueError(f"No saved mapping for column '{col}'")
        
        mapping = saved_mappings[col]
        
        # Convert column to string for mapping (as original mapping likely based on strings)
        col_values = encoded_df[col].astype(str)
        
        # Check for unknown values
        unique_values = set(col_values.unique())
        known_values = set(mapping.keys())
        unknown_values = unique_values - known_values
        
        if unknown_values:
            # Warn about unknown values and assign -1
            import warnings
            warnings.warn(
                f"Column '{col}' contains values not in saved mapping: {unknown_values}. "
                f"Assigning -1 to unknown values."
            )
            # Create mapping with default value -1 for unknown values
            def map_with_unknown(val):
                return mapping.get(val, -1)
            encoded_df[col] = col_values.map(map_with_unknown)
        else:
            # Apply mapping normally
            encoded_df[col] = col_values.map(mapping)
    
    return encoded_df


def apply_one_hot_encoding(df, columns, return_encoder=False):
    """
    Apply one-hot encoding to categorical columns.
    
    Parameters
    ----------
    df : pandas.DataFrame
    columns : list
        Columns to encode
    return_encoder : bool, default=False
        Whether to return the fitted OneHotEncoder
    
    Returns
    -------
    encoded_df : pandas.DataFrame
        DataFrame with one-hot encoded columns
    encoder : OneHotEncoder, optional
        Fitted OneHotEncoder (only if return_encoder=True)
    """
    encoded_df = df.copy()
    
    for col in columns:
        ohc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        one_hot_encoded = ohc.fit_transform(encoded_df[[col]])
        
        # Create column names
        ohc_df = pd.DataFrame(
            one_hot_encoded,
            columns=[f"{col}_{category}" for category in ohc.categories_[0]],
            index=encoded_df.index
        )
        
        # Drop original column and concatenate one-hot encoded columns
        encoded_df = encoded_df.drop(col, axis=1)
        encoded_df = pd.concat([encoded_df, ohc_df], axis=1)
    
    if return_encoder:
        return encoded_df, ohc
    else:
        return encoded_df


def encode_categorical(df, columns, method='label', **kwargs):
    """
    Encode categorical columns using specified method.
    
    Parameters
    ----------
    df : pandas.DataFrame
    columns : list
        Columns to encode
    method : {'label', 'onehot'}
        Encoding method
    **kwargs : dict
        Additional arguments passed to encoding function
    
    Returns
    -------
    encoded_df : pandas.DataFrame
        Encoded DataFrame
    """
    if method == 'label':
        return apply_label_encoding(df, columns, **kwargs)
    elif method == 'onehot':
        return apply_one_hot_encoding(df, columns, **kwargs)
    else:
        raise ValueError(f"Unknown encoding method: {method}")
