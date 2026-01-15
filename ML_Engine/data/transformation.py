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


def apply_scaling(data, scaler_type='StandardScaler', return_scaler=False):
    """
    Apply scaling transformation to data.
    
    Parameters
    ----------
    data : array-like or pandas.DataFrame
        Data to scale
    scaler_type : {'StandardScaler', 'MinMaxScaler', 'Normalizer', 'RobustScaler'}
        Type of scaler to use
    return_scaler : bool, default=False
        Whether to return the fitted scaler object
    
    Returns
    -------
    scaled_data : array-like or pandas.DataFrame
        Scaled data
    scaler : object, optional
        Fitted scaler (only if return_scaler=True)
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
    
    if return_scaler:
        return scaled_data, scaler
    else:
        return scaled_data


def apply_label_encoding(df, columns, return_encoders=False):
    """
    Apply label encoding to categorical columns.
    
    Parameters
    ----------
    df : pandas.DataFrame
    columns : list
        Columns to encode
    return_encoders : bool, default=False
        Whether to return the fitted label encoders
    
    Returns
    -------
    encoded_df : pandas.DataFrame
        DataFrame with encoded columns
    encoders : dict, optional
        Dictionary of fitted LabelEncoders (only if return_encoders=True)
    """
    encoded_df = df.copy()
    encoders = {}
    
    for col in columns:
        le = LabelEncoder()
        encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
        encoders[col] = le
    
    if return_encoders:
        return encoded_df, encoders
    else:
        return encoded_df


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
            raise ValueError(
                f"Column '{col}' contains values not in saved mapping: {unknown_values}"
            )
        
        # Apply mapping
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
        ohc = OneHotEncoder(sparse_output=False)
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


def analyze_distribution(data):
    """
    Analyze distribution of data.
    
    Parameters
    ----------
    data : pandas.Series or array-like
    
    Returns
    -------
    dict
        Distribution statistics
    """
    data = pd.Series(data) if not isinstance(data, pd.Series) else data
    
    stats_dict = {
        'mean': data.mean(),
        'std': data.std(),
        'min': data.min(),
        'max': data.max(),
        'skewness': data.skew(),
        'kurtosis': data.kurtosis(),
        'median': data.median(),
        'q1': data.quantile(0.25),
        'q3': data.quantile(0.75),
    }
    
    return stats_dict


def check_distribution_normality(data, test='shapiro', sample_limit=5000):
    """
    Test normality of distribution.
    
    Parameters
    ----------
    data : pandas.Series or array-like
    test : {'shapiro', 'anderson', 'ks'}
        Normality test to use
    sample_limit : int, default=5000
        Maximum sample size for Shapiro-Wilk test
    
    Returns
    -------
    dict
        Test results
    """
    data = pd.Series(data) if not isinstance(data, pd.Series) else data
    data = data.dropna()
    
    if len(data) < 3:
        return {'normal': False, 'p_value': 0, 'statistic': 0, 'message': 'Insufficient data'}
    
    if test == 'shapiro':
        # Limit sample size for Shapiro-Wilk
        sample_size = min(len(data), sample_limit)
        sample_data = data.sample(sample_size) if len(data) > sample_size else data
        statistic, p_value = stats.shapiro(sample_data)
        normal = p_value > 0.05
    
    elif test == 'anderson':
        result = stats.anderson(data)
        statistic = result.statistic
        # Compare with critical values
        normal = statistic < result.critical_values[2]  # 5% significance level
        p_value = None  # Anderson-Darling doesn't provide p-value
    
    elif test == 'ks':
        # Kolmogorov-Smirnov test against normal distribution
        norm_data = (data - data.mean()) / data.std()
        statistic, p_value = stats.kstest(norm_data, 'norm')
        normal = p_value > 0.05
    
    else:
        raise ValueError(f"Unknown normality test: {test}")
    
    return {
        'normal': normal,
        'p_value': p_value,
        'statistic': statistic,
        'test': test
    }


def suggest_transformation(data):
    """
    Suggest transformation based on distribution characteristics.
    
    Parameters
    ----------
    data : pandas.Series or array-like
    
    Returns
    -------
    str
        Suggested transformation
    """
    stats = analyze_distribution(data)
    skewness = stats['skewness']
    
    if abs(skewness) < 0.5:
        return 'None'  # Already fairly symmetric
    elif skewness > 1:
        # Right skewed
        if data.min() > 0:
            return 'Log'
        else:
            return 'Yeo-Johnson'
    elif skewness < -1:
        # Left skewed
        return 'Square'  # Or other transformations for left skew
    else:
        return 'Yeo-Johnson'  # General purpose


def get_available_transformations():
    """
    Get list of available transformations.
    
    Returns
    -------
    list
        Available transformation names
    """
    return [
        'None',
        'Log',
        'Square Root',
        'Square',
        'Cube',
        'Box-Cox',
        'Yeo-Johnson',
        'Reciprocal'
    ]
