import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

def extract_date_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Extracts date-related features (year, month, day, weekday, hour, quarter)
    from a datetime column and adds them to the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col (str): The name of the datetime column.

    Returns:
        pd.DataFrame: A new DataFrame with the extracted date features added.
    """
    df_new = df.copy()
    df_new[col] = pd.to_datetime(df_new[col])
    
    df_new[f'{col}_year'] = df_new[col].dt.year
    df_new[f'{col}_month'] = df_new[col].dt.month
    df_new[f'{col}_day'] = df_new[col].dt.day
    df_new[f'{col}_weekday'] = df_new[col].dt.weekday
    df_new[f'{col}_hour'] = df_new[col].dt.hour
    df_new[f'{col}_quarter'] = df_new[col].dt.quarter
    
    return df_new

def create_lag_features(df: pd.DataFrame, col: str, lags: list) -> pd.DataFrame:
    """
    Creates lag features for a specified column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col (str): The name of the column to create lags for.
        lags (list): A list of integers representing the lag periods.

    Returns:
        pd.DataFrame: A new DataFrame with the lag features added.
    """
    df_new = df.copy()
    for lag in lags:
        df_new[f'{col}_lag_{lag}'] = df_new[col].shift(lag)
    return df_new

def create_rolling_features(df: pd.DataFrame, col: str, windows: list, agg: list = ['mean', 'std']) -> pd.DataFrame:
    """
    Creates rolling window features for a specified column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col (str): The name of the column to create rolling features for.
        windows (list): A list of integers representing the window sizes.
        agg (list, optional): A list of aggregation functions to apply.
                              Defaults to ['mean', 'std'].

    Returns:
        pd.DataFrame: A new DataFrame with the rolling features added.
    """
    df_new = df.copy()
    for window in windows:
        for func in agg:
            df_new[f'{col}_rolling_{window}_{func}'] = df_new[col].rolling(window=window).agg(func)
    return df_new

def check_stationarity(series: pd.Series):
    """
    Performs the Augmented Dickey-Fuller (ADF) test to check for stationarity.

    Args:
        series (pd.Series): The time series data.

    Returns:
        tuple: A tuple containing the p-value, a boolean indicating if the series
               is stationary (p-value < 0.05), and a dictionary with the test results.

    Raises:
        ImportError: If statsmodels is not installed.
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        raise ImportError("statsmodels is required for check_stationarity. Please install it using 'pip install statsmodels'.")

    result = adfuller(series.dropna())
    p_value = result[1]
    is_stationary = p_value < 0.05
    interpretation = {
        'ADF Statistic': result[0],
        'p-value': p_value,
        'Critical Values': result[4]
    }
    
    return p_value, is_stationary, interpretation

def split_timeseries(df: pd.DataFrame, date_col: str, test_size: float = 0.2):
    """
    Splits a time series DataFrame into training and testing sets, preserving
    the temporal order.

    Args:
        df (pd.DataFrame): The input DataFrame.
        date_col (str): The name of the date column used for sorting.
        test_size (float, optional): The proportion of the dataset to include in the test split.
                                     Defaults to 0.2.

    Returns:
        tuple: A tuple containing (X_train, X_test, y_train, y_test).
               Note: This function currently returns the split DataFrames directly,
               assuming the target is part of the DataFrame or handled separately.
               Adjust return values as needed based on specific usage patterns.
    """
    # Ensure data is sorted by date
    df_sorted = df.sort_values(by=date_col)
    
    split_index = int(len(df_sorted) * (1 - test_size))
    
    train = df_sorted.iloc[:split_index]
    test = df_sorted.iloc[split_index:]
    
    # Assuming the last column is the target for simplicity, or return full DFs
    # Adjusting to match the requested signature (X_train, X_test, y_train, y_test)
    # This assumes the target is the last column. If not, this logic needs refinement.
    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]
    X_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1]
    
    return X_train, X_test, y_train, y_test
