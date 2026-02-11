import pandas as pd
from scipy.stats import zscore

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Provides a comprehensive summary of a DataFrame, including data types,
    null counts, null percentages, and unique value counts for each column.

    Args:
        df (pd.DataFrame): The input DataFrame to be summarized.

    Returns:
        pd.DataFrame: A DataFrame containing the summary statistics for each
                      column of the input DataFrame. The summary includes:
                      - 'Dtype': Data type of the column.
                      - 'Null_Count': Total number of null values.
                      - 'Null_Percent': Percentage of null values.
                      - 'Unique_Count': Number of unique values.
    """
    summary_df = pd.DataFrame({
        'Dtype': df.dtypes,
        'Null_Count': df.isnull().sum(),
        'Null_Percent': df.isnull().mean() * 100,
        'Unique_Count': df.nunique()
    })
    return summary_df

def analyze_target(df: pd.DataFrame, target: str) -> dict:
    """
    Analyzes the target variable of a DataFrame to determine its distribution,
    class balance for classification problems, or skewness for regression problems.

    Args:
        df (pd.DataFrame): The input DataFrame containing the target variable.
        target (str): The name of the target variable column.

    Returns:
        dict: A dictionary containing the analysis of the target variable.
              For classification tasks, it includes 'class_balance'.
              For regression tasks, it includes 'skewness'.
              The dictionary also includes 'distribution_summary' which provides
              descriptive statistics of the target variable.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")

    target_series = df[target]
    analysis = {'distribution_summary': target_series.describe().to_dict()}

    # Auto-detect problem type
    if pd.api.types.is_numeric_dtype(target_series) and target_series.nunique() > 20:
        # Regression
        analysis['problem_type'] = 'regression'
        analysis['skewness'] = target_series.skew()
    else:
        # Classification
        analysis['problem_type'] = 'classification'
        analysis['class_balance'] = (target_series.value_counts(normalize=True) * 100).to_dict()

    return analysis

def detect_outliers(df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
    """
    Detects outliers in the numeric columns of a DataFrame using either the
    Interquartile Range (IQR) method or the Z-score method.

    Args:
        df (pd.DataFrame): The input DataFrame.
        method (str, optional): The method to use for outlier detection.
                                Can be 'iqr' or 'zscore'. Defaults to 'iqr'.

    Returns:
        pd.DataFrame: A boolean DataFrame of the same shape as the input,
                      where True indicates an outlier.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    outlier_mask = pd.DataFrame(False, index=df.index, columns=df.columns)

    if method == 'iqr':
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_mask[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
    elif method == 'zscore':
        for col in numeric_cols:
            col_zscores = zscore(df[col].dropna())
            outlier_mask.loc[df[col].notna(), col] = abs(col_zscores) > 3
    else:
        raise ValueError("Method not supported. Choose 'iqr' or 'zscore'.")

    return outlier_mask

def get_cardinality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the cardinality (number of unique values) for each column
    in a DataFrame and flags columns with high cardinality.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with columns 'Cardinality' and 'Is_High_Cardinality'.
    """
    cardinality = df.nunique()
    cardinality_df = pd.DataFrame({
        'Cardinality': cardinality,
        'Is_High_Cardinality': cardinality > 50  # Threshold can be adjusted
    })
    return cardinality_df

def get_skewness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the skewness and kurtosis for all numeric columns in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing the skewness and kurtosis for each
                      numeric column.
    """
    numeric_cols = df.select_dtypes(include=['number'])
    skewness_df = pd.DataFrame({
        'Skewness': numeric_cols.skew(),
        'Kurtosis': numeric_cols.kurtosis()
    })
    return skewness_df
