"""
Data Input/Output and Basic Operations
=======================================

Functions for loading, saving, splitting data, and applying transformations.
"""

import json
import os
import warnings
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import PowerTransformer, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, GroupShuffleSplit


def apply_transformation(data, transformation, col_name=None, mode="train", params=None, multiple=False):
    """
    Apply numeric transformation to data.
    
    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
        Data to transform
    transformation : str
        Transformation type: 'Log', 'Square Root', 'Square', 'Cube', 
        'Box-Cox', 'Yeo-Johnson', 'Reciprocal'
    col_name : str, optional
        Column name (for error messages)
    mode : {'train', 'test'}
        Whether in training mode (fit parameters) or test mode (use saved params)
    params : dict, optional
        Saved transformation parameters for test mode
    multiple : bool, default=False
        Whether data is multiple columns (DataFrame) or single column (Series)
    
    Returns
    -------
    transformed_data : pandas.Series or pandas.DataFrame
        Transformed data
    transformation_params : dict, optional
        Only returned in train mode, contains fitted parameters
    """
    transformed_data = data.copy()
    transformation_params = {}
    
    # Apply main transformation
    if transformation == "Log":
        if any(transformed_data <= 0):
            transformed_data = np.log1p(transformed_data - transformed_data.min() + 1)
        else:
            transformed_data = np.log1p(transformed_data)
    
    elif transformation == "Square Root":
        if any(transformed_data < 0):
            raise ValueError("Cannot apply Square Root transformation to negative values")
        else:
            transformed_data = np.sqrt(transformed_data)
    
    elif transformation == "Square":
        transformed_data = np.square(transformed_data)
    
    elif transformation == "Cube":
        transformed_data = np.power(transformed_data, 3)
    
    elif transformation == "Box-Cox":
        if mode == "train":
            if any(transformed_data <= 0):
                raise ValueError("Cannot apply Box-Cox transformation to non-positive values")
            else:
                transformed_data, lambda_param = stats.boxcox(transformed_data)
            transformation_params["boxcox_lambda"] = lambda_param
        elif mode == "test":
            lambda_param = params.get("boxcox_lambda")
            if lambda_param is not None:
                if any(transformed_data <= 0):
                    shifted_data = transformed_data - transformed_data.min() + 1
                    transformed_data = stats.boxcox(shifted_data, lmbda=lambda_param)
                else:
                    transformed_data = stats.boxcox(transformed_data, lmbda=lambda_param)
    
    elif transformation == "Yeo-Johnson":
        if multiple:
            if mode == "train":
                # Use PowerTransformer for multiple columns
                transformer = PowerTransformer(method='yeo-johnson', standardize=False)
                transformed_data = pd.DataFrame(
                    transformer.fit_transform(transformed_data),
                    columns=transformed_data.columns,
                    index=transformed_data.index
                )
                transformation_params["yeo_johnson_transformer"] = transformer
            elif mode == "test":
                transformer = params.get("yeo_johnson_transformer")
                if transformer is not None:
                    transformed_data = pd.DataFrame(
                        transformer.transform(transformed_data),
                        columns=transformed_data.columns,
                        index=transformed_data.index
                    )
        else:
            if mode == "train":
                try:
                    transformed_data, lambda_param = stats.yeojohnson(transformed_data)
                    transformation_params["yeo_johnson_lambda"] = lambda_param
                except Exception as e:
                    raise ValueError(f"Failed to apply Yeo-Johnson transformation on {col_name}: {e}")
            elif mode == "test":
                lambda_param = params.get("yeo_johnson_lambda")
                if lambda_param is not None:
                    transformed_data = stats.yeojohnson(transformed_data, lmbda=lambda_param)
    
    elif transformation == "Reciprocal":
        if any(transformed_data == 0):
            transformed_data = 1 / (transformed_data + 1)
        else:
            transformed_data = 1 / transformed_data
    
    if mode == "train":
        return transformed_data, transformation_params
    elif mode == "test":
        return transformed_data


def load_json(file_path):
    """Load JSON file."""
    with open(file_path, "r") as json_file:
        return json.load(json_file)


def save_json(data, file_path):
    """Save data to JSON file."""
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=2)


def load_data(file_path, file_type=None):
    """
    Load data from CSV or JSON file.
    
    Parameters
    ----------
    file_path : str
        Path to data file
    file_type : {'csv', 'json', 'auto'}, optional
        File type. If None, inferred from extension.
    
    Returns
    -------
    pandas.DataFrame
    """
    if file_type is None:
        _, ext = os.path.splitext(file_path)
        file_type = ext.lower().lstrip('.')
    
    if file_type == 'csv':
        return pd.read_csv(file_path, index_col=0)
    elif file_type == 'json':
        return pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def drop_inf_values(df):
    """
    Drop rows containing infinite values.
    
    Parameters
    ----------
    df : pandas.DataFrame
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with infinite values removed
    """
    # Replace infinite values with NaN and drop them
    df_replaced = df.replace([float('inf'), float('-inf')], float('nan'))
    return df_replaced.dropna()


def custom_split_fn(df, split_column, test_size=0.2, random_state=42):
    """
    Performs a stratified split on the dataframe based on the specified column.

    This ensures that the distribution of values in the split_column is
    preserved in both the training and testing sets.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to split.
    split_column : str
        The name of the column to stratify by.
    test_size : float, default=0.2
        The proportion of the dataset to allocate to the test split.
    random_state : int, default=42
        Seed for the random number generator for reproducibility.

    Returns
    -------
    train_df : pandas.DataFrame
        The training set dataframe.
    test_df : pandas.DataFrame
        The testing set dataframe.
    """
    stratify_col = df[split_column]
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col
    )
    return train_df, test_df


def split_data_train_test(df, selected_features, selected_targets, 
                          test_size=0.2, split_method='random', 
                          split_column=None, group_column=None, random_state=42):
    """
    Split data into training and testing sets.
    
    Parameters
    ----------
    df : pandas.DataFrame
    selected_features : list
        Feature column names
    selected_targets : list
        Target column names
    test_size : float, default=0.2
        Proportion for test set
    split_method : {'random', 'custom', 'group_shuffle'}
        Splitting method
    split_column : str, optional
        Column for custom split (required for 'custom' method)
    group_column : str, optional
        Column for group shuffle split (required for 'group_shuffle' method)
    random_state : int, default=42
        Seed for random number generator
    
    Returns
    -------
    X_train : pandas.DataFrame
    X_test : pandas.DataFrame
    y_train : pandas.DataFrame
    y_test : pandas.DataFrame
    """
    if split_method == 'custom' and split_column is not None:
        train_df, test_df = custom_split_fn(df, split_column, test_size, random_state)
        X_train = train_df[selected_features]
        X_test = test_df[selected_features]
        y_train = train_df[selected_targets]
        y_test = test_df[selected_targets]
    
    elif split_method == 'group_shuffle' and group_column is not None:
        X = df[selected_features]
        y = df[selected_targets]
        gss = GroupShuffleSplit(n_splits=1, train_size=1-test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups=df[group_column]))
        
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
    
    else:  # random split
        X = df[selected_features]
        y = df[selected_targets]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    
    return X_train, X_test, y_train, y_test


def make_creating_validations_possible(X_train, X_test, y_train, y_test, 
                                       selected_features=None):
    """
    Create DataFrames from train/test splits for validation.
    
    Parameters
    ----------
    X_train, X_test : array-like or pandas.DataFrame
    y_train, y_test : array-like or pandas.DataFrame
    selected_features : list, optional
        Feature names for X data
    
    Returns
    -------
    train_df : pandas.DataFrame
    test_df : pandas.DataFrame
    """
    # Convert to DataFrames if needed
    if not isinstance(X_train, pd.DataFrame):
        if selected_features is None:
            selected_features = [f"feature_{i}" for i in range(X_train.shape[1])]
        X_train_df = pd.DataFrame(X_train, columns=selected_features)
        X_test_df = pd.DataFrame(X_test, columns=selected_features)
    else:
        X_train_df = X_train.copy()
        X_test_df = X_test.copy()
    
    if not isinstance(y_train, pd.DataFrame):
        y_train_df = pd.DataFrame(y_train)
        y_test_df = pd.DataFrame(y_test)
    else:
        y_train_df = y_train.copy()
        y_test_df = y_test.copy()
    
    # Reset indices
    X_train_df = X_train_df.reset_index(drop=True)
    y_train_df = y_train_df.reset_index(drop=True)
    X_test_df = X_test_df.reset_index(drop=True)
    y_test_df = y_test_df.reset_index(drop=True)
    
    # Combine X and y
    train_df = pd.concat([X_train_df, y_train_df], axis=1)
    test_df = pd.concat([X_test_df, y_test_df], axis=1)
    
    return train_df, test_df


def generate_transformation_result(original_df, transformed_df, 
                                  transformed_columns, transformations=None):
    """
    Generate transformation summary statistics.
    
    Parameters
    ----------
    original_df : pandas.DataFrame
        Original data
    transformed_df : pandas.DataFrame
        Transformed data
    transformed_columns : list
        Columns that were transformed
    transformations : dict, optional
        Mapping of column to transformation type
    
    Returns
    -------
    pandas.DataFrame
        Summary statistics
    """
    summary_data = []
    
    for col in transformed_columns:
        if transformations is not None:
            transformation = transformations.get(col, "Unknown")
        else:
            transformation = "Unknown"
        
        orig_data = original_df[col]
        new_data = transformed_df[col]
        
        # Calculate skewness
        orig_skew = stats.skew(orig_data)
        new_skew = stats.skew(new_data)
        
        # Test normality (Shapiro-Wilk, sample limited to 5000)
        sample_size = min(len(orig_data), 5000)
        if sample_size >= 3:
            _, orig_p = stats.shapiro(orig_data.sample(sample_size))
            _, new_p = stats.shapiro(new_data.sample(sample_size))
            orig_normal = orig_p > 0.05
            new_normal = new_p > 0.05
        else:
            orig_normal = new_normal = False
        
        summary_data.append({
            "Column": col,
            "Transformation": transformation,
            "Scaling": "No Scaling",
            "Original Skewness": orig_skew,
            "New Skewness": new_skew,
            "Original Normality": orig_normal,
            "New Normality": new_normal,
            "remove": False
        })
    
    return pd.DataFrame(summary_data)


def can_convert_to_float(value):
    """
    Check if a value can be converted to float.
    
    Parameters
    ----------
    value : any
    
    Returns
    -------
    bool
    """
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def apply_saved_filter(data, df_columns, df):
    """
    Apply saved filter configuration to a dataframe.
    
    Parameters
    ----------
    data : dict
        Filter configuration dictionary with keys:
        - use_filtered_df: bool
        - filter_selected_column_type: str ('number' or other)
        - filter_operation: str ('greater than', 'less than', 'equal to')
        - filter_selected_column: str
        - filter_value: numeric or str
    df_columns : list
        List of column names in the dataframe
    df : pandas.DataFrame
        Input dataframe
    
    Returns
    -------
    filtered_df : pandas.DataFrame
        Filtered dataframe (or original if filtering not applied)
    filter_applied : bool
        Whether filtering was successfully applied
    message : str
        Status message
    """
    if not data.get("use_filtered_df", False):
        return df.copy(), False, "Filter not applied (use_filtered_df=False)"
    
    if data.get("filter_selected_column_type") == "number":
        filter_operation = data.get("filter_operation", "equal to")
        selected_column = data.get("filter_selected_column", "")
        filter_value = data.get("filter_value", None)
        
        if selected_column not in df_columns:
            return df.copy(), False, f"Filter column '{selected_column}' not found in dataframe"
        
        try:
            filter_value = float(filter_value)
        except (ValueError, TypeError):
            return df.copy(), False, f"Filter value '{filter_value}' is not numeric"
        
        # Apply numeric filter
        if filter_operation == "greater than":
            filtered_df = df[df[selected_column] > filter_value]
        elif filter_operation == "less than":
            filtered_df = df[df[selected_column] < filter_value]
        else:  # equal to
            filtered_df = df[df[selected_column] == filter_value]
        
        return filtered_df, True, f"Applied {filter_operation} filter on {selected_column}"
    
    else:
        # Categorical filter
        selected_column = data.get("filter_selected_column", "")
        filter_value = data.get("filter_value", "")
        
        if selected_column not in df_columns:
            return df.copy(), False, f"Filter column '{selected_column}' not found in dataframe"
        
        if filter_value not in list(df[selected_column].unique()):
            return df.copy(), False, f"Filter value '{filter_value}' not found in column '{selected_column}'"
        
        filtered_df = df[df[selected_column] == filter_value]
        return filtered_df, True, f"Applied categorical filter on {selected_column} = {filter_value}"


def execute_loaded_prep_config(data, df):
    """
    Execute loaded preprocessing configuration on a dataframe.
    
    Parameters
    ----------
    data : dict
        Preprocessing configuration dictionary
    df : pandas.DataFrame
        Input dataframe
    
    Returns
    -------
    processed_df : pandas.DataFrame
        Processed dataframe
    train_df : pandas.DataFrame or None
        Training split (if splitting configured)
    test_df : pandas.DataFrame or None
        Test split (if splitting configured)
    messages : list of str
        Status messages
    """
    messages = []
    df_processed = df.copy()
    df_columns = list(df_processed.columns)
    
    # 1. Drop columns
    columns_to_drop = data.get("column_droped", [])
    for column in columns_to_drop:
        if column in df_columns:
            df_processed = df_processed.drop(columns=[column])
            messages.append(f"Column '{column}' dropped")
            df_columns.remove(column)
        else:
            messages.append(f"Warning: Column '{column}' not found for dropping")
    
    # 2. Handle null values
    if data.get("drop_null", False):
        initial_rows = len(df_processed)
        df_processed = df_processed.dropna()
        removed = initial_rows - len(df_processed)
        messages.append(f"Null values dropped: {removed} rows removed")
    
    # 3. Handle duplicates
    if data.get("drop_dupplicated", False):
        initial_rows = len(df_processed)
        df_processed = df_processed.drop_duplicates()
        removed = initial_rows - len(df_processed)
        messages.append(f"Duplicates dropped: {removed} rows removed")
    
    # 4. Handle infinite values
    if data.get("drop_inf", False):
        initial_rows = len(df_processed)
        df_replaced = df_processed.replace([float('inf'), float('-inf')], float('nan'))
        df_processed = df_replaced.dropna()
        removed = initial_rows - len(df_processed)
        messages.append(f"Infinite values dropped: {removed} rows removed")
    
    # 5. Label encoding
    label_encoded_columns = data.get("label_encoded_columns", [])
    if label_encoded_columns:
        label_encoders = {}
        for col in label_encoded_columns:
            if col in df_columns:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                label_encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))
                messages.append(f"Label encoding applied to column '{col}'")
            else:
                messages.append(f"Warning: Column '{col}' not found for label encoding")
    
    # 6. One-hot encoding
    one_hot_encoded_columns = data.get("one_hot_encoded_columns", [])
    if one_hot_encoded_columns:
        for col in one_hot_encoded_columns:
            if col in df_columns:
                ohc = OneHotEncoder(sparse_output=False)
                one_hot_encoded = ohc.fit_transform(df_processed[[col]])
                ohc_df = pd.DataFrame(
                    one_hot_encoded,
                    columns=[f"{col}_{category}" for category in ohc.categories_[0]]
                )
                df_processed = df_processed.drop(col, axis=1)
                df_processed = pd.concat([df_processed, ohc_df], axis=1)
                messages.append(f"One-hot encoding applied to column '{col}'")
            else:
                messages.append(f"Warning: Column '{col}' not found for one-hot encoding")
    
    # 7. Apply filter
    filtered_df, filter_applied, filter_msg = apply_saved_filter(data, df_columns, df_processed)
    if filter_applied:
        df_processed = filtered_df
        messages.append(filter_msg)
    
    # 8. Split data (simplified - returns None for train/test if not configured)
    train_df = None
    test_df = None
    if data.get("test_size", 0) > 0:
        # This is a simplified version - original has custom split logic
        test_size = data.get("test_size", 0.2)
        random_state = data.get("random_state", 42)
        train_df, test_df = train_test_split(df_processed, test_size=test_size, random_state=random_state)
        messages.append(f"Data split: train={len(train_df)}, test={len(test_df)}")
    
    return df_processed, train_df, test_df, messages
