"""
Tests for the data.io module.
"""

import pandas as pd
import pytest
from io import StringIO
from ML_Engine.data import io

# Fixture to create a sample DataFrame
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    })

def test_load_data_csv(sample_df, tmp_path):
    """
    Test that load_data can correctly load a CSV from a temporary file.
    """
    # Create a temporary file path
    temp_file = tmp_path / "temp_test.csv"
    
    # Convert the sample DataFrame to a CSV string and write to the temp file
    sample_df.to_csv(temp_file, index=False)
    
    # Load the data using the library function
    loaded_df = io.load_data(temp_file, file_type='csv')
    
    # Assert that the loaded DataFrame is identical to the original
    pd.testing.assert_frame_equal(sample_df, loaded_df)

def test_split_data_train_test(sample_df):
    """
    Test the random train-test split functionality.
    """
    features = ['A', 'B']
    targets = ['C']
    
    X_train, X_test, y_train, y_test = io.split_data_train_test(
        df=sample_df,
        selected_features=features,
        selected_targets=targets,
        test_size=0.33, # Approx 1/3 for a 3-row df
        random_state=42
    )
    
    # Check the shapes of the output
    assert X_train.shape == (2, 2)
    assert X_test.shape == (1, 2)
    assert y_train.shape == (2, 1)
    assert y_test.shape == (1, 1)
    
    # Check that the columns are correct
    assert list(X_train.columns) == features
    assert list(y_train.columns) == targets
