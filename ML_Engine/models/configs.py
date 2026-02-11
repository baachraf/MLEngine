"""
Model Configurations
====================

Loads and manages model configurations from external YAML files.
"""

import yaml
import os
import warnings
from sklearn import (
    gaussian_process, linear_model, discriminant_analysis, dummy,
    ensemble, tree, neighbors, svm, naive_bayes, kernel_ridge,
    neural_network
)
from sklearn.linear_model import SGDClassifier

# Optional imports
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    xgb = None
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    lgb = None
    LGB_AVAILABLE = False

# --- Helper for missing libraries ---
class _MissingLibrary:
    def __init__(self, library_name):
        self.library_name = library_name
    def __call__(self, *args, **kwargs):
        raise ImportError(f"{self.library_name} is not installed.")

# --- Mapping model names to their actual class objects ---
MODEL_CLASS_MAP = {
    # Classification
    "GaussianProcessClassifier": gaussian_process.GaussianProcessClassifier,
    "RidgeClassifier": linear_model.RidgeClassifier,
    "QuadraticDiscriminantAnalysis": discriminant_analysis.QuadraticDiscriminantAnalysis,
    "LinearDiscriminantAnalysis": discriminant_analysis.LinearDiscriminantAnalysis,
    "DummyClassifier": dummy.DummyClassifier,
    "LogisticRegression": linear_model.LogisticRegression,
    "GradientBoostingClassifier": ensemble.GradientBoostingClassifier,
    "ExtraTreesClassifier": ensemble.ExtraTreesClassifier,
    "RandomForestClassifier": ensemble.RandomForestClassifier,
    "XGBClassifier": xgb.XGBClassifier if XGB_AVAILABLE else _MissingLibrary("xgboost"),
    "LGBMClassifier": lgb.LGBMClassifier if LGB_AVAILABLE else _MissingLibrary("lightgbm"),
    "DecisionTreeClassifier": tree.DecisionTreeClassifier,
    "KNeighborsClassifier": neighbors.KNeighborsClassifier,
    "SVC": svm.SVC,
    "GaussianNB": naive_bayes.GaussianNB,
    "AdaBoostClassifier": ensemble.AdaBoostClassifier,
    "SGDClassifier": SGDClassifier,
    # Regression
    "KNeighborsRegressor": neighbors.KNeighborsRegressor,
    "DecisionTreeRegressor": tree.DecisionTreeRegressor,
    "AdaBoostRegressor": ensemble.AdaBoostRegressor,
    "DummyRegressor": dummy.DummyRegressor,
    "PassiveAggressiveRegressor": linear_model.PassiveAggressiveRegressor,
    "RANSACRegressor": linear_model.RANSACRegressor,
    "TheilSenRegressor": linear_model.TheilSenRegressor,
    "HuberRegressor": linear_model.HuberRegressor,
    "KernelRidge": kernel_ridge.KernelRidge,
    "SVR": svm.SVR,
    "Lars": linear_model.Lars,
    "LassoLars": linear_model.LassoLars,
    "OrthogonalMatchingPursuit": linear_model.OrthogonalMatchingPursuit,
    "BayesianRidge": linear_model.BayesianRidge,
    "ARDRegression": linear_model.ARDRegression,
    "LinearRegression": linear_model.LinearRegression,
    "Ridge": linear_model.Ridge,
    "Lasso": linear_model.Lasso,
    "GradientBoostingRegressor": ensemble.GradientBoostingRegressor,
    "ExtraTreesRegressor": ensemble.ExtraTreesRegressor,
    "ElasticNet": linear_model.ElasticNet,
    "RandomForestRegressor": ensemble.RandomForestRegressor,
    "XGBRegressor": xgb.XGBRegressor if XGB_AVAILABLE else _MissingLibrary("xgboost"),
    "LGBMRegressor": lgb.LGBMRegressor if LGB_AVAILABLE else _MissingLibrary("lightgbm"),
}

# --- Common aliases for model names ---
_COMMON_MODEL_ALIASES = {
    'XGBoost': 'XGBClassifier',
    'LightGBM': 'LGBMClassifier',
}

# --- Regression-specific aliases ---
_REGRESSION_ALIASES = {
    'XGBoost': 'XGBRegressor',
    'LightGBM': 'LGBMRegressor',
    'RandomForest': 'RandomForestRegressor',
    'DecisionTree': 'DecisionTreeRegressor',
    'GradientBoosting': 'GradientBoostingRegressor',
    'ExtraTrees': 'ExtraTreesRegressor',
    'AdaBoost': 'AdaBoostRegressor',
    'KNeighbors': 'KNeighborsRegressor',
    'SVR': 'SVR',
    'LinearRegression': 'LinearRegression',
    'Ridge': 'Ridge',
    'Lasso': 'Lasso',
    'ElasticNet': 'ElasticNet',
}

_MODEL_CONFIGS = {}

def _load_model_configs(config_path: str = None):
    """Load model configurations from a YAML file."""
    global _MODEL_CONFIGS
    if _MODEL_CONFIGS:
        return

    if config_path is None:
        # Default path assumes `configs` directory is at the project root
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        config_path = os.path.join(base_dir, 'configs', 'model_defaults.yml')

    if not os.path.exists(config_path):
        warnings.warn(f"Configuration file not found at {config_path}. No models will be available.")
        return

    with open(config_path, 'r') as f:
        yaml_configs = yaml.safe_load(f)
    
    # Add the actual class object back into the loaded config
    for problem_type, models in yaml_configs.items():
        for model_name, config in models.items():
            if model_name in MODEL_CLASS_MAP:
                config['class'] = MODEL_CLASS_MAP[model_name]
            else:
                warnings.warn(f"Class for model '{model_name}' not found in MODEL_CLASS_MAP.")
    
    _MODEL_CONFIGS = yaml_configs

def get_model_config(model_name: str, problem_type: str, config_path: str = None) -> dict:
    """
    Get configuration for a specific model.

    Parameters
    ----------
    model_name : str
        Name of the model.
    problem_type : {'classification', 'regression'}
        Type of problem (case-insensitive).
    config_path : str, optional
        Path to a custom model configuration YAML file.

    Returns
    -------
    dict
        Model configuration.
    """
    _load_model_configs(config_path)
    
    # Normalize problem_type to lowercase for case-insensitive matching
    problem_type = problem_type.lower()
    
    models_dict = _MODEL_CONFIGS.get(problem_type)
    if not models_dict:
        raise ValueError(f"No models found for problem type: {problem_type}")
    
    # Check for common aliases first
    if problem_type == 'classification':
        aliased_name = _COMMON_MODEL_ALIASES.get(model_name, model_name)
    elif problem_type == 'regression':
        # Use regression-specific aliases
        aliased_name = _REGRESSION_ALIASES.get(model_name, model_name)
    else:
        aliased_name = model_name # Should not happen due to problem_type check

    model_name_to_check = aliased_name

    if model_name_to_check not in models_dict:
        # Fallback to appending suffix if alias didn't work or wasn't used
        if problem_type == 'classification' and f"{model_name_to_check}Classifier" in models_dict:
            model_name_to_check = f"{model_name_to_check}Classifier"
        elif problem_type == 'regression' and f"{model_name_to_check}Regressor" in models_dict:
            model_name_to_check = f"{model_name_to_check}Regressor"
        else:
            raise ValueError(f"Model '{model_name}' not found for {problem_type} models. "
                             f"Available models: {list(models_dict.keys())}")
    
    return models_dict[model_name_to_check]
def list_models(problem_type: str, config_path: str = None) -> list:
    """
    List available models for a problem type.

    Parameters
    ----------
    problem_type : {'classification', 'regression'}
    config_path : str, optional
        Path to a custom model configuration YAML file.

    Returns
    -------
    list
        List of model names.
    """
    _load_model_configs(config_path)
    
    # Normalize problem_type to lowercase for case-insensitive matching
    problem_type = problem_type.lower()
    
    models_dict = _MODEL_CONFIGS.get(problem_type)
    if not models_dict:
        return []
        
    return list(models_dict.keys())
