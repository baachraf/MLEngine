"""
Model Configurations
====================

Pre-defined model configurations for classification and regression.
"""

from sklearn import (
    gaussian_process, linear_model, discriminant_analysis, dummy,
    ensemble, tree, neighbors, svm, naive_bayes, kernel_ridge,
    neural_network
)
from sklearn.linear_model import SGDClassifier
try:
    import xgboost as xgb
except ImportError:
    xgb = None
    import warnings
    warnings.warn('xgboost not installed. XGBoost models will not be available.')

try:
    import lightgbm as lgb
except ImportError:
    lgb = None
    import warnings
    warnings.warn('lightgbm not installed. LightGBM models will not be available.')


# Helper class for missing libraries
class _MissingLibrary:
    """Placeholder for missing library that raises informative ImportError."""
    def __init__(self, library_name):
        self.library_name = library_name
    
    def __call__(self, *args, **kwargs):
        raise ImportError(
            f"{self.library_name} is not installed. "
            f"Please install it to use this model."
        )

# ============================================================================
# CLASSIFICATION MODELS
# ============================================================================

CLASSIFICATION_MODELS = {
    "GaussianProcessClassifier": {
        'class': gaussian_process.GaussianProcessClassifier,
        'params': {
            'max_iter_predict': {'type': 'slider', 'label': 'Maximum Iterations for Prediction', 'min': 100,
                                 'max': 1000, 'default': 100},
            'n_restarts_optimizer': {'type': 'slider', 'label': 'Number of Optimizer Restarts', 'min': 0, 'max': 10,
                                     'default': 0}
        },
        'multi_output': False,  # Requires MultiOutputClassifier wrapper
        'supports_sample_weight': False,
        'supports_class_weight': False,
        'notes': 'Probabilistic classification using Gaussian processes. Can be computationally intensive for large datasets.'
    },
    # "MLPClassifier": {
    #     'class': neural_network.MLPClassifier,
    #     'params': {
    #         'hidden_layer_sizes': {'type': 'slider', 'label': 'Hidden Layer Size', 'min': 10, 'max': 200,
    #                                'default': 100},
    #         'learning_rate_init': {'type': 'slider', 'label': 'Initial Learning Rate', 'min': 0.0001, 'max': 0.1,
    #                                'default': 0.001},
    #         'max_iter': {'type': 'slider', 'label': 'Maximum Iterations', 'min': 100, 'max': 1000, 'default': 200},
    #         'alpha': {'type': 'slider', 'label': 'Alpha (L2 penalty)', 'min': 0.0001, 'max': 0.1, 'default': 0.0001}
    #     },
    #     'multi_output': True,  # Native multi-output support
    #     'supports_sample_weight': False,
    #     'supports_class_weight': False,
    #     'notes': 'Neural network using backpropagation. Good for complex non-linear relationships.'
    # },
    "RidgeClassifier": {
        'class': linear_model.RidgeClassifier,
        'params': {
            'alpha': {'type': 'slider', 'label': 'Alpha (regularization strength)', 'min': 0.01, 'max': 10.0,
                      'default': 1.0},
            'max_iter': {'type': 'slider', 'label': 'Maximum Iterations', 'min': 100, 'max': 2000, 'default': 1000}
        },
        'multi_output': True,  # Native multi-output support
        'supports_sample_weight': True,
        'supports_class_weight': True,
        'notes': 'Linear classifier with L2 regularization. Good for high-dimensional data.'
    },
    "QuadraticDiscriminantAnalysis": {
        'class': discriminant_analysis.QuadraticDiscriminantAnalysis,
        'params': {
            'reg_param': {'type': 'slider', 'label': 'Regularization Parameter', 'min': 0.0, 'max': 1.0,
                          'default': 0.0},
            'tol': {'type': 'input', 'label': 'Tolerance', 'min': 1e-6, 'max': 1e-2, 'default': 1e-4}
        },
        'multi_output': False,  # Requires MultiOutputClassifier wrapper
        'supports_sample_weight': True,
        'supports_class_weight': False,
        'notes': 'Quadratic classifier with no assumptions of equal covariance matrices.'
    },
    "LinearDiscriminantAnalysis": {
        'class': discriminant_analysis.LinearDiscriminantAnalysis,
        'params': {
            'n_components': {'type': 'slider', 'label': 'Number of Components', 'min': 1, 'max': 10, 'default': None},
            'tol': {'type': 'input', 'label': 'Tolerance', 'min': 1e-6, 'max': 1e-2, 'default': 1e-4}
        },
        'multi_output': False,  # Requires MultiOutputClassifier wrapper
        'supports_sample_weight': True,
        'supports_class_weight': False,
        'notes': 'Linear classifier with assumption of equal covariance matrices. Also useful for dimensionality reduction.'
    },
    "DummyClassifier": {
        'class': dummy.DummyClassifier,
        'params': {
            'strategy': {'type': 'select', 'label': 'Strategy',
                         'options': ['stratified', 'most_frequent', 'prior', 'uniform', 'constant'], 'default': 'prior'}
        },
        'multi_output': True,  # Native multi-output support
        'supports_sample_weight': True,
        'supports_class_weight': False,
        'notes': 'Baseline classifier that makes predictions using simple rules. Useful as a baseline comparison.'
    },
    "LogisticRegression": {
        'class': linear_model.LogisticRegression,
        'params': {
            'C': {'type': 'slider', 'label': 'C (Inverse of regularization strength)', 'min': 0.01, 'max': 100.0, 'default': 1.0},
            'solver': {'type': 'select', 'label': 'Solver', 'options': ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'], 'default': 'lbfgs'}
        },
        'multi_output': False,  # Requires MultiOutputClassifier wrapper
        'supports_sample_weight': True,
        'supports_class_weight': True,
        'notes': 'Good for binary classification, needs wrapper for multi-output'
    },
    "GradientBoostingClassifier": {
        'class': ensemble.GradientBoostingClassifier,
        'params': {
            'n_estimators': {'type': 'slider', 'label': 'Number of Estimators', 'min': 10, 'max': 300, 'default': 100},
            'learning_rate': {'type': 'slider', 'label': 'Learning Rate', 'min': 0.01, 'max': 1.0, 'default': 0.1},
            'max_depth': {'type': 'slider', 'label': 'Max Depth', 'min': 1, 'max': 20, 'default': 3},
            'min_samples_split': {'type': 'slider', 'label': 'Min Samples Split', 'min': 2, 'max': 20, 'default': 2},
            'subsample': {'type': 'slider', 'label': 'Subsample', 'min': 0.1, 'max': 1.0, 'default': 1.0}
        },
        'multi_output': False,  # Requires MultiOutputClassifier wrapper
        'supports_sample_weight': True,
        'supports_class_weight': False,
        'notes': 'Gradient Boosting for classification. Can overfit on small datasets but often works well on larger ones.'
    },
    "ExtraTreesClassifier": {
        'class': ensemble.ExtraTreesClassifier,
        'params': {
            'n_estimators': {'type': 'slider', 'label': 'Number of Estimators', 'min': 10, 'max': 300, 'default': 100},
            'max_depth': {'type': 'slider', 'label': 'Max Depth', 'min': 1, 'max': 50, 'default': 10},
            'min_samples_split': {'type': 'slider', 'label': 'Min Samples Split', 'min': 2, 'max': 20, 'default': 2}
        },
        'multi_output': True,  # Native multi-output support
        'supports_sample_weight': True,
        'supports_class_weight': True,
        'notes': 'Randomized decision trees for classification.'
    },
    "RandomForestClassifier": {
        'class': ensemble.RandomForestClassifier,
        'params': {
            'n_estimators': {'type': 'slider', 'label': 'Number of Estimators', 'min': 10, 'max': 300, 'default': 100},
            'max_depth': {'type': 'slider', 'label': 'Max Depth', 'min': 1, 'max': 50, 'default': 10},
            'min_samples_split': {'type': 'slider', 'label': 'Min Samples Split', 'min': 2, 'max': 20, 'default': 2}
        },
        'multi_output': True,  # Native multi-output support
        'supports_sample_weight': True,
        'supports_class_weight': True,
        'notes': 'Natively supports multi-output classification'
    },
    "XGBClassifier": {
        'class': xgb.XGBClassifier if xgb is not None else _MissingLibrary('xgboost'),
        'params': {
            'n_estimators': {
                'type': 'slider',
                'label': 'Number of Estimators',
                'min': 10,
                'max': 300,
                'default': 100,
                'help': 'The number of boosting rounds or trees. More trees increase accuracy but also training time.'
            },
            'max_depth': {
                'type': 'slider',
                'label': 'Max Depth',
                'min': 1,
                'max': 20,
                'default': 6,
                'help': 'The maximum depth of each tree. Higher depth can capture more patterns but may lead to overfitting.'
            },
            'learning_rate': {
                'type': 'slider',
                'label': 'Learning Rate',
                'min': 0.01,
                'max': 1.0,
                'default': 0.3,
                'help': 'Step size used to shrink the contribution of each tree. Lower values make the learning process slower but may yield better results.'
            },
            'subsample': {
                'type': 'slider',
                'label': 'Subsample',
                'min': 0.1,
                'max': 1.0,
                'default': 1.0,
                'help': 'The fraction of samples used in each boosting round. Reducing this can help prevent overfitting.'
            },
            'colsample_bytree': {
                'type': 'slider',
                'label': 'Colsample by Tree',
                'min': 0.1,
                'max': 1.0,
                'default': 1.0,
                'help': 'The fraction of features used when constructing each tree. Lower values can help reduce overfitting.'
            },
            'min_child_weight': {
                'type': 'slider',
                'label': 'Min Child Weight',
                'min': 1,
                'max': 10,
                'default': 1,
                'help': 'Minimum sum of instance weight in a child. Helps prevent overfitting by controlling the complexity of the model.'
            },
            'gamma': {
                'type': 'slider',
                'label': 'Gamma',
                'min': 0.0,
                'max': 10.0,
                'default': 0.0,
                'help': 'The minimum loss reduction required for further partitioning. A higher value leads to more conservative splits.'
            },
            'reg_lambda': {
                'type': 'slider',
                'label': 'L2 Regularization (Lambda)',
                'min': 0.0,
                'max': 10.0,
                'default': 1.0,
                'help': 'L2 regularization for weights. Larger values make the model more resistant to overfitting.'
            },
            'reg_alpha': {
                'type': 'slider',
                'label': 'L1 Regularization (Alpha)',
                'min': 0.0,
                'max': 10.0,
                'default': 0.0,
                'help': 'L1 regularization for weights. It adds a penalty for large weights, encouraging sparsity and reducing overfitting.'
            },
            'scale_pos_weight': {
                'type': 'slider',
                'label': 'Scale Pos Weight (for Imbalance)',
                'min': 0.1,
                'max': 10.0,
                'default': 1.0,
                'help': 'Controls the balance of positive and negative classes. A higher value gives more weight to positive classes in imbalanced datasets.'
            },
            'early_stopping_rounds': {
                'type': 'slider',
                'label': 'Early Stopping Rounds',
                'min': 0,
                'max': 100,
                'default': 10,
                'help': 'Stops training if no improvement in evaluation metric after a set number of rounds. Helps avoid overfitting.'
            }
        },
        'multi_output': False,  # Requires MultiOutputClassifier wrapper
        'supports_sample_weight': True,
        'supports_class_weight': True,
        'notes': 'Needs wrapper for multi-output, supports both sample and class weights'
    },
    "LGBMClassifier": {
        'class': lgb.LGBMClassifier if lgb is not None else _MissingLibrary('lightgbm'),
        'params': {
            'n_estimators': {'type': 'slider', 'label': 'Number of Estimators', 'min': 10, 'max': 300, 'default': 100},
            'max_depth': {'type': 'slider', 'label': 'Max Depth', 'min': 1, 'max': 20, 'default': 6},
            'learning_rate': {'type': 'slider', 'label': 'Learning Rate', 'min': 0.01, 'max': 1.0, 'default': 0.3}
        },
        'multi_output': False,  # Requires MultiOutputClassifier wrapper
        'supports_sample_weight': True,
        'supports_class_weight': True,
        'notes': 'Needs wrapper for multi-output, efficient for large datasets'
    },
    "DecisionTreeClassifier": {
        'class': tree.DecisionTreeClassifier,
        'params': {
            'max_depth': {'type': 'slider', 'label': 'Max Depth', 'min': 1, 'max': 50, 'default': 10},
            'min_samples_split': {'type': 'slider', 'label': 'Min Samples Split', 'min': 2, 'max': 20, 'default': 2}
        },
        'multi_output': True,  # Native multi-output support
        'supports_sample_weight': True,
        'supports_class_weight': True,
        'notes': 'Natively supports multi-output classification'
    },
    "KNearestNeighbors": {
        'class': neighbors.KNeighborsClassifier,
        'params': {
            'n_neighbors': {'type': 'slider', 'label': 'Number of Neighbors', 'min': 1, 'max': 50, 'default': 5},
            'algorithm': {'type': 'select', 'label': 'Algorithm', 'options': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'default': 'auto'}
        },
        'multi_output': True,  # Native multi-output support
        'supports_sample_weight': False,  # KNN doesn't support sample weights
        'supports_class_weight': False,
        'notes': 'Natively supports multi-output, but no weight support'
    },
    "SVC": {
        'class': svm.SVC,
        'params': {
            'C': {'type': 'slider', 'label': 'C (Regularization Parameter)', 'min': 0.01, 'max': 100.0, 'default': 1.0},
            'kernel': {'type': 'select', 'label': 'Kernel', 'options': ['linear', 'poly', 'rbf', 'sigmoid'], 'default': 'rbf'}
        },
        'multi_output': False,  # Requires MultiOutputClassifier wrapper
        'supports_sample_weight': True,
        'supports_class_weight': True,
        'notes': 'Needs wrapper for multi-output, can be slow on large datasets'
    },
    "GaussianNB": {
        'class': naive_bayes.GaussianNB,
        'params': {},
        'multi_output': False,  # Requires MultiOutputClassifier wrapper
        'supports_sample_weight': True,
        'supports_class_weight': False,
        'notes': 'Needs wrapper for multi-output, supports sample weights only'
    },
    "AdaBoostClassifier": {
        'class': ensemble.AdaBoostClassifier,
        'params': {
            'n_estimators': {'type': 'slider', 'label': 'Number of Estimators', 'min': 10, 'max': 300, 'default': 50},
            'learning_rate': {'type': 'slider', 'label': 'Learning Rate', 'min': 0.01, 'max': 2.0, 'default': 1.0}
        },
        'multi_output': False,  # Requires MultiOutputClassifier wrapper
        'supports_sample_weight': True,
        'supports_class_weight': False,
        'notes': 'Needs wrapper for multi-output, inherently handles imbalanced datasets'
    },
    "SGDClassifier": {
    'class': SGDClassifier,
    'params': {
        'loss': {
            'type': 'select',
            'label': 'Loss Function',
            'options': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
            'default': 'hinge',
            'help': 'Specifies the loss function to be used. For instance, "hinge" is for SVM, "log" for logistic regression, and "perceptron" for linear perceptron.'
        },
        'penalty': {
            'type': 'select',
            'label': 'Penalty',
            'options': ['l2', 'l1', 'elasticnet'],
            'default': 'l2',
            'help': 'Regularization term to avoid overfitting. Use "l2" for Ridge, "l1" for Lasso, and "elasticnet" for a mix of both.'
        },
        'alpha': {
            'type': 'input',
            'label': 'Alpha (Regularization Strength)',
            'min': 1e-6,
            'max': 1e-1,
            'default': 0.0001,
            'help': 'Constant that multiplies the regularization term. Smaller values make the model more flexible but risk overfitting.'
        },
        'l1_ratio': {
            'type': 'slider',
            'label': 'L1 Ratio (ElasticNet Mixing)',
            'min': 0.0,
            'max': 1.0,
            'default': 0.15,
            'help': 'The ElasticNet mixing parameter. Used only if "penalty" is set to "elasticnet". 0 means L2 penalty only, and 1 means L1 penalty only.'
        },
        'max_iter': {
            'type': 'slider',
            'label': 'Max Iterations',
            'min': 100,
            'max': 10000,
            'default': 1000,
            'help': 'The maximum number of iterations the optimizer will run. Increase this if the model does not converge.'
        },
        'tol': {
            'type': 'input',
            'label': 'Tolerance',
            'min': 1e-6,
            'max': 1e-2,
            'default': 1e-3,
            'help': 'The stopping criterion for optimization. Smaller values require more precision but take longer.'
        },
        'learning_rate': {
            'type': 'select',
            'label': 'Learning Rate Schedule',
            'options': ['constant', 'optimal', 'invscaling', 'adaptive'],
            'default': 'optimal',
            'help': 'Controls the step size in updating weights. "optimal" is recommended for most tasks, while "adaptive" adjusts based on performance.'
        },
        'eta0': {
            'type': 'input',
            'label': 'Initial Learning Rate (Eta0)',
            'min': 1e-5,
            'max': 1.0,
            'default': 0.01,
            'help': 'The initial step size used by the learning rate schedule. Relevant only if "learning_rate" is not "optimal".'
        },
        'power_t': {
            'type': 'slider',
            'label': 'Power T (Learning Rate Exponent)',
            'min': 0.1,
            'max': 1.0,
            'default': 0.5,
            'help': 'Exponent for "invscaling" learning rate. Higher values slow down learning as training progresses.'
        },

        'validation_fraction': {
            'type': 'slider',
            'label': 'Validation Fraction',
            'min': 0.1,
            'max': 0.5,
            'default': 0.1,
            'help': 'The proportion of training data to set aside as validation data for early stopping.'
        },
        'n_iter_no_change': {
            'type': 'slider',
            'label': 'Number of Iterations Without Improvement',
            'min': 1,
            'max': 10,
            'default': 5,
            'help': 'Number of epochs with no improvement to wait before stopping early. Used with early stopping.'
        },

    },
    'multi_output': False,  # Requires MultiOutputClassifier wrapper
    'supports_sample_weight': True,
    'supports_class_weight': True,
    'notes': 'SGDClassifier is a highly flexible model that works well for linear classification tasks. Be mindful of tuning the regularization parameters to avoid overfitting or underfitting.'
}
}

# ============================================================================
# REGRESSION MODELS
# ============================================================================

REGRESSION_MODELS = {
    "KNeighborsRegressor": {
        'class': neighbors.KNeighborsRegressor,
        'params': {
            'n_neighbors': {
                'type': 'slider',
                'label': 'Number of Neighbors',
                'min': 1,
                'max': 50,
                'default': 5,
                'help': 'Number of neighbors to use for prediction. Higher values reduce the impact of noise but make boundaries less distinct.'
            },
            'weights': {
                'type': 'select',
                'label': 'Weight Function',
                'options': ['uniform', 'distance'],
                'default': 'uniform',
                'help': 'Weight function used in prediction. Uniform: all points weighted equally. Distance: points weighted by inverse of distance.'
            },
            'p': {
                'type': 'slider',
                'label': 'Power Parameter',
                'min': 1,
                'max': 2,
                'default': 2,
                'help': 'Power parameter for the Minkowski metric. p=1 is Manhattan distance, p=2 is Euclidean distance.'
            }
        },
        'multi_output': True,  # Native multi-output support
        'supports_sample_weight': False,
        'notes': 'Simple yet effective method for non-linear regression. Performance depends heavily on feature scaling.'
    },

    "DecisionTreeRegressor": {
        'class': tree.DecisionTreeRegressor,
        'params': {
            'max_depth': {
                'type': 'slider',
                'label': 'Maximum Depth',
                'min': 1,
                'max': 100,
                'default': None,
                'help': 'Maximum depth of the tree. Deeper trees can model more complex patterns but may overfit.'
            },
            'min_samples_split': {
                'type': 'slider',
                'label': 'Minimum Samples Split',
                'min': 2,
                'max': 50,
                'default': 2,
                'help': 'Minimum number of samples required to split an internal node.'
            },
            'min_samples_leaf': {
                'type': 'slider',
                'label': 'Minimum Samples Leaf',
                'min': 1,
                'max': 50,
                'default': 1,
                'help': 'Minimum number of samples required to be at a leaf node.'
            },
            'max_features': {
                'type': 'slider',
                'label': 'Maximum Features',
                'min': 0.1,
                'max': 1.0,
                'default': None,
                'help': 'Fraction of features to consider when looking for the best split.'
            }
        },
        'multi_output': True,  # Native multi-output support
        'supports_sample_weight': True,
        'notes': 'Simple to understand and interpret. Tends to overfit if not properly constrained.'
    },

    "AdaBoostRegressor": {
        'class': ensemble.AdaBoostRegressor,
        'params': {
            'n_estimators': {
                'type': 'slider',
                'label': 'Number of Estimators',
                'min': 10,
                'max': 500,
                'default': 50,
                'help': 'Maximum number of estimators at which boosting is terminated.'
            },
            'learning_rate': {
                'type': 'slider',
                'label': 'Learning Rate',
                'min': 0.01,
                'max': 1.0,
                'default': 1.0,
                'help': 'Weight applied to each regressor at each boosting iteration.'
            },
            'loss': {
                'type': 'select',
                'label': 'Loss Function',
                'options': ['linear', 'square', 'exponential'],
                'default': 'linear',
                'help': 'The loss function to use when updating the weights after each boosting iteration.'
            }
        },
        'multi_output': False,  # Requires MultiOutputRegressor wrapper
        'supports_sample_weight': True,
        'notes': 'Ensemble method that combines weak learners sequentially. Good for reducing bias and variance.'
    },

    # "MLPRegressor": {
    #     'class': neural_network.MLPRegressor,
    #     'params': {
    #         'hidden_layer_sizes': {
    #             'type': 'text',
    #             'label': 'Hidden Layer Sizes',
    #             'default': '(100,)',
    #             'help': 'Number of neurons in each hidden layer. Format: tuple of integers.'
    #         },
    #         'activation': {
    #             'type': 'select',
    #             'label': 'Activation Function',
    #             'options': ['identity', 'logistic', 'tanh', 'relu'],
    #             'default': 'relu',
    #             'help': 'Activation function for hidden layer.'
    #         },
    #         'solver': {
    #             'type': 'select',
    #             'label': 'Solver',
    #             'options': ['lbfgs', 'sgd', 'adam'],
    #             'default': 'adam',
    #             'help': 'The solver for weight optimization.'
    #         },
    #         'alpha': {
    #             'type': 'slider',
    #             'label': 'Alpha (L2 penalty)',
    #             'min': 0.0001,
    #             'max': 1.0,
    #             'default': 0.0001,
    #             'help': 'L2 penalty (regularization term) parameter.'
    #         },
    #         'learning_rate': {
    #             'type': 'select',
    #             'label': 'Learning Rate',
    #             'options': ['constant', 'invscaling', 'adaptive'],
    #             'default': 'constant',
    #             'help': 'Learning rate schedule for weight updates.'
    #         },
    #         'max_iter': {
    #             'type': 'slider',
    #             'label': 'Maximum Iterations',
    #             'min': 100,
    #             'max': 1000,
    #             'default': 200,
    #             'help': 'Maximum number of iterations.'
    #         }
    #     },
    #     'multi_output': True,  # Native multi-output support
    #     'supports_sample_weight': True,
    #     'notes': 'Neural network model. Good for complex non-linear relationships but requires careful tuning.'
    # },

    "DummyRegressor": {
        'class': dummy.DummyRegressor,
        'params': {
            'strategy': {
                'type': 'select',
                'label': 'Strategy',
                'options': ['mean', 'median', 'quantile', 'constant'],
                'default': 'mean',
                'help': 'Strategy to use for generating predictions.'
            },
            'constant': {
                'type': 'slider',
                'label': 'Constant Value',
                'min': -100,
                'max': 100,
                'default': 0,
                'help': 'Constant value to predict if strategy is constant.'
            },
            'quantile': {
                'type': 'slider',
                'label': 'Quantile',
                'min': 0.0,
                'max': 1.0,
                'default': 0.5,
                'help': 'Quantile to predict if strategy is quantile.'
            }
        },
        'multi_output': True,  # Native multi-output support
        'supports_sample_weight': True,
        'notes': 'Baseline regressor that makes simple predictions. Useful as a benchmark for comparing with other regressors.'
    },
    "PassiveAggressiveRegressor": {
        'class': linear_model.PassiveAggressiveRegressor,
        'params': {
            'C': {
                'type': 'slider',
                'label': 'Regularization Parameter',
                'min': 0.01,
                'max': 10.0,
                'default': 1.0,
                'help': 'Maximum step size for weight updates. Larger values mean more aggressive updates.'
            },
            'max_iter': {
                'type': 'slider',
                'label': 'Maximum Iterations',
                'min': 100,
                'max': 2000,
                'default': 1000,
                'help': 'Maximum number of passes over the training data.'
            },
            'tol': {
                'type': 'input',
                'label': 'Tolerance',
                'min': 1e-5,
                'max': 1e-2,
                'default': 1e-3,
                'help': 'Stopping criterion. Stop when loss is less than tol.'
            }
        },
        'multi_output': False,  # Requires MultiOutputRegressor wrapper
        'supports_sample_weight': True,
        'notes': 'Online learning algorithm that remains passive for correct predictions and aggressive for incorrect ones.'
    },

    "RANSACRegressor": {
        'class': linear_model.RANSACRegressor,
        'params': {
            'min_samples': {
                'type': 'slider',
                'label': 'Minimum Samples',
                'min': 1,
                'max': 50,
                'default': None,
                'help': 'Minimum number of samples chosen randomly from original data.'
            },
            'max_trials': {
                'type': 'slider',
                'label': 'Maximum Trials',
                'min': 10,
                'max': 1000,
                'default': 100,
                'help': 'Maximum number of iterations for random sample selection.'
            },
            'residual_threshold': {
                'type': 'slider',
                'label': 'Residual Threshold',
                'min': 0.1,
                'max': 10.0,
                'default': None,
                'help': 'Maximum residual for a sample to be classified as an inlier.'
            }
        },
        'multi_output': False,  # Requires MultiOutputRegressor wrapper
        'supports_sample_weight': False,
        'notes': 'Robust method for fitting a model in the presence of outliers.'
    },

    "TheilSenRegressor": {
        'class': linear_model.TheilSenRegressor,
        'params': {
            'max_iter': {
                'type': 'slider',
                'label': 'Maximum Iterations',
                'min': 100,
                'max': 2000,
                'default': 300,
                'help': 'Maximum number of iterations for the optimization.'
            },
            'tol': {
                'type': 'input',
                'label': 'Tolerance',
                'min': 1e-5,
                'max': 1e-2,
                'default': 1e-3,
                'help': 'Tolerance for stopping criterion.'
            }
        },
        'multi_output': False,  # Requires MultiOutputRegressor wrapper
        'supports_sample_weight': True,
        'notes': 'Robust regression method that is resistant to outliers.'
    },

    "HuberRegressor": {
        'class': linear_model.HuberRegressor,
        'params': {
            'epsilon': {
                'type': 'slider',
                'label': 'Epsilon',
                'min': 1.1,
                'max': 5.0,
                'default': 1.35,
                'help': 'Parameter that controls the number of samples that should be classified as outliers.'
            },
            'max_iter': {
                'type': 'slider',
                'label': 'Maximum Iterations',
                'min': 100,
                'max': 2000,
                'default': 100,
                'help': 'Maximum number of iterations for the optimization.'
            },
            'alpha': {
                'type': 'input',
                'label': 'Alpha',
                'min': 0.0,
                'max': 1.0,
                'default': 0.0001,
                'help': 'Regularization parameter.'
            }
        },
        'multi_output': False,  # Requires MultiOutputRegressor wrapper
        'supports_sample_weight': True,
        'notes': 'Robust regression method that combines the best of both square loss and absolute loss.'
    },

    "KernelRidge": {
        'class': kernel_ridge.KernelRidge,
        'params': {
            'alpha': {
                'type': 'slider',
                'label': 'Alpha',
                'min': 0.0,
                'max': 10.0,
                'default': 1.0,
                'help': 'Regularization strength.'
            },
            'kernel': {
                'type': 'select',
                'label': 'Kernel',
                'options': ['linear', 'rbf', 'poly', 'sigmoid'],
                'default': 'rbf',
                'help': 'Kernel type to be used.'
            },
            'gamma': {
                'type': 'slider',
                'label': 'Gamma',
                'min': 0.0,
                'max': 10.0,
                'default': None,
                'help': 'Kernel coefficient for rbf, poly and sigmoid kernels.'
            }
        },
        'multi_output': True,  # Native multi-output support
        'supports_sample_weight': False,
        'notes': 'Combines ridge regression with kernel trick. Useful for non-linear relationships.'
    },

    "SVR": {
        'class': svm.SVR,
        'params': {
            'C': {
                'type': 'slider',
                'label': 'C (Regularization)',
                'min': 0.1,
                'max': 100.0,
                'default': 1.0,
                'help': 'Regularization parameter. Trades off margin size against training error.'
            },
            'kernel': {
                'type': 'select',
                'label': 'Kernel',
                'options': ['linear', 'rbf', 'poly', 'sigmoid'],
                'default': 'rbf',
                'help': 'Kernel type to be used.'
            },
            'epsilon': {
                'type': 'slider',
                'label': 'Epsilon',
                'min': 0.01,
                'max': 1.0,
                'default': 0.1,
                'help': 'Specifies the epsilon-tube within which no penalty is associated.'
            },
            "gamma": {
            "type": "select",
            "label": "Gamma",
            "options": [
              "scale",
              "auto"],
                'default': 'scale',
                'help': 'Kernel coefficient for rbf, poly and sigmoid kernels.'
            }
        },
        'multi_output': False,  # Requires MultiOutputRegressor wrapper
        'supports_sample_weight': True,
        'notes': 'Powerful regression method that handles non-linear relationships well.'
    },
    "Lars": {
    "class": linear_model.Lars,
    "params": {
      "n_nonzero_coefs": {
        "type": "slider",
        "label": "Non-zero Coefficients",
        "min": 1,
        "max": 100,
        "default": 500,
        "help": "Target number of non-zero coefficients. Controls model sparsity."
      },
      "fit_intercept": {
        "type": "checkbox",
        "label": "Fit Intercept",
        "default": True,
        "help": "Whether to calculate the intercept for this model."
      },
      "eps": {
        "type": "input",
        "label": "Epsilon",
        "min": 1e-16,
        "max": 1e-2,
        "default": 2.220446049250313e-16,
        "help": "The machine-precision regularization in the computation of the Cholesky diagonal factors."
      },
      "copy_X": {
        "type": "checkbox",
        "label": "Copy X",
        "default": True,
        "help": "If True, X will be copied; else, it may be overwritten."
      },


    },
    "multi_output": False,
    "supports_sample_weight": False,
    "notes": "Efficient for high-dimensional data with L1 regularization. Note: Data normalization should be handled through preprocessing steps using StandardScaler."

    },

    "LassoLars": {
        'class': linear_model.LassoLars,
        'params': {
            'alpha': {
                'type': 'slider',
                'label': 'Alpha (Regularization Strength)',
                'min': 0.0,
                'max': 10.0,
                'default': 1.0,
                'help': 'Regularization parameter. Controls the strength of L1 penalty.'
            },
            'max_iter': {
                'type': 'slider',
                'label': 'Maximum Iterations',
                'min': 100,
                'max': 1000,
                'default': 500,
                'help': 'Maximum number of iterations to perform.'
            }
        },
        'multi_output': False,  # Requires MultiOutputRegressor wrapper
        'supports_sample_weight': False,
        'notes': 'Combines Lasso with LAR algorithm for efficient regularization path computation.'
    },

    "OrthogonalMatchingPursuit": {
        'class': linear_model.OrthogonalMatchingPursuit,
        'params': {
            'n_nonzero_coefs': {
                'type': 'slider',
                'label': 'Non-zero Coefficients',
                'min': 1,
                'max': 100,
                'default': None,
                'help': 'Target number of non-zero coefficients.'
            },
            'tol': {
                'type': 'input',
                'label': 'Tolerance',
                'min': 1e-8,
                'max': 1e-3,
                'default': 1e-6,
                'help': 'Maximum residual norm stopping criterion.'
            }
        },
        'multi_output': False,  # Requires MultiOutputRegressor wrapper
        'supports_sample_weight': False,
        'notes': 'Greedy algorithm for sparse approximation of signals.'
    },

    "BayesianRidge": {
        'class': linear_model.BayesianRidge,
        'params': {
            'n_iter': {
                'type': 'slider',
                'label': 'Maximum Iterations',
                'min': 100,
                'max': 1000,
                'default': 300,
                'help': 'Maximum number of iterations for the optimization algorithm.'
            },
            'alpha_1': {
                'type': 'input',
                'label': 'Alpha 1',
                'min': 1e-7,
                'max': 1e-4,
                'default': 1e-6,
                'help': 'Shape parameter for Gamma distribution of precision of weights.'
            },
            'alpha_2': {
                'type': 'input',
                'label': 'Alpha 2',
                'min': 1e-7,
                'max': 1e-4,
                'default': 1e-6,
                'help': 'Shape parameter for Gamma distribution of precision of noise.'
            }
        },
        'multi_output': False,  # Requires MultiOutputRegressor wrapper
        'supports_sample_weight': False,
        'notes': 'Bayesian approach to ridge regression with automatic relevance determination.'
    },

    "ARDRegression": {
        'class': linear_model.ARDRegression,
        'params': {
            'n_iter': {
                'type': 'slider',
                'label': 'Maximum Iterations',
                'min': 100,
                'max': 1000,
                'default': 300,
                'help': 'Maximum number of iterations for the optimization.'
            },
            'threshold_lambda': {
                'type': 'input',
                'label': 'Threshold Lambda',
                'min': 1e-6,
                'max': 1e-3,
                'default': 1e-4,
                'help': 'Threshold for removing weights with high precision.'
            }
        },
        'multi_output': False,  # Requires MultiOutputRegressor wrapper
        'supports_sample_weight': False,
        'notes': 'Bayesian ARD regression with adaptive regularization.'
    },
    "LinearRegression": {
        'class': linear_model.LinearRegression,
        'params': {},
        'multi_output': True,  # Native multi-output support
        'supports_sample_weight': True,
        'notes': 'Natively supports multi-output regression'
    },
    "Ridge": {
        'class': linear_model.Ridge,
        'params': {
            'alpha': {
                'type': 'slider',
                'label': 'Alpha (Regularization Strength)',
                'min': 0.0,
                'max': 100.0,
                'default': 1.0,
                'help': 'Controls the regularization strength. Higher values increase regularization, which can reduce overfitting but may lead to underfitting. Use 0 for no regularization.'
            }
        },
        'multi_output': True,  # Native multi-output support
        'supports_sample_weight': True,
        'notes': 'Supports L2 regularization to improve generalization in regression tasks.'
    },


    "Lasso": {
        'class': linear_model.Lasso,
        'params': {
            'alpha': {
                'type': 'slider',
                'label': 'Alpha (Regularization Strength)',
                'min': 0.0,
                'max': 100.0,
                'default': 1.0,
                'help': 'Determines the L1 regularization strength. Higher values lead to sparser models by forcing some coefficients to zero, reducing complexity but potentially underfitting.'
            }
        },
        'multi_output': False,  # Requires MultiOutputRegressor wrapper
        'supports_sample_weight': True,
        'notes': 'Applies L1 regularization; useful for sparse models. Multi-output requires a wrapper.'
    },

    "GradientBoostingRegressor": {
    'class': ensemble.GradientBoostingRegressor,
    'params': {
        'n_estimators': {
            'type': 'slider',
            'label': 'Number of Estimators',
            'min': 10,
            'max': 500,
            'default': 100,
            'help': 'The number of boosting stages (trees) to fit. More trees increase model complexity and accuracy but require more computation.'
        },
        'learning_rate': {
            'type': 'slider',
            'label': 'Learning Rate',
            'min': 0.01,
            'max': 1.0,
            'default': 0.1,
            'help': 'Determines the contribution of each tree. Lower values require more trees for convergence but may improve performance.'
        },
        'max_depth': {
            'type': 'slider',
            'label': 'Max Depth',
            'min': 1,
            'max': 100,
            'default': 3,
            'help': 'Limits the depth of each tree. Shallower trees reduce overfitting but may underfit.'
        },
        'min_samples_split': {
            'type': 'slider',
            'label': 'Min Samples Split',
            'min': 2,
            'max': 50,
            'default': 2,
            'help': 'The minimum number of samples required to split an internal node. Higher values prevent overfitting on noisy datasets.'
        },
        'subsample': {
            'type': 'slider',
            'label': 'Subsample',
            'min': 0.1,
            'max': 1.0,
            'default': 1.0,
            'help': 'Fraction of samples used for fitting each tree. Lower values prevent overfitting but increase variance.'
        },
        'max_features': {
            'type': 'slider',
            'label': 'Max Features',
            'min': 0.1,
            'max': 1.0,
            'default': 1.0,
            'help': 'The fraction of features considered when looking for the best split. Lower values can reduce overfitting but may underfit if set too low.'
        }
    },
    'multi_output': False,  # Requires MultiOutputRegressor wrapper
    'supports_sample_weight': True,
    'notes': 'Flexible boosting model that handles nonlinear relationships well. Ensure parameter tuning for optimal performance.'
},

    "ExtraTreesRegressor": {
        'class': ensemble.ExtraTreesRegressor,
        'params': {
            'n_estimators': {
                'type': 'slider',
                'label': 'Number of Estimators',
                'min': 10,
                'max': 500,
                'default': 100,
                'help': 'The number of trees in the forest. Increasing this generally improves performance but increases computation time.'
            },
            'max_depth': {
                'type': 'slider',
                'label': 'Max Depth',
                'min': 1,
                'max': 100,
                'default': 10,
                'help': 'The maximum depth of each tree. Shallower trees reduce overfitting but may underfit.'
            },
            'min_samples_split': {
                'type': 'slider',
                'label': 'Min Samples Split',
                'min': 2,
                'max': 50,
                'default': 2,
                'help': 'The minimum number of samples required to split an internal node. Higher values reduce overfitting.'
            },
            'max_features': {
                'type': 'slider',
                'label': 'Max Features',
                'min': 0.1,
                'max': 1.0,
                'default': 1.0,
                'help': 'Fraction of features considered at each split. Lower values may improve generalization but risk underfitting.'
            }
        },
        'multi_output': True,  # Native multi-output support
        'supports_sample_weight': True,
        'notes': 'Randomized decision trees; excellent for high-dimensional data. Tuning max_features is critical to avoid overfitting.'
    },

    "ElasticNet": {
    'class': linear_model.ElasticNet,
    'params': {
        'alpha': {
            'type': 'slider',
            'label': 'Alpha (Regularization Strength)',
            'min': 0.01,
            'max': 100.0,
            'default': 1.0,
            'help': 'Controls the combined regularization strength of L1 and L2 penalties. Higher values increase regularization.'
        },
        'l1_ratio': {
            'type': 'slider',
            'label': 'L1 Ratio',
            'min': 0.0,
            'max': 1.0,
            'default': 0.5,
            'help': 'Proportion of L1 regularization in the combined penalty. 0 = Ridge, 1 = Lasso, values in between are a mix.'
        },
        'max_iter': {
            'type': 'slider',
            'label': 'Max Iterations',
            'min': 100,
            'max': 10000,
            'default': 1000,
            'help': 'Maximum number of iterations for the optimization algorithm. Increase if the model does not converge.'
        }
    },
    'multi_output': False,  # Requires MultiOutputRegressor wrapper
    'supports_sample_weight': True,
    'notes': 'Combines L1 and L2 penalties for balanced regularization. Suitable for sparse datasets.'
},

    "RandomForestRegressor": {
    'class': ensemble.RandomForestRegressor,
    'params': {
        'n_estimators': {
            'type': 'slider',
            'label': 'Number of Estimators',
            'min': 10,
            'max': 500,
            'default': 100,
            'help': 'The number of trees in the forest. More trees typically improve performance but increase computation time.'
        },
        'max_depth': {
            'type': 'slider',
            'label': 'Max Depth',
            'min': 1,
            'max': 100,
            'default': 10,
            'help': 'Limits the depth of each tree. Shallower trees generalize better but may underfit.'
        },
        'min_samples_split': {
            'type': 'slider',
            'label': 'Min Samples Split',
            'min': 2,
            'max': 50,
            'default': 2,
            'help': 'Minimum number of samples required to split an internal node. Higher values reduce overfitting.'
        },
        'max_features': {
            'type': 'slider',
            'label': 'Max Features',
            'min': 0.1,
            'max': 1.0,
            'default': 1.0,
            'help': 'Fraction of features considered at each split. Lower values reduce overfitting but may underfit.'
        }
    },
    'multi_output': True,  # Native multi-output support
    'supports_sample_weight': True,
    'notes': 'Ensemble method with bagging and random feature selection; works well on high-dimensional data.'
},

    "XGBRegressor": {
    'class': xgb.XGBRegressor if xgb is not None else _MissingLibrary('xgboost'),
    'params': {
        'n_estimators': {
            'type': 'slider',
            'label': 'Number of Estimators',
            'min': 10,
            'max': 500,
            'default': 100,
            'help': 'The number of boosting rounds or trees to build. Higher values improve performance but increase computation time.'
        },
        'max_depth': {
            'type': 'slider',
            'label': 'Max Depth',
            'min': 1,
            'max': 100,
            'default': 6,
            'help': 'The maximum depth of each tree. Deeper trees can capture complex patterns but may overfit.'
        },
        'learning_rate': {
            'type': 'slider',
            'label': 'Learning Rate',
            'min': 0.01,
            'max': 1.0,
            'default': 0.3,
            'help': 'Step size for updating weights in each boosting round. Smaller values slow training but improve accuracy.'
        },
        'subsample': {
            'type': 'slider',
            'label': 'Subsample',
            'min': 0.1,
            'max': 1.0,
            'default': 1.0,
            'help': 'Proportion of training samples used in each boosting round. Helps prevent overfitting.'
        },
        'colsample_bytree': {
            'type': 'slider',
            'label': 'Colsample by Tree',
            'min': 0.1,
            'max': 1.0,
            'default': 1.0,
            'help': 'Fraction of features used to build each tree. Lower values may improve generalization.'
        },
        'min_child_weight': {
            'type': 'slider',
            'label': 'Min Child Weight',
            'min': 1,
            'max': 10,
            'default': 1,
            'help': 'Minimum sum of instance weights in a leaf node. Higher values prevent overfitting in small datasets.'
        },
        'gamma': {
            'type': 'slider',
            'label': 'Gamma',
            'min': 0.0,
            'max': 10.0,
            'default': 0.0,
            'help': 'Minimum loss reduction required for splitting a node. Larger values make the algorithm more conservative.'
        },
        'reg_lambda': {
            'type': 'slider',
            'label': 'L2 Regularization (Lambda)',
            'min': 0.0,
            'max': 10.0,
            'default': 1.0,
            'help': 'L2 penalty for weights. Helps control model complexity and prevent overfitting.'
        },
        'reg_alpha': {
            'type': 'slider',
            'label': 'L1 Regularization (Alpha)',
            'min': 0.0,
            'max': 10.0,
            'default': 0.0,
            'help': 'L1 penalty for weights. Higher values lead to sparser models, reducing complexity.'
        },
        'early_stopping_rounds': {
            'type': 'slider',
            'label': 'Early Stopping Rounds',
            'min': 0,
            'max': 100,
            'default': 10,
            'help': 'Stops training if the evaluation metric does not improve after these rounds. Helps avoid overfitting.'
        },
        'scale_pos_weight': {
            'type': 'slider',
            'label': 'Scale Positive Weight',
            'min': 0.0,
            'max': 100.0,
            'default': 1.0,
            'help': 'Controls the balance of positive and negative weights. Useful for handling class imbalance in the dataset.'
        }
    },
    'multi_output': False,  # Requires MultiOutputRegressor wrapper
    'supports_sample_weight': True,
    'notes': 'Highly versatile model with extensive parameter control. Best for complex and structured datasets.'
},

    "LGBMRegressor": {
    'class': lgb.LGBMRegressor if lgb is not None else _MissingLibrary('lightgbm'),
    'params': {
        'n_estimators': {
            'type': 'slider',
            'label': 'Number of Estimators',
            'min': 10,
            'max': 500,
            'default': 100,
            'help': 'Number of boosting rounds. Higher values increase accuracy but require more computation.'
        },
        'max_depth': {
            'type': 'slider',
            'label': 'Max Depth',
            'min': -1,
            'max': 20,
            'default': -1,
            'help': 'Maximum depth of trees. Use -1 for no limit. Shallower trees are faster but may underfit.'
        },
        'learning_rate': {
            'type': 'slider',
            'label': 'Learning Rate',
            'min': 0.01,
            'max': 1.0,
            'default': 0.1,
            'help': 'Controls the step size in updating weights. Smaller values require more boosting rounds.'
        },
        'subsample': {
            'type': 'slider',
            'label': 'Subsample',
            'min': 0.1,
            'max': 1.0,
            'default': 1.0,
            'help': 'Fraction of training data used for each boosting step. Helps prevent overfitting.'
        },
        'colsample_bytree': {
            'type': 'slider',
            'label': 'Feature Fraction',
            'min': 0.1,
            'max': 1.0,
            'default': 1.0,
            'help': 'Fraction of features randomly selected per tree. Reduces overfitting in high-dimensional datasets.'
        },
        'min_child_samples': {
            'type': 'slider',
            'label': 'Min Child Samples',
            'min': 1,
            'max': 50,
            'default': 20,
            'help': 'Minimum number of samples required in a leaf. Larger values prevent overfitting.'
        },
        'reg_lambda': {
            'type': 'slider',
            'label': 'L2 Regularization (Lambda)',
            'min': 0.0,
            'max': 10.0,
            'default': 0.0,
            'help': 'L2 penalty for weights. Controls overfitting by penalizing large weights.'
        },
        'reg_alpha': {
            'type': 'slider',
            'label': 'L1 Regularization (Alpha)',
            'min': 0.0,
            'max': 10.0,
            'default': 0.0,
            'help': 'L1 penalty for weights. Encourages sparsity in the model.'
        },
        'feature_pre_filter': {
            'type': 'checkbox',
            'label': 'Feature Pre-Filter',
            'default': True,
            'help': 'If enabled, LightGBM filters features with no predictive power early. Can speed up training.'
        }
    },
    'multi_output': False,  # Requires MultiOutputRegressor wrapper
    'supports_sample_weight': True,
    'notes': 'Efficient for large datasets and high-dimensional data. Supports histogram-based learning for faster training.'
},

}

def get_model_config(model_name, problem_type='classification'):
    """
    Get configuration for a specific model.
    
    Parameters
    ----------
    model_name : str
        Name of the model
    problem_type : {'classification', 'regression'}
        Type of problem
    
    Returns
    -------
    dict
        Model configuration
    """
    if problem_type == 'classification':
        models_dict = CLASSIFICATION_MODELS
    else:
        models_dict = REGRESSION_MODELS
    
    if model_name not in models_dict:
        raise ValueError(f"Model '{model_name}' not found in {problem_type} models")
    
    return models_dict[model_name]


def list_models(problem_type='classification'):
    """
    List available models for a problem type.
    
    Parameters
    ----------
    problem_type : {'classification', 'regression'}
    
    Returns
    -------
    list
        Model names
    """
    if problem_type == 'classification':
        return list(CLASSIFICATION_MODELS.keys())
    else:
        return list(REGRESSION_MODELS.keys())