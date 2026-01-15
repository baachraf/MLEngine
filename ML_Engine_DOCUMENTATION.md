# ML_Engine Library Quick Reference

A modular Python library for building and evaluating machine learning pipelines.

## Project Structure

This tree shows the organization of the library. Each file has a comment describing its primary role.

```
ML_Modules/
└── ML_Engine/
    ├── __init__.py
    ├── data/
    │   ├── __init__.py
    │   ├── cleaning.py       #-> Functions for handling nulls, duplicates, etc.
    │   ├── imbalance.py      #-> Functions for handling imbalanced datasets.
    │   ├── io.py             #-> Functions for reading and writing data.
    │   └── transformation.py #-> Functions for scaling and encoding data.
    ├── evaluation/
    │   ├── __init__.py
    │   └── metrics.py        #-> Functions for calculating model performance metrics.
    ├── features/
    │   ├── __init__.py
    │   ├── reduction.py      #-> Functions for dimensionality reduction (PCA, t-SNE).
    │   └── selection.py      #-> Functions for feature selection (RFE, Boruta).
    ├── models/
    │   ├── __init__.py
    │   ├── automl.py         #-> Functions for running PyCaret and H2O.
    │   ├── configs.py        #-> Pre-defined configurations for all models.
    │   └── training.py       #-> Core functions for training and tuning models.
    ├── utils/
    │   ├── __init__.py
    │   ├── common.py         #-> General helper functions.
    │   └── docs.py           #-> Script for generating markdown documentation.
    └── visualization/
        ├── __init__.py
        ├── categorical.py    #-> Functions for plotting categorical data (bar, count).
        ├── distribution.py   #-> Functions for plotting single variable distributions (hist, kde).
        ├── evaluation.py     #-> Functions for plotting model evaluation results (confusion matrix, ROC).
        ├── features.py       #-> Functions for plotting feature characteristics (variance).
        ├── matrix.py         #-> Functions for plotting matrix data (heatmap).
        ├── relationship.py   #-> Functions for plotting relationships between variables (scatter, line).
        └── style.py          #-> Functions for managing plot aesthetics.
```

## Key Functions by Task

This index helps you find the right function for a specific job.

### Data Handling
- **Load Data**: `ML_Engine.data.io.load_data(file_path, ...)`
- **Save Data (JSON)**: `ML_Engine.data.io.save_json(data, file_path)`
- **Clean Data**: `ML_Engine.data.cleaning.clean_data(df, ...)`
- **Split Data**: `ML_Engine.data.io.split_data_train_test(df, ...)`
- **Handle Imbalance (Classification)**: `ML_Engine.data.imbalance.handle_classification_imbalance(X, y, ...)`
- **Handle Imbalance (Regression)**: `ML_Engine.data.imbalance.handle_regression_imbalance(df, ...)`

### Transformation & Encoding
- **Scale Data**: `ML_Engine.data.transformation.apply_scaling(data, scaler_type)`
- **Label Encode**: `ML_Engine.data.transformation.apply_label_encoding(df, columns)`
- **One-Hot Encode**: `ML_Engine.data.transformation.apply_one_hot_encoding(df, columns)`

### Feature Engineering & Selection
- **Select K-Best Features**: `ML_Engine.features.selection.select_k_best_features(X, y, ...)`
- **Recursive Feature Elimination (RFE)**: `ML_Engine.features.selection.recursive_feature_elimination(X, y, ...)`
- **Boruta Feature Selection**: `ML_Engine.features.selection.boruta_feature_selection(X, y, ...)`
- **Apply PCA**: `ML_Engine.features.reduction.apply_pca(X, ...)`

### Model Training & Management
- **Train a Model**: `ML_Engine.models.training.train_model(model_name, X_train, y_train, ...)`
- **List Available Models**: `ML_Engine.models.configs.list_models(problem_type)`
- **Get Model Configuration**: `ML_Engine.models.configs.get_model_config(model_name, problem_type)`
- **Save Model**: `ML_Engine.models.training.save_model(model, filepath, ...)`
- **Load Model**: `ML_Engine.models.training.load_model(filepath)`

### AutoML
- **Run PyCaret**: `ML_Engine.models.automl.run_pycaret_automl(train_df, ...)`
- **Run H2O**: `ML_Engine.models.automl.run_h2o_automl(X_train, y_train, ...)`

### Evaluation
- **Calculate Metrics**: `ML_Engine.evaluation.metrics.get_metric_result(y_true, y_pred, ...)`

### Visualization
- **Set Global Style**: `ML_Engine.visualization.style.set_style(...)`
- **Plot Histogram**: `ML_Engine.visualization.distribution.plot_histogram(data, x, ...)`
- **Plot Scatter**: `ML_Engine.visualization.relationship.plot_scatter(data, x, y, ...)`
- **Plot Bar**: `ML_Engine.visualization.categorical.plot_barplot(data, x, y, ...)`
- **Plot Heatmap**: `ML_Engine.visualization.matrix.plot_heatmap(data, ...)`
- **Plot Confusion Matrix**: `ML_Engine.visualization.evaluation.plot_confusion_matrix(y_true, y_pred, ...)`
- **Plot ROC Curve**: `ML_Engine.visualization.evaluation.plot_roc_curve(y_true, y_prob, ...)`
