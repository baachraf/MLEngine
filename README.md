# ML Engine

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/your-username/your-repo)
[![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen)](https://github.com/your-username/your-repo)
[![PyPI version](https://img.shields.io/pypi/v/ml_engine.svg)](https://pypi.python.org/pypi/ml_engine)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

`ML Engine` is a comprehensive and modular Python library designed to streamline the entire machine learning workflow. From initial data loading and cleaning to advanced feature engineering, model training, and visualization, `ML_Engine` provides a robust toolkit for both rapid prototyping and building production-ready ML pipelines.

This project includes both the `ML_Engine` library and a powerful, config-driven pipeline runner, `run_pipeline.py`, for executing end-to-end experiments.

---

## Library Structure

| Sub-package                                     | Description                                                                                                                                                                                          |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [`data`](./ML_Engine/data)                      | **Data Handling & Preparation.** Contains modules for `io`, `cleaning`, data `transformation` (scaling, encoding), and `imbalance` handling.                                                              |
| [`features`](./ML_Engine/features)              | **Feature Engineering.** Includes modules for feature `selection` (RFE, Boruta) and dimensionality `reduction` (PCA, t-SNE).                                                                              |
| [`models`](./ML_Engine/models)                  | **Model Training & Management.** Provides a unified `training` interface to train a single model, compare multiple models, or run full AutoML pipelines using `PyCaret` and `H2O` (optional, not tested) backends. Includes advanced `Optuna` hyperparameter tuning. |
| [`evaluation`](./ML_Engine/evaluation)          | **Model Evaluation.** Contains tools for calculating performance `metrics` for classification and regression.                                                                                            |
| [`visualization`](./ML_Engine/visualization)    | **Plotting Toolkit.** A flexible library to visualize `distribution`, `relationship`, `categorical`, `matrix`, and model `evaluation` plots.                                                              |
| [`utils`](./ML_Engine/utils)                    | **Utilities.** Contains `common` helper functions and scripts for `docs` generation.                                                                                                                   |

---

## Example Notebooks

The `examples/` directory contains comprehensive Jupyter notebooks demonstrating key ML_Engine capabilities. Most notebooks use the **Adult Census dataset**, while Notebook 5 demonstrates financial time series analysis with real stock data. All notebooks include inline visualization and automatic plot saving to the `outputs/` directory.

| Notebook | Description | Key Features |
|----------|-------------|--------------|
| [`00_Dataset_Preparation.ipynb`](./examples/00_Dataset_Preparation.ipynb) | Data loading, cleaning, and preparation workflow. | Data I/O, cleaning, transformation, imbalance handling |
| [`01_Full_Classification_Pipeline.ipynb`](./examples/01_Full_Classification_Pipeline.ipynb) | End-to-end classification pipeline. | Feature selection, model comparison, hyperparameter tuning, confusion matrix visualization |
| [`02_Feature_Selection_Deep_Dive.ipynb`](./examples/02_Feature_Selection_Deep_Dive.ipynb) | Comprehensive feature engineering exploration. | Variance threshold, K-best, RFE, Boruta, PCA, t-SNE visualization |
| [`03_Hyperparameter_Tuning_Deep_Dive.ipynb`](./examples/03_Hyperparameter_Tuning_Deep_Dive.ipynb) | Advanced hyperparameter optimization. | Manual tuning, Optuna automated tuning, optimization history visualization |
| [`04_Full_Regression_Pipeline.ipynb`](./examples/04_Full_Regression_Pipeline.ipynb) | Complete regression workflow. | Regression model comparison, R² scoring, actual vs predicted visualization |
| [`05_EDA_and_Time_Series_Analysis.ipynb`](./examples/05_EDA_and_Time_Series_Analysis.ipynb) | Advanced financial time series analysis with real stock data. | yfinance integration, returns calculation, volatility analysis, date/lag/rolling features, stationarity testing, time series splitting |

### Running the Examples

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Launch Jupyter**: `jupyter notebook`
3. **Open any notebook** in the `examples/` directory
4. **Restart the kernel** and run all cells (notebooks have been fixed for the Adult Census dataset)

**Note**: Notebook 5 requires `yfinance` for downloading real stock data, which is included in `requirements.txt`.

**Dataset**: Most notebooks use the Adult Census dataset (`dataset/adult_census_sample.csv`) with:
- Classification target: `income` (binary classification)
- Regression target: `hours-per-week` (continuous regression)

**Notebook 5** uses real-time financial data downloaded via `yfinance` (AAPL stock data by default).

**Plot Saving**: Every visualization automatically saves to `outputs/{notebook_name}/` for reproducible analysis.

---

## Automated Pipeline Runner

The primary way to use this project is through the `run_pipeline.py` script, which orchestrates the entire ML workflow based on configuration files.

### Configuration System

The pipeline uses a modular configuration system where different YAML files define specific aspects of the ML workflow:

#### **Configuration File Structure**

| File | Purpose | Example Use |
|------|---------|-------------|
| **`configs/pipeline.yml`** | Main experiment definitions | Define data source, feature selection, training, and evaluation |
| **`configs/experiments.yml`** | Model comparison experiments | Pre-defined sets of models to train with specific parameters |
| **`configs/feature_selection_experiments.yml`** | Feature selection experiments | Methods and parameters for feature selection |
| **`configs/model_defaults.yml`** | Default model parameters | Library-wide defaults for all scikit-learn models |
| **`configs/preprocessing_pipelines.yml`** | Preprocessing configurations | Data cleaning and transformation settings |
| **`configs/tuning_spaces.yml`** | Hyperparameter tuning spaces | Search spaces for Optuna hyperparameter optimization |

#### **How to Create a New Pipeline Experiment**

1. **Create a new experiment in `pipeline.yml`**:
   ```yaml
   your_experiment_name:
     description: "Description of your experiment"
     
     data:
       source: "path/to/your/data.csv"
       target_columns: ['target_column']
       drop_columns: ['unneeded_column']
     
     feature_selection:
       run: true  # or false to skip
       experiment: 'experiment_name_from_feature_selection_experiments.yml'
     
     training:
       backend: 'scikit-learn'  # or 'pycaret'
       problem_type: 'Classification'  # or 'Regression'
       experiment: 'experiment_name_from_experiments.yml'
     
     evaluation:
       metrics: ['Accuracy', 'F1 Score']  # Classification metrics
       # or: ['R2', 'MAE', 'MSE']  # Regression metrics
   ```

2. **Create model experiments in `experiments.yml`** (optional - use existing ones):
   ```yaml
   your_model_experiment:
     description: "Your model comparison experiment"
     problem_type: "Classification"  # or "Regression"
     sort_by: "Accuracy"  # Metric to sort results by
     models:
       - name: "RandomForestClassifier"
         params:
           n_estimators: 100
           max_depth: 10
       - name: "LogisticRegression"
         params:
           C: 1.0
     # Set models: null to use ALL available models for the problem type
   ```

3. **Create feature selection experiments in `feature_selection_experiments.yml`** (optional):
   ```yaml
   your_feature_selection_experiment:
     description: "Your feature selection experiment"
     methods:
       - name: "variance"
         params:
           threshold: 0.01
       - name: "kbest"
         params:
           k: 10
           score_func: "f_classif"  # Use "f_regression" for regression
   ```

4. **Reference existing experiments** from `experiments.yml` and `feature_selection_experiments.yml`, or create your own as shown above.

5. **Use relative paths** for data sources, relative to the project root (parent of `configs/` directory).

#### **Available Experiments in `pipeline.yml`**

| Experiment Name | Type | Description | Dependencies | Command |
|-----------------|------|-------------|--------------|---------|
| `simple_regression_test` | Regression | Baseline linear regression test | None | `python examples/run_pipeline.py --experiment simple_regression_test` |
| `advanced_classification_pipeline` | Classification | Compare classifiers with feature selection | None | `python examples/run_pipeline.py --experiment advanced_classification_pipeline` |
| `full_classification_suite` | Classification | Run all available classification models* | All scikit-learn classifiers | `python examples/run_pipeline.py --experiment full_classification_suite` |
| `classification_boruta` | Classification | Classification with Boruta feature selection | `boruta` package | `python examples/run_pipeline.py --experiment classification_boruta` |
| `robust_regression` | Regression | Robust regressors with feature selection | None | `python examples/run_pipeline.py --experiment robust_regression` |
| `regression_boruta` | Regression | Regression with Boruta feature selection | `boruta` package | `python examples/run_pipeline.py --experiment regression_boruta` |
| `pycaret_automl_run` | Classification | PyCaret AutoML | `pycaret` package | `python examples/run_pipeline.py --experiment pycaret_automl_run` |

**Notes:**
- *`full_classification_suite`: Currently limited due to model configuration loading issues. Works best with explicitly defined model sets.
- Boruta experiments (`classification_boruta`, `regression_boruta`): Require `pip install boruta`
- PyCaret experiment: Requires `pip install pycaret`
- H2O AutoML: Functionality is implemented but not included in pre-configured pipeline experiments; requires manual configuration.
- All experiments work with basic scikit-learn installation

### How to Use

**1. Define Your Pipeline in `configs/pipeline.yml`**

This master configuration file defines every step of your experiment, from data source to evaluation.

**2. Run the Pipeline from Your Terminal**

Execute the script from the `ML_Modules` root directory, specifying which experiment you want to run. First ensure the ML_Engine package is installed (see Installation section).

```sh
# Run the 'advanced_classification_pipeline' experiment using all available CPU cores
python examples/run_pipeline.py --experiment advanced_classification_pipeline

# Run the same experiment but limit it to 4 parallel jobs
python examples/run_pipeline.py --experiment advanced_classification_pipeline --n_jobs 4

# You can also specify a custom config file path
python examples/run_pipeline.py --experiment simple_regression_test --config configs/pipeline.yml
```

**3. Get Your Results**

The pipeline runner automatically creates a unique, timestamped directory for each run inside the `examples/outputs/pipeline/` folder. The directory structure is organized by config file, experiment name, and timestamp: `examples/outputs/pipeline/{config_name}/{experiment_name}/{timestamp}/`. This directory contains all the artifacts needed for reproducibility:

-   `final_report.json`: A comprehensive leaderboard of all model results.
-   `feature_sets.json`: The lists of features selected by each method.
-   **Sub-folders for each pipeline branch**, containing:
    -   The trained model file (e.g., `model_RandomForestClassifier.pkl`).
    -   The fitted scaler (`scaler.pkl`).
    -   The list of features used (`selected_features.json`).

### Troubleshooting

- **ModuleNotFoundError: No module named 'ML_Engine'**: The script includes path adjustments to import ML_Engine from the local directory. Ensure you run from the `ML_Modules` directory or have the package installed (`pip install -e .`). Installation is recommended for dependency management but not required for the pipeline script.

- **FileNotFoundError for config or data files**: The script resolves data paths relative to the project root (parent of the configs directory). Config file paths (`experiments.yml`, `feature_selection_experiments.yml`) are resolved relative to the location of the main pipeline config file. Ensure your configuration file uses relative paths that are correct from the project root.

- **ValueError: could not convert string to float**: The pipeline now automatically encodes categorical columns. If you still encounter this error, check that categorical columns are properly identified (object or category dtype).

---

## Installation

### 1. Clone the Repository

```sh
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Install Dependencies

Install the core packages and any optional extras you need for your experiments.

```sh
# Install core requirements
pip install -r requirements.txt

# Install the library in editable mode
pip install -e .

# To install all optional packages for advanced features
pip install -e .[all]
```

---

## Documentation & Testing

### Documentation

The full library documentation is built with Sphinx.

1.  **Navigate to the docs directory:** `cd ML_Engine/docs`
2.  **Build the HTML:** `.\\make.bat html` (on Windows) or `make html` (on macOS/Linux)
3.  **View the documentation** by opening `build/html/index.html`.

### Running Tests

The project uses `pytest` for testing the library's components.

1.  **Install pytest:** `pip install pytest`
2.  **Run the test suite** from the root `ML_Modules` directory: `pytest`

---

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
