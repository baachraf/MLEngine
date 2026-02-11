# ML_Engine Examples

This folder contains example scripts and notebooks demonstrating the ML_Engine library.

## Pipeline Runner

The main pipeline script (`run_pipeline.py`) executes predefined ML experiments from the configuration file (`configs/pipeline.yml`). It handles data loading, feature selection, model training, evaluation, and artifact saving in a single command.

### Quick Start

1. **Ensure you are in the correct environment** (Python 3.9–3.11 recommended for PyCaret compatibility):
   ```bash
   conda activate RGBMLops  # or your Python 3.10 environment
   ```

2. **Run an experiment**:
   ```bash
   python run_pipeline.py --experiment simple_regression_test --n_jobs 1
   ```

### Available Experiments

All experiments are defined in `configs/pipeline.yml`. The following are pre‑configured and validated:

| Experiment Name | Description | Models | Feature Selection |
|----------------|-------------|--------|-------------------|
| `simple_regression_test` | Basic regression with linear models | LinearRegression, Ridge, Lasso, ElasticNet | None |
| `advanced_classification_pipeline` | Compare classifiers across different feature sets | RandomForestClassifier, LogisticRegression, SVC | Variance, K‑Best, RFE |
| `pycaret_automl_run` | Let PyCaret handle everything (feature selection, model choice) | PyCaret’s full model library | PyCaret internal |
| `full_classification_suite` | Run all available classification models with defaults | 17 classifiers (GaussianProcess, XGBoost, LightGBM, etc.) | None |
| `classification_boruta` | Classification using only Boruta feature selection | RandomForestClassifier, LogisticRegression, SVC | Boruta |
| `robust_regression` | Compare robust regressors for noisy data | HuberRegressor, RANSACRegressor, TheilSenRegressor | Variance, K‑Best, RFE |
| `regression_boruta` | Regression using Boruta feature selection | LinearRegression, Ridge, Lasso, ElasticNet | Boruta |

### Command‑Line Arguments

```bash
python run_pipeline.py --experiment <experiment_name> [--n_jobs <num_parallel_jobs>] [--config <path_to_pipeline.yml>]
```

- `--experiment` (**required**): Name of the experiment as defined in `configs/pipeline.yml`
- `--n_jobs` (default: 1): Number of parallel jobs for model training (use -1 for all cores)
- `--config` (default: `configs/pipeline.yml`): Path to pipeline configuration file

### Output Structure

Each run creates a timestamped directory under `examples/outputs/pipeline/`:

```
examples/outputs/pipeline/
└── <experiment_name>/
    └── YYYYMMDD_HHMMSS/              # Timestamp of the run
        ├── feature_sets.json         # Selected features per method
        ├── final_report.json         # Aggregated performance metrics
        ├── <feature_set>_branch/     # Per‑feature‑set results
        │   ├── model_<name>.pkl      # Trained model (pickle)
        │   ├── training_report.json  # Detailed training metrics
        │   └── (additional artifacts)
        └── (for PyCaret runs)
            ├── model_pycaret_best.pkl
            └── pycaret_leaderboard.json
```

### Environment Requirements

The pipeline depends on several optional libraries. Install them with:

```bash
pip install pycaret==3.3.2 boruta xgboost lightgbm
```

**Important PyCaret notes**:
- PyCaret 3.3.2 works with Python 3.9–3.11 (Python 3.12 raises a `RuntimeError`).
- The `silent` and `feature_interaction` parameters are deprecated and have been removed from the configuration.
- `remove_outliers` is set to `false` by default to avoid row‑count mismatches; you can enable it but may need to handle index alignment.

### Troubleshooting

- **“PyCaret AutoML failed: Sort method not supported”**: Ensure `problem_type` is lowercase (`'classification'`/`'regression'`) in the config.
- **“Length of values (3800) does not match length of index (4000)”**: Set `remove_outliers: false` in `pycaret_setup_args`.
- **“ImportError: No module named 'boruta'”**: Install Boruta with `pip install boruta`.
- **“ImportError: No module named 'xgboost'”**: Install XGBoost with `pip install xgboost`.

### Jupyter Notebooks

The folder also contains standalone notebooks that illustrate specific library features:

- `00_Dataset_Preparation.ipynb` – Data loading, cleaning, and basic exploration
- `01_Full_Classification_Pipeline.ipynb` – End‑to‑end classification workflow
- `02_Feature_Selection_Deep_Dive.ipynb` – Comparing feature selection methods
- `03_Hyperparameter_Tuning_Deep_Dive.ipynb` – Grid search, random search, and Optuna tuning
- `04_Full_Regression_Pipeline.ipynb` – End‑to‑end regression workflow

These notebooks are self‑contained and can be run independently.

### Customizing Experiments

To create your own experiment, edit `configs/pipeline.yml` following the existing structure. You can:

1. Define a new experiment block with `data`, `feature_selection`, `training`, and `evaluation` sections.
2. Reference predefined model suites from `configs/experiments.yml`.
3. Use any feature‑selection method defined in `configs/feature_selection_experiments.yml`.
4. Adjust hyperparameters in `configs/model_defaults.yml`.

### Example: Run All Preconfigured Experiments

```bash
for exp in simple_regression_test advanced_classification_pipeline pycaret_automl_run full_classification_suite classification_boruta robust_regression regression_boruta; do
    python run_pipeline.py --experiment $exp --n_jobs 1
done
```

This will execute all seven experiments sequentially, each producing its own output directory.
