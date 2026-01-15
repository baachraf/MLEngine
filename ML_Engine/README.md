# ML Engine

`ML_Engine` is a Python library providing a comprehensive toolkit for building and evaluating machine learning pipelines, from data cleaning and feature engineering to model training and AutoML.

## Installation

### 1. Clone the Repository

```sh
git clone <your-repository-url>
cd <your-repository-directory>
```

### 2. Install Core Dependencies

The core library requires the following packages. You can install them using the provided `requirements.txt` file located in the root of the `ML_Modules` project.

```sh
pip install -r ../requirements.txt
```

### 3. Install the Library

To make the `ML_Engine` library importable in your projects, install it in editable mode from the `ML_Modules` directory:

```sh
pip install -e .
```
*(Note: This requires a `setup.py` file in the `ML_Modules` root, which we can create next if you don't have one.)*

## Optional Dependencies

For advanced features like AutoML, specialized feature selection, and uncertainty quantification, you need to install additional packages.

### AutoML

- **PyCaret**: For running AutoML experiments with PyCaret.
  ```sh
  pip install pycaret
  ```
- **H2O**: For running AutoML experiments with H2O.
  ```sh
  pip install h2o
  ```

### Advanced Feature Selection

- **Boruta**: For using the Boruta feature selection algorithm.
  ```sh
  pip install boruta
  ```
- **XGBoost**: For using XGBoost-based feature importance.
  ```sh
  pip install xgboost
  ```

### Imbalance Handling

- **imbalanced-learn**: For most classification imbalance techniques (SMOTE, ADASYN, etc.).
  ```sh
  pip install imbalanced-learn
  ```
- **ImbalancedLearningRegression**: For regression-specific imbalance techniques.
  ```sh
  pip install ImbalancedLearningRegression
  ```

### Uncertainty Quantification

- **MAPIE**: For calculating prediction intervals.
  ```sh
  pip install mapie
  ```

### Plotting

- **Matplotlib & Seaborn**: For generating visualizations.
  ```sh
  pip install matplotlib seaborn
  ```
