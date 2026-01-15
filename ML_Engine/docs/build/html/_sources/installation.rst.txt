Installation
============

This guide provides instructions for installing the ``ML_Engine`` library and its dependencies.

Core Dependencies
-----------------

The core library requires a set of essential packages to function. You can install these using the ``requirements.txt`` file located in the root of the project.

.. code-block:: sh

   pip install -r requirements.txt


Optional Dependencies
---------------------

For advanced features like AutoML, specialized feature selection, and plotting, you need to install additional packages. These are not included in the main ``requirements.txt`` file to keep the core library lightweight.

AutoML
~~~~~~

- **PyCaret**: For running AutoML experiments with PyCaret.

  .. code-block:: sh

     pip install pycaret

- **H2O**: For running AutoML experiments with H2O.

  .. code-block:: sh

     pip install h2o

Advanced Feature Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Boruta**: For using the Boruta feature selection algorithm.

  .. code-block:: sh

     pip install boruta

- **XGBoost**: For using XGBoost-based feature importance.

  .. code-agressive: sh

     pip install xgboost

Imbalance Handling
~~~~~~~~~~~~~~~~~~

- **imbalanced-learn**: For most classification imbalance techniques (SMOTE, ADASYN, etc.).

  .. code-block:: sh

     pip install imbalanced-learn

- **ImbalancedLearningRegression**: For regression-specific imbalance techniques.

  .. code-block:: sh

     pip install ImbalancedLearningRegression

Uncertainty Quantification
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **MAPIE**: For calculating prediction intervals.

  .. code-block:: sh

     pip install mapie

Plotting
~~~~~~~~

- **Matplotlib & Seaborn**: For generating visualizations.

  .. code-block:: sh

     pip install matplotlib seaborn
