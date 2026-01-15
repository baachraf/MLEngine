from setuptools import setup, find_packages

setup(
    name="ML_Engine",
    version="0.1.0",
    author="ACHRAF BEN AHMED",
    description="A comprehensive toolkit for building and evaluating machine learning pipelines.",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "scipy",
    ],
    extras_require={
        "automl": ["pycaret", "h2o"],
        "feature_selection": ["boruta", "xgboost"],
        "imbalance": ["imbalanced-learn", "ImbalancedLearningRegression"],
        "uncertainty": ["mapie"],
        "plotting": ["matplotlib", "seaborn"],
        "all": [
            "pycaret", "h2o", "boruta", "xgboost", 
            "imbalanced-learn", "ImbalancedLearningRegression", 
            "mapie", "matplotlib", "seaborn"
        ]
    },
    python_requires=">=3.8",
)
