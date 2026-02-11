"""
ML Engine Pipeline Runner
=========================

This script is the main entry point for running end-to-end machine learning
pipelines defined in a YAML configuration file.
"""

import argparse
import yaml
import os
import sys
import pandas as pd
import pickle
import json
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from datetime import datetime

# Add parent directory to sys.path to allow importing ML_Engine when running as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ML_Engine.data import io, transformation
from ML_Engine.features import selection
from ML_Engine.models import training
from ML_Engine.models.configs import list_models
from ML_Engine.models.automl import run_pycaret_automl
from ML_Engine.evaluation import metrics as eval_metrics
from ML_Engine.utils.logger import get_logger

logger = get_logger("PipelineRunner")

def process_pipeline_branch(fs_name, fs_list, X_full, y_full, training_config, config_dir, output_dir):
    """
    Executes the training and evaluation for a single feature set branch and saves all artifacts.
    """
    branch_dir = os.path.join(output_dir, f"{fs_name}_branch")
    os.makedirs(branch_dir, exist_ok=True)
    logger.info(f"--- Starting branch for feature set: '{fs_name}'. Artifacts will be saved in: {branch_dir} ---")

    # --- Data Transformation ---
    X_selected = transformation.apply_feature_selection(X_full, fs_list)
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y_full, test_size=0.2, random_state=42)

    backend = training_config['backend']
    
    if backend == 'scikit-learn':
        # --- Encode categorical columns ---
        # Identify categorical columns (object or category dtype)
        cat_columns = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_columns:
            logger.info(f"Encoding categorical columns: {cat_columns}")
            # Apply label encoding to training data and get mappings
            X_train_encoded, mappings = transformation.apply_label_encoding(X_train, cat_columns)
            # Apply same mappings to test data
            X_test_encoded = transformation.apply_saved_label_encoding(X_test, cat_columns, mappings)
            # Replace data with encoded versions
            X_train = X_train_encoded
            X_test = X_test_encoded
            logger.info(f"Categorical columns encoded using label encoding")
        else:
            logger.info("No categorical columns to encode")
        
        # --- Scaling ---
        scaler_type = training_config.get('scaler', 'StandardScaler')
        X_train_scaled, scaler = transformation.fit_and_apply_scaling(X_train, scaler_type)
        X_test_scaled = transformation.apply_scaler(X_test, scaler)
        
        # Save the scaler
        with open(os.path.join(branch_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
        # Save the list of selected features
        with open(os.path.join(branch_dir, 'selected_features.json'), 'w') as f:
            json.dump(fs_list, f, indent=4)
        
        # --- Model Training ---
        exp_results = []
        exp_config_path = os.path.join(config_dir, 'experiments.yml')
        with open(exp_config_path, 'r') as f:
            experiments = yaml.safe_load(f)
        
        exp_details = experiments[training_config['experiment']]
        
        # Handle case where models is null (means use all available models)
        if exp_details.get('models') is None:
            # Get all available models for this problem type
            all_model_names = list_models(training_config['problem_type'])
            model_specs = [{'name': model_name, 'params': {}} for model_name in all_model_names]
        else:
            model_specs = exp_details['models']
        
        for model_spec in model_specs:
            model_name = model_spec['name']
            model_params = model_spec.get('params', {})
            
            logger.info(f"Training {model_name} on feature set '{fs_name}'...")
            try:
                # Extract tuning parameters from training_config
                tuning_method = training_config.get('tuning_method')
                tuning_config = training_config.get('tuning_config')
                tuning_metric = training_config.get('tuning_metric')
                n_trials = training_config.get('n_trials', 100)
                cv = training_config.get('cv', 5)
                
                model, training_info = training.train_model(
                    model_name=model_name,
                    X_train=X_train_scaled,
                    y_train=y_train,
                    problem_type=training_config['problem_type'],
                    model_params=model_params,
                    tuning_method=tuning_method,
                    tuning_config=tuning_config,
                    tuning_metric=tuning_metric,
                    n_trials=n_trials,
                    cv=cv
                )
            except (ImportError, RuntimeError) as e:
                logger.warning(f"Skipping model {model_name} due to missing dependency: {e}")
                continue
            
            # Save the trained model
            training.save_model(model, os.path.join(branch_dir, f"model_{model_name}.pkl"))
            
            # Save training info (hyperparameter tuning results)
            try:
                with open(os.path.join(branch_dir, f'training_info_{model_name}.json'), 'w') as f:
                    json.dump(training_info, f, indent=4, default=str)
            except Exception as e:
                logger.warning(f"Failed to save training info for {model_name}: {e}")
            
            # Evaluate the model
            y_pred = model.predict(X_test_scaled)
            metrics_to_calc = ['R^2 Score', 'Mean Squared Error', 'Mean Absolute Error'] if training_config['problem_type'].lower() == 'regression' else ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            result = eval_metrics.get_metric_result(y_test, y_pred, training_config['problem_type'], metrics_to_calc, ['target'])
            
            flat_result = {'Model': model_name}
            flat_result.update(result['target'])
            exp_results.append(flat_result)
            
        model_results_df = pd.DataFrame(exp_results)
        model_results_df['feature_set'] = fs_name
        return model_results_df
    
    elif backend == 'pycaret':
        # Save the list of selected features (for reproducibility)
        with open(os.path.join(branch_dir, 'selected_features.json'), 'w') as f:
            json.dump(fs_list, f, indent=4)
        
        # Combine X and y for PyCaret (PyCaret expects target column in DataFrame)
        train_df = X_train.copy()
        train_df['_target'] = y_train.values if hasattr(y_train, 'values') else y_train
        
        # Extract PyCaret configuration
        sort_by = training_config.get('sort_by', 'Accuracy' if training_config['problem_type'].lower() == 'classification' else 'R2')
        pycaret_kwargs = training_config.get('pycaret_setup_args', {})
        
        logger.info(f"Running PyCaret AutoML on feature set '{fs_name}'...")
        try:
            best_model, pycaret_results = run_pycaret_automl(
                train_df=train_df,
                target_column='_target',
                problem_type=training_config['problem_type'],
                sort_by=sort_by,
                **pycaret_kwargs
            )
            
            # Save the best model
            training.save_model(best_model, os.path.join(branch_dir, 'model_pycaret_best.pkl'))
            
            # Save PyCaret results (leaderboard) as JSON
            leaderboard = pycaret_results.get('pycaret_results', {})
            if leaderboard:
                with open(os.path.join(branch_dir, 'pycaret_leaderboard.json'), 'w') as f:
                    json.dump(leaderboard, f, indent=4)
                
                # Convert leaderboard to final report format
                exp_results = []
                # leaderboard is a dict with column names as keys and lists as values
                # Convert to DataFrame for easier manipulation
                leaderboard_df = pd.DataFrame(leaderboard)
                # Ensure there's a 'Model' column (PyCaret uses 'Model' column)
                if 'Model' in leaderboard_df.columns:
                    for _, row in leaderboard_df.iterrows():
                        model_name = row['Model']
                        # Extract metrics - use all numeric columns except index
                        metric_cols = [col for col in leaderboard_df.columns if col != 'Model' and pd.api.types.is_numeric_dtype(leaderboard_df[col])]
                        flat_result = {'Model': model_name}
                        for col in metric_cols:
                            flat_result[col] = row[col]
                        exp_results.append(flat_result)
                else:
                    # Fallback: just include the best model
                    logger.warning("PyCaret leaderboard missing 'Model' column, reporting only best model")
                    # Evaluate best model on test set
                    y_pred = best_model.predict(X_test)
                    metrics_to_calc = ['R^2 Score', 'Mean Squared Error', 'Mean Absolute Error'] if training_config['problem_type'].lower() == 'regression' else ['Accuracy', 'Precision', 'Recall', 'F1 Score']
                    result = eval_metrics.get_metric_result(y_test, y_pred, training_config['problem_type'], metrics_to_calc, ['target'])
                    flat_result = {'Model': 'PyCaret_Best'}
                    flat_result.update(result['target'])
                    exp_results.append(flat_result)
                
                model_results_df = pd.DataFrame(exp_results)
                model_results_df['feature_set'] = fs_name
                return model_results_df
            else:
                logger.warning("PyCaret returned empty leaderboard")
                return pd.DataFrame()
                
        except ImportError as e:
            logger.error(f"PyCaret not installed: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"PyCaret AutoML failed: {e}")
            return pd.DataFrame()
    
    else:
        logger.warning(f"Backend '{backend}' is not yet fully implemented.")
        return pd.DataFrame()


def run_pipeline(experiment_name: str, config_path: str, n_jobs: int):
    """
    Runs a full ML pipeline based on a configuration file.
    """
    logger.info(f"Loading pipeline configuration from: {config_path}")
    config_dir = os.path.dirname(os.path.abspath(config_path))
    project_root = os.path.dirname(config_dir)
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)

    if experiment_name not in full_config:
        raise ValueError(f"Experiment '{experiment_name}' not found in {config_path}")

    exp_config = full_config[experiment_name]
    logger.info(f"Starting experiment: {exp_config.get('description', experiment_name)}")

    # --- Create unique output directory ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    output_dir = os.path.join(project_root, "examples", "outputs", "pipeline", experiment_name, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"All artifacts for this run will be saved in: {output_dir}")

    # --- 1. Data Loading ---
    logger.info("--- Step 1: Loading Data ---")
    data_config = exp_config['data']
    
    # Resolve data source path relative to project root (parent of configs directory)
    data_source = data_config['source']
    if not os.path.isabs(data_source):
        # config_dir is the directory containing the config file (e.g., ML_Modules/configs)
        # Project root is parent of config_dir (e.g., ML_Modules) - already computed
        data_source = os.path.join(project_root, data_source)
    logger.info(f"Loading data from: {data_source}")
    
    full_df = io.load_data(data_source)
    
    # Drop columns not needed for training
    if 'drop_columns' in data_config:
        full_df = full_df.drop(columns=data_config['drop_columns'], errors='ignore')

    X_full = full_df.drop(columns=data_config['target_columns'])
    y_full = full_df[data_config['target_columns'][0]]

    # --- 2. Feature Selection ---
    logger.info("--- Step 2: Running Feature Selection ---")
    fs_config = exp_config['feature_selection']
    feature_sets = {}
    if fs_config.get('run', False):
        fs_results = selection.run_feature_selection_experiment(
            experiment_name=fs_config['experiment'],
            X=X_full,
            y=y_full,
            problem_type=exp_config['training']['problem_type']
        )
        feature_sets.update(fs_results)
    else:
        logger.info("Skipping feature selection step.")
        feature_sets['all_features'] = X_full.columns.tolist()
    
    with open(os.path.join(output_dir, 'feature_sets.json'), 'w') as f:
        json.dump(feature_sets, f, indent=4)
    logger.info(f"Feature selection results saved to {os.path.join(output_dir, 'feature_sets.json')}")

    # --- 3. Parallel Model Training ---
    logger.info(f"--- Step 3: Running Model Training in Parallel (n_jobs={n_jobs}) ---")
    training_config = exp_config['training']
    
    # Filter out feature sets that are error messages
    filtered_items = []
    for fs_name, fs_list in feature_sets.items():
        if isinstance(fs_list, str) and fs_list.startswith('Error:'):
            logger.warning(f"Skipping feature set '{fs_name}' due to error: {fs_list}")
            continue
        filtered_items.append((fs_name, fs_list))
    
    all_results = Parallel(n_jobs=n_jobs)(
        delayed(process_pipeline_branch)(fs_name, fs_list, X_full, y_full, training_config, config_dir, output_dir)
        for fs_name, fs_list in filtered_items
    )

    # --- 4. Final Report ---
    logger.info("--- Step 4: Generating Final Report ---")
    if not all_results:
        logger.warning("No results were generated.")
        return

    final_report_df = pd.concat(all_results, ignore_index=True)
    
    if not final_report_df.empty:
        cols = ['feature_set'] + [col for col in final_report_df.columns if col != 'feature_set']
        final_report_df = final_report_df[cols]

    report_path = os.path.join(output_dir, "final_report.json")
    final_report_df.to_json(report_path, orient='records', indent=4)
    logger.info(f"Pipeline finished. Final report saved to: {report_path}")
    print(final_report_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Engine Pipeline Runner")
    parser.add_argument("--experiment", type=str, required=True, help="The name of the experiment to run.")
    parser.add_argument("--config", type=str, default="configs/pipeline.yml", help="Path to the pipeline config file.")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs. -1 uses all cores.")
    args = parser.parse_args()
    
    run_pipeline(experiment_name=args.experiment, config_path=args.config, n_jobs=args.n_jobs)
