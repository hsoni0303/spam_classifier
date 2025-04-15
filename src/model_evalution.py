import os
import numpy as np
import pandas as pd
import pickle
import json
import logging 
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from dvclive import Live
import yaml

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_evalution')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_evalution.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path):
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameter load successful')
        return params
    except Exception as e:
        logger.error("Issues with loading parameters %s", e)
        raise

def load_model(file_path):
    """Load trained model from a file"""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', file_path)
        return model
    except Exception as e:
        logger.error('Unexpected error while loading the model: %s', e)
        raise

def load_data(file_path):
    """Load data from CSV file"""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from: %s', file_path)
        return df
    except Exception as e:
        logger.error("Unexpected error occured while loading data: %s", e)
        raise

def evaluate_model(clf, X_test, y_test):
    """Evalute model and return evalution metrics."""
    try:
        y_pred = clf.predict(X_test)
        y_pred_prob = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_prob)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }

        logger.debug('Model evalution metrics successfully calculated.')
        return metrics_dict
    except Exception as e:
        logger.error('Error whle evaluting model : %s', e)
        raise

def save_metrics(metrics, file_path):
    """Save metrics dict to JSON."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error("Error occurred while saving the metrics: %s", e)
        raise

def main():
    try:
        params = load_params('params.yaml')
        clf = load_model('./model/model.pkl')
        test_data = load_data('./data/processed/test_tfidf.csv')

        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics = evaluate_model(clf, X_test, y_test)
        # Experiment tracking using DVC live
        with Live(save_dvc_exp=True) as live:
            live.log_metric('accuracy', metrics['accuracy'])
            live.log_metric('precision', metrics['precision'])
            live.log_metric('Recall', metrics['recall'])
            live.log_params(params)

        save_metrics(metrics, 'report/metrics.json')
        
    except Exception as e:
        logger.error('Failed to complete the model eavalution process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()