import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
import yaml

# set up logger 
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path):
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('loading parameters successfully.')
        return params
    except Exception as e:
        logger.error('Unexpected error while loading params %s', e)
        raise

def load_data(file_path):
    """Load data from csv file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data Loaded from %s with shape %s', file_path, df.shape)
        return df
    except Exception as e:
        logger.error("Unexpected error occured while loading the data: %s", e)
        raise

def train_model(X_train, y_train, params):
    """Train a RandomForest model."""
    try:
        logger.debug("Initializing Randomforest model with parameters: %s", params)
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=params['random_state'])

        logger.debug('model training started')
        clf.fit(X_train, y_train)
        logger.debug('Model training completed.')
        return clf
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise

def save_model(model, file_path):
    """Save trained model to a file"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug("Model saved successfully")
    except Exception as e:
        logger.error('Error occured while saving the model: %s', e)
        raise

def main():
    try:
        params = load_params(params_path='params.yaml')['model_training']
        train_data = load_data('./data/processed/train_tfidf.csv')
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        clf = train_model(X_train, y_train, params)
        model_save_path = 'model/model.pkl'
        save_model(clf, model_save_path)
        
    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()