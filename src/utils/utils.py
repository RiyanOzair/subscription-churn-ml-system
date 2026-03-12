import os
import sys
import dill
import yaml

from sklearn.metrics import roc_auc_score

from src.logger import logging
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved successfully at {file_path}")
    except Exception as e:
        logging.error(f"Error saving object: {str(e)}")
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            obj = dill.load(file_obj)
        logging.info(f"Object loaded successfully from {file_path}")
        return obj
    except Exception as e:
        logging.error(f"Error loading object: {str(e)}")
        raise CustomException(e, sys)


def load_yaml(file_path):

    with open(file_path, "r") as file:
        return yaml.safe_load(file)