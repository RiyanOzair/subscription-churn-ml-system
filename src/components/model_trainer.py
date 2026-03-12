import os
import sys

import mlflow
from mlflow.sklearn import log_model

import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score
)
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from src.logger import logging
from src.exception import CustomException
from src.constants import ARTIFACT_DIR, MODEL_DIR, MODEL_FILE_NAME
from src.utils.utils import load_object, save_object, load_yaml


from sklearn.model_selection import GridSearchCV


class ModelTrainer:
    def __init__(self, X, y, preprocessor):
        self.X = X
        self.y = y
        self.preprocessor = preprocessor
    
    def get_models(self):
        return {
            "LogisticRegression": LogisticRegression(random_state=42),
            "DecisionTree": DecisionTreeClassifier(random_state=42),
            "RandomForest": RandomForestClassifier(random_state=42),
            "GradientBoosting": GradientBoostingClassifier(random_state=42),
            "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        }
    
    def get_param_grids(self):
        return {
            "LogisticRegression": {
                "C": [0.01, 0.1, 1, 10],
                "penalty": ["l2"],
                "solver": ["lbfgs"]
            },
            "DecisionTree": {
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5, 10]
            },
            "RandomForest": {
                "n_estimators": [100, 200],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5]
            },
            "GradientBoosting": {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 5]
            },
            "XGBoost": {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 5]
            }
        }

    def train(self):
        try:
            config = load_yaml("configs/training_config.yaml")
            X_train, X_test, y_train, y_test = train_test_split(
                self.X,
                self.y,
                test_size=config["training"]["test_size"],
                random_state=config["training"]["random_state"]
            )
            import pandas as pd
            if not isinstance(X_train, pd.DataFrame):
                X_train = pd.DataFrame(X_train, columns=self.X.columns)
            if not isinstance(X_test, pd.DataFrame):
                X_test = pd.DataFrame(X_test, columns=self.X.columns)
            X_train_transformed = self.preprocessor.fit_transform(X_train)
            X_test_transformed = self.preprocessor.transform(X_test)
            models = self.get_models()
            mlflow.set_experiment(config["experiment"]["name"])
            best_model = None
            best_score = 0
            best_model_name = None
            for name, model in models.items():
                run = mlflow.start_run(run_name=name)
                try:
                    mlflow.log_dict(config, "training_config.yaml")
                    logging.info(f"Training {name}")
                    model.fit(X_train_transformed, y_train)
                    preds = model.predict(X_test_transformed)
                    if hasattr(model, 'predict_proba'):
                        preds_proba = model.predict_proba(X_test_transformed)[:, 1]
                    else:
                        preds_proba = preds
                    metrics = {
                        "roc_auc": roc_auc_score(y_test, preds_proba),
                        "accuracy": accuracy_score(y_test, preds),
                        "precision": precision_score(y_test, preds),
                        "recall": recall_score(y_test, preds)
                    }
                    mlflow.log_param("model_name", name)
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(metric_name, value)
                    param_grids = self.get_param_grids()
                    if metrics["roc_auc"] > config["tuning"]["threshold"] and name in param_grids:
                        logging.info(f"Starting hyperparameter tuning for {name}")
                        grid = GridSearchCV(
                            model,
                            param_grids[name],
                            cv=config["tuning"]["cv"],
                            scoring="roc_auc",
                            n_jobs=-1
                        )
                        grid.fit(X_train_transformed, y_train)
                        model = grid.best_estimator_
                        score = grid.best_score_
                        logging.info(f"Tuned {name} score: {score}")
                        mlflow.log_params(grid.best_params_)
                    log_model(model, artifact_path=name)
                    logging.info(f"{name} ROC-AUC: {metrics['roc_auc']}")
                    if metrics["roc_auc"] > best_score:
                        best_score = metrics["roc_auc"]
                        best_model = model
                        best_model_name = name
                finally:
                    mlflow.end_run()
            logging.info("Generating SHAP explanations")
            feature_names = self.preprocessor.get_feature_names_out()
            import shap
            # Handle GridSearchCV or direct estimator
            model_to_check = best_model
            if hasattr(best_model, 'best_estimator_'):
                model_to_check = best_model.best_estimator_
            tree_models = (RandomForestClassifier, GradientBoostingClassifier, XGBClassifier)
            linear_models = (LogisticRegression,)
            if isinstance(model_to_check, tree_models):
                explainer = shap.TreeExplainer(best_model)
            elif isinstance(model_to_check, linear_models):
                explainer = shap.LinearExplainer(best_model, X_train_transformed)
            else:
                explainer = shap.Explainer(best_model)
            sample_size = min(200, X_test_transformed.shape[0])
            X_sample = X_test_transformed[:sample_size]
            shap_values = explainer(X_sample)
            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=feature_names,
                show=False
            )
            plt.tight_layout()
            plot_dir = os.path.dirname("artifacts/shap_summary.png")
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = "artifacts/shap_summary.png"
            plt.savefig(plot_path)
            plt.close()
            mlflow.log_artifact(plot_path)
            logging.info(f"Best model: {best_model_name} with score {best_score}")
            os.makedirs(MODEL_DIR, exist_ok=True)
            model_path = os.path.join(MODEL_DIR, MODEL_FILE_NAME)
            save_object(model_path, best_model)
            logging.info("Best model saved")
            
            # Save the fitted preprocessor
            preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.pkl")
            save_object(preprocessor_path, self.preprocessor)
            logging.info("Fitted preprocessor saved")
            
            return best_model
        except Exception as e:
            logging.error(f"Error during model training: {str(e)}")
            raise CustomException(e, sys)
