import os
import sys
import pandas as pd 

from src.logger import logging
from src.exception import CustomException
from src.constants import ARTIFACT_DIR

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.utils.utils import save_object

class DataTransformation:
    def __init__(self,df: pd.DataFrame):
        self.df = df
        self.preprocessor_path = os.path.join(ARTIFACT_DIR, "models", "preprocessor.pkl")
        
    
    def clean_data(self):
        try:
            self.df["TotalCharges"] = pd.to_numeric(self.df["TotalCharges"], errors='coerce')
            logging.info("Data cleaning successful. 'TotalCharges' converted to numeric.")
        except Exception as e:
            logging.error(f"Error during data cleaning: {str(e)}")
            raise CustomException(e, sys)
    
    def create_preprocessor(self, X):
        try:
            numeric_features = X.select_dtypes(exclude=['object']).columns.tolist()
            categorical_features = X.select_dtypes(include=['object']).columns.tolist()
            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            preprocessor = ColumnTransformer(transformers=[
                ('num', numerical_pipeline, numeric_features),
                ('cat', categorical_pipeline, categorical_features)
            ])

            logging.info("Preprocessor creation successful.")

            return preprocessor
        except Exception as e:
            logging.error(f"Error during preprocessor creation: {str(e)}")
            raise CustomException(e, sys)
        
    def transform(self):
        try:
            logging.info("Data transformation started.")

            self.clean_data()

            X = self.df.drop(["Churn", "customerID"], axis=1)

            # Convert 'Churn' to numeric: 'No' -> 0, 'Yes' -> 1
            y = self.df["Churn"].map({"No": 0, "Yes": 1})

            preprocessor = self.create_preprocessor(X)
            save_object(self.preprocessor_path, preprocessor)
            logging.info("Preprocessor saved")
            return X, y, preprocessor
        except Exception as e:
            logging.error(f"Error during data transformation: {str(e)}")
            raise CustomException(e, sys)
        