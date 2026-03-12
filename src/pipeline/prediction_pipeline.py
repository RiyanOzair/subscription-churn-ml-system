import sys
import os
import pandas as pd

from src.utils.utils import load_object
from src.exception import CustomException
from src.constants import ARTIFACT_DIR


class PredictionPipeline:

    def __init__(self):

        self.preprocessor = load_object(
            os.path.join(ARTIFACT_DIR, "models", "preprocessor.pkl")
        )

        self.model = load_object(
            os.path.join(ARTIFACT_DIR, "models", "churn_model.pkl")
        )

    def predict(self, input_data):

        try:

            df = pd.DataFrame([input_data])

            X = self.preprocessor.transform(df)

            prediction = self.model.predict(X)

            probability = self.model.predict_proba(X)[:,1]

            return {
                "prediction": int(prediction[0]),
                "churn_probability": float(probability[0])
            }

        except Exception as e:
            raise CustomException(e, sys)