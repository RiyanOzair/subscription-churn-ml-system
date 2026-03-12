import sys

from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainingPipeline:

    def start_pipeline(self):

        try:

            logging.info("Training pipeline started")

            logging.info("Data ingestion started")
            ingestion = DataIngestion()
            df = ingestion.load_data()
            logging.info(f"Dataset shape: {df.shape}")
            logging.info("Data ingestion completed")

            logging.info("Data validation started")
            validation = DataValidation(df)
            validation.validate()
            logging.info("Data validation completed")

            logging.info("Data transformation started")
            transformation = DataTransformation(df)
            X, y, preprocessor = transformation.transform()
            logging.info("Data transformation completed")

            logging.info("Model training started")
            trainer = ModelTrainer(X, y, preprocessor)
            trainer.train()
            logging.info("Model training completed")



        except Exception as e:

            raise CustomException(e, sys)


if __name__ == "__main__":

    pipeline = TrainingPipeline()

    pipeline.start_pipeline()