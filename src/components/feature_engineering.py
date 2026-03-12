import sys
from src.logger import logging
from src.exception import CustomException


class FeatureEngineering:

    def __init__(self, df):
        self.df = df

    def apply_features(self):

        return self.df