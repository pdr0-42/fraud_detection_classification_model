import asyncio
import logging
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
)
from sklearn.exceptions import NotFittedError
from abc import ABC, abstractmethod


class PredictModel:
    def __init__(self, x_test: pd.DataFrame, y_test: pd.Series) -> None:
        self.x_test = x_test
        self.y_test = y_test
        self.y_pred = None

    def __str__(self) -> str:
        return ''

    def predict_results(self) -> str:
        """Function that to predict"""
        decision_tree = DecisionTreeClassifier()

        try:
            self.y_pred = decision_tree.predict(self.x_test)

        except NotFittedError:
            logging.exception('There was error to fit model')
            raise

        return self.y_pred


class MetricCaculator(ABC):

    @abstractmethod
    async def calculate_metric(self):
        pass


class F1ScoreCalculator(MetricCaculator):
    def __init__(self, y_pred, y_test) -> None:
        self.y_pred = y_pred
        self.y_test = y_test

    async def calculate_metric(self) -> float:
        f1_calculated = f1_score(self.y_test, self.y_pred, pos_label= 1)
        return f1_calculated


class RecallCalculator(MetricCaculator):
    def __init__(self, y_pred, y_test) -> None:
        self.y_pred = y_pred
        self.y_test = y_test

    async def calculate_metric(self):
        recall_calculated = recall_score(self.y_test, self.y_pred, pos_label = 1)
        return recall_calculated


class PrecisionCalculator(MetricCaculator):
    def __init__(self, y_pred, y_test) -> None:
        self.y_pred = y_pred
        self.y_test = y_test

    async def calculate_metric(self):
        precision_calculated = precision_score(self.y_test, self.y_pred, pos_label = 1)
        return precision_calculated


class ConfusionMatrix(MetricCaculator):
    def __init__(self, y_pred, y_test) -> None:
        self.y_pred = y_pred
        self.y_test = y_test

    async def calculate_metric(self):
        confusion_matrix_result = confusion_matrix(self.y_pred, self.y_test)
        return confusion_matrix_result
