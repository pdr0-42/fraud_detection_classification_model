import logging
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
)
from sklearn.exceptions import NotFittedError


class PredictModel:
    def __init__(self, x_test: pd.DataFrame, y_test) -> None:
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

    async def recall_score(self) -> float:
        """recall score - Metric that allows"""
        try:
            recall = recall_score(self.y_test, self.y_pred, pos_label=1)

        except Exception:
            logging.exception('There was error to generate Recall Score')
            raise

        return recall

    async def f1_score(self) -> float:
        """F1 Score is a metric that allow us to validate model performance"""
        try:
            f1 = f1_score(self.y_test, self.y_pred, pos_label=1)

        except Exception:
            logging.exception('There was error to generate F1 Score')

        return f1

    async def precision_score(self) -> float:
        """"""
        try:
            precision_score(self.y_test, self.y_pred, pos_label=1)

        except Exception:
            logging.exception('There was error to generate precision score')
            raise

    async def cross_validate(self):
        """"""
        try:
            