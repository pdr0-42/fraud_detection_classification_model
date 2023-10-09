import logging
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from sklearn.exceptions import NotFittedError

class PredictModel():

    def __init__(self, fraud_dataset: pd.DataFrame, y_pred: np.ndarray,
                 x_test: pd.DataFrame) -> None:

        self.fraud_dataset = fraud_dataset
        self.x_test = x_test


    def __str__(self) -> str:

        return ""


    def predict_results(self) -> str:
        """Function that to predict"""
        decision_tree = DecisionTreeClassifier()

        try:
            y_pred = decision_tree.fit(self.x_test)

        except NotFittedError:
            logging.exception("There was error to fit model")
            raise

        return y_pred


    async def recall_score(self) -> float:
        """recall score - Metric that allows """
        try:
            recall_score(y_test, y_pred, pos_label = 1)

        except Exception:
            logging.exception("There was error to generate recall score")
            raise
