import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class TrainingModel:
    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.dataframe = dataframe

    def split_data_train_test(self, dataframe: pd.DataFrame):
        X = dataframe.drop(columns="Class")
        y = dataframe["Class"]
        return X, y


    def train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        tree = DecisionTreeClassifier()
