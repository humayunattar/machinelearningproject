import sys
import os
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainingConfig:
    trained_model_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    def initiate_model_trainer(self, train_array, test_array):

        try:
            logging.info("Splitting training and test input data")

            x_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            x_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models = {

                "Random Forest": RandomForestRegressor(),

                "Decision Tree": DecisionTreeRegressor(),

                "Gradient Boosting": GradientBoostingRegressor(),

                "Linear Regression": LinearRegression(),

                "XGBoost Regressor": XGBRegressor(),

                "CatBoost Regressor": CatBoostRegressor(verbose=False),

                "AdaBoost Regressor": AdaBoostRegressor(),

                "K-Neighbors Regressor": KNeighborsRegressor()

            }

            params = {

                "Decision Tree": {
                    'criterion': [
                        'squared_error',
                        'friedman_mse',
                        'absolute_error',
                        'poisson'
                    ]
                },

                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },

                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },

                "Linear Regression": {},

                "XGBoost Regressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },

                "CatBoost Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },

                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },

                "K-Neighbors Regressor": {
                    'n_neighbors': [3, 5, 7, 9]
                }

            }

            model_report = evaluate_models(
                X_train=x_train,
                y_train=y_train,
                X_test=x_test,
                y_test=y_test,
                models=models,
                param=params
            )

            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)

            logging.info("Best model found")

            best_model.fit(x_train, y_train)

            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test)

            score = r2_score(y_test, predicted)

            return score

        except Exception as e:
            raise CustomException(e, sys)