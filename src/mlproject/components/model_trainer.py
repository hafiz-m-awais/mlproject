import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            params={
                "Decision Tree": {
                    "max_depth": [10],
                    "min_samples_split": [5],
                    "min_samples_leaf": [2]
                },
                "Random Forest": {
                    "n_estimators": [100],
                    "max_depth": [None],
                    "min_samples_split": [5],
                    "min_samples_leaf": [2],
                    "max_features": ["sqrt"]
                },
                "Gradient Boosting": {
                    "n_estimators": [100],
                    "learning_rate": [0.1],
                    "max_depth": [3],
                    "subsample": [0.8]
                },
                "Linear Regression": {},
                "K-Neighbors Regressor": {
                    "n_neighbors": [5],
                    "weights": ["distance"],
                    "p": [2]   # Euclidean
                },
                "XGBRegressor": {
                    "n_estimators": [100],
                    "learning_rate": [0.1],
                    "max_depth": [6],
                    "subsample": [0.8],
                    "colsample_bytree": [0.8],
                    "reg_alpha": [0],
                    "reg_lambda": [1]
                },
                "CatBoosting Regressor": {
                    "iterations": [100],
                    "learning_rate": [0.1],
                    "depth": [6],
                    "l2_leaf_reg": [3]
                },
                "AdaBoost Regressor": {
                    "n_estimators": [100],
                    "learning_rate": [0.1],
                    "loss": ["linear"]
                }
            }
            model_report:dict=evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)
            ##to get best model score from dict
            best_model_score=max(sorted(model_report.values()))

            ##to get best model name from dict
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found on both training and testing dataset: {best_model_name}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted=best_model.predict(X_test)
            r2_square=r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
        
