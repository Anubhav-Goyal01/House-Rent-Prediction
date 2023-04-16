import os
import sys
from dataclasses import dataclass

from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from src.exception import HousingException
from src.logger import logging

from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', "model.pkl")



class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting train and test array")
            X_train, y_train, X_test, Y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            models = {
                'xgboost' : XGBRegressor(),
                'catboost' : CatBoostRegressor(verbose=0),
                'lightgbm' : LGBMRegressor(),
                'gradient boosting' : GradientBoostingRegressor(),
                'random forest' : RandomForestRegressor(),
            }


            params={
                "xgboost":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },

                "catboost":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },

                "lightgbm":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'max_depth': [6,8,10],
                    'n_estimators': [8,16,32,64,128,256]
                }

                "gradient boosting":{
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256]
                },

                "random forest":{
                    'n_estimators': [8,16,32,64,128,256]
                }           
            }

            model_report:dict = evaluate_models(X_train, y_train, X_test, Y_test, models, params)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise HousingException("No best Model Found")

            logging.info("Model trained successfully")

            save_model(
                filepath = self.model_trainer_config.trained_model_file_path, 
                obj = best_model,
            )

            predicted = best_model.predict(X_test)
            r2_score = r2_score(y_test, predicted)
            return r2_score


        except Exception as e:
            raise HousingException(e, sys)