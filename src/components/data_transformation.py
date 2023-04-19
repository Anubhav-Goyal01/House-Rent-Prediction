import sys
import os
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import HousingException
from src.logger import logging
from sklearn.base import BaseEstimator,TransformerMixin
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
    features_preprocessor_obj_file_path = os.path.join('artifacts', "features_preprocessor.pkl")


class FeaturesGenerator(BaseEstimator, TransformerMixin):

    def fit(self, X):
        return self


    def transform(self, X):
        try:
            logging.info("Adding new features")           
            # X = X.dropna()
            X['Floor Level'] = X['Floor Level'].astype(int)
            X['Total Floors'] = X['Total Floors'].astype(int)

            X['Posted On'] = pd.to_datetime(X['Posted On'])
            X['month posted'] = X['Posted On'].dt.month
            X['day posted'] = X['Posted On'].dt.day
            X['day of week posted'] = X['Posted On'].dt.day_of_week
            X['quarter posted'] = X['Posted On'].dt.quarter

            X.drop('Posted On', axis = 1, inplace= True)
            logging.info("New features added")

            return X

        except Exception as e:
            raise HousingException(e, sys)



class LogScaling(BaseEstimator, TransformerMixin):

    def fit(self, X):
        return self   

    def transform(self, X):
        return np.log(X)

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformer_object(self):
        try:
            log_scaling_cols = ['Size']
            cat_cols = ['Area Type', 'City', 'Furnishing Status', 'Tenant Preferred']
            num_cols = ['BHK', 'Size', 'Bathroom',  'month posted',  'day posted', 'day of week posted', 'quarter posted', 'Total Floors', 'Floor Level']


            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )


            feature_eng_pipeline = Pipeline(
                steps=[
                ('feature_generator', FeaturesGenerator())
                ]
            )

            preprocessor = ColumnTransformer([
                ("log_transform", LogScaling(), log_scaling_cols),
                ("num_pipeline", num_pipeline, num_cols),
                ("cat_pipelines",cat_pipeline,cat_cols)
                ], remainder= 'passthrough')

            return feature_eng_pipeline, preprocessor
        
        except Exception as e:
            raise HousingException(e, sys)



    def initiate_data_transformation(self,train_path,test_path):

        try:

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Data Read successfully")

            for i in train_df['Floor'].str.split(' out of '):
                train_df['Floor Level'] = i[0]
                try:
                    train_df['Total Floors'] = i[1]
                except:
                    train_df['Total Floors'] = i[0]
        
            train_df['Floor Level'] = train_df.apply(lambda x: 0 if x['Floor Level'] =='Ground' \
                                 else ( -1 if x['Floor Level'] =='Lower Basement' else (x['Total Floors']) ) , axis=1) 
             
            train_df.drop('Floor', axis=1, inplace=True) 
            # We will ask for these 2 new features from the UI

            for i in test_df['Floor'].str.split(' out of '):
                test_df['Floor Level'] = i[0]
                try:
                    test_df['Total Floors'] = i[1]
                except:
                    test_df['Total Floors'] = i[0]
        
            test_df['Floor Level'] = test_df.apply(lambda x: 0 if x['Floor Level'] =='Ground' \
                                 else ( -1 if x['Floor Level'] =='Lower Basement' else (x['Total Floors']) ) , axis=1)   
            test_df.drop('Floor', axis=1, inplace=True) 


            logging.info("Obtaining preprocessing object")

            features_obj, preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Rent"


            X_train = train_df.drop(columns=[target_column_name], axis=1)
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns=[target_column_name], axis=1)
            y_test = test_df[target_column_name]

            logging.info("Creating new Features")
            X_train = features_obj.fit_transform(X_train)
            X_test = features_obj.fit_transform(X_test)

            logging.info(f"Applying preprocessing object on training and test set")
            X_train_arr = preprocessing_obj.fit_transform(X_train)
            X_test_arr  = preprocessing_obj.transform(X_test)

            train_arr = np.c_[X_train_arr, np.array(y_train)]
            test_arr = np.c_[X_test_arr, np.array(y_test)]

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            save_object(
                file_path = self.data_transformation_config.features_preprocessor_obj_file_path,
                obj = features_obj
            )

            logging.info(f"Saved preprocessing object.")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
                self.data_transformation_config.features_preprocessor_obj_file_path
            )
        except Exception as e:
            raise HousingException(e,sys)
