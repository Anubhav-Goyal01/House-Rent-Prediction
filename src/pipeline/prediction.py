import sys
import pandas as pd
from src.exception import HousingException
from src.utils import load_object
from src.logger import logging
import os

class PredictionPipeline:
    def __init__(self) -> None:
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            features_preprocessor_path=os.path.join('artifacts','features_preprocessor.pkl')
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')

            model=load_object(file_path=model_path)
            features_preprocessor = load_object(file_path=features_preprocessor_path)
            preprocessor=load_object(file_path=preprocessor_path)

            features = features_preprocessor.transform(features)
            data_preprocessed = preprocessor.transform(features)
            preds = model.predict(data_preprocessed)
            return preds
        
        except Exception as e:
            raise HousingException(e,sys)


class CustomData:

    def __init__(
      self,
      posted_on:str,
      bhk: int,
      size: int,
      floor_level:int,
      total_floors: int,
      area_type: str,
      city: str,
      furnishing_status: str,
      tenant_preferred: str,
      bathrooms: int      
    ) -> None:
        
        self.posted_on = posted_on
        self.bhk = bhk
        self.size = size
        self.floor_level = floor_level
        self.total_floors = total_floors
        self.area_type = area_type
        self.city = city
        self.furnishing_status = furnishing_status
        self.tenant_preferred = tenant_preferred
        self.bathroom = bathrooms

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Posted On": [self.posted_on],
                "BHK": [self.bhk],
                "Size": [self.size],
                "Total Floors": [self.total_floors],
                "Floor Level": [self.floor_level],
                "City": [self.city],
                "Area Type": [self.area_type],
                "Furnishing Status": [self.furnishing_status],
                "Tenant Preferred": [self.tenant_preferred],
                "Bathroom": [self.bathroom],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise HousingException(e, sys)