## Mysql---> Train test split--->dataset
import os
import sys
from src.mlproject.expception import CustomException
from src.mlproject.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            ##reading code
            logging.info("Reading from mysql database")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            

        except Exception as e:
            raise CustomException(e,sys)
