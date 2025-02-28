import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join("artifacts" , "train.csv")
    test_data_path: str=os.path.join("artifacts" , "test.csv")
    raw_data_path: str=os.path.join("artifacts" , "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Initiated data Ingestion")
            df = pd.read_csv("notebook\data\stud.csv")
            logging.info("Loaded dataset into dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path) , exist_ok=True)

            logging.info("Created artifacts directory")

            df.to_csv(self.ingestion_config.raw_data_path  , index=False , header=True)

            logging.info("Stored Raw data")

            logging.info('Train Test Split initiated')

            train_set , test_set = train_test_split(df , test_size=0.2 , random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path ,index=False , header=True)
            train_set.to_csv(self.ingestion_config.test_data_path, index=False , header=True)

            logging.info("Data Ingestion is completed")

            return (
                self.ingestion_config.train_data_path , self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    obj = DataIngestion()
    train_path , test_path = obj.initiate_data_ingestion()


    data_transformation = DataTransformation()
    data_transformation.initiate_data_transform(train_path=train_path , test_path=test_path)