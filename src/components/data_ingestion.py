import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'raw_data.csv')
    train_data_path: str = os.path.join('artifacts', 'train_data.csv')
    test_data_path: str = os.path.join('artifacts', 'test_data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion process")
        try:
            df=pd.read_csv('notebook/data/student_performance.csv')
            logging.info("Dataset read successfully")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data saved successfully")

            train_set, test_set = train_test_split(df, test_size=0.2)
            logging.info("Data split into train and test sets")

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Train and test data saved successfully")

            return (self.ingestion_config.train_data_path,   self.ingestion_config.test_data_path)
    
        except Exception as e:
            logging.error("Error occurred during data ingestion")
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion() 
    obj.initiate_data_ingestion()
