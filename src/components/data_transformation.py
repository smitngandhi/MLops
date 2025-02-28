import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder ,StandardScaler
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts' , "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transform_obj(self):
        try:

            logging.info("Data_Transformation_object_generation_started")


            numeric_features = ['reading_score' , 'writing_score']
            categorical_features = ['gender',
                                    'race_ethnicity',
                                    'parental_level_of_education',
                                    'lunch',
                                    'test_preparation_course']
            
            num_pipeline = Pipeline(
                steps=[
                    ("impute" , SimpleImputer(strategy="median")),
                    ("scaler" , StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ('imputer' , SimpleImputer(strategy="most_frequent")),
                    ('one_hot_encoding' , OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ('scaler' , StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                [
                ('numerical_pipelines' , num_pipeline , numeric_features),
                ('categorical_pipelines' , categorical_pipeline , categorical_features)]
            )

            logging.info("Data_Transformation_object_generation_finished")

            return preprocessor
        
        except Exception as e:
            raise CustomException(e , sys)


    def initiate_data_transform(self , train_path , test_path):
        
        try:

            logging.info("Loading Dataset")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)


            preprocess_obj = self.get_data_transform_obj()

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name] , axis=1)
            target_feature_train_df = train_df[target_column_name]


            input_feature_test_df = test_df.drop(columns=[target_column_name] , axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Dataset_loaded into train and test input and target features")

            input_feature_train_arr = preprocess_obj.fit_transform(input_feature_train_df )
            input_feature_test_arr = preprocess_obj.transform(input_feature_test_df)

            logging.info("Transforming data completed")

            logging.info("Concatenating train and test data started")
            train_arr = np.c_[
                input_feature_train_arr , np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr , np.array(target_feature_test_df)
            ]

            logging.info("Concatenating train and test data completed")

            
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocess_obj
            )
            
        
            return (
                train_arr ,
                test_arr ,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e , sys)
