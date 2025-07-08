import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_tranformed(self):
        try:
            numerical_features=[
                'math score',
                'reading score',
                'writing score'
            ]

            cat_features=[
                'gender', 'race/ethnicity',
                'parental level of education', 
                'lunch',
                'test preparation course'
            ]

            numerical_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy="median")),
                    ('scaler',StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy="most_frequent")),
                    ('encoder',OneHotEncoder()),
                    ('scaler',StandardScaler())
                ]
            )
            logging.info("numerical pipeline completed")
            logging.info("categorical pipeline completed")

            prepocessor=ColumnTransformer(
            [
                ("num_pipeline",numerical_pipeline,numerical_features),
                ("cat_pipeline",cat_pipeline,cat_features)
            ]
            )
            return prepocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("training and testing data readed")

            preprocessor_obj=self.get_tranformed()

            trageted_column_name="math score"
   
            input_feature_train_column=train_df.drop(columns=trageted_column_name,axis=1)
            trageted_column_train_column=train_df["math score"]

            input_feature_test_column=test_df.drop(columns=trageted_column_name,axis=1)
            trageted_column_test_column=test_df["math score"]

            logging.info("applying preprocessing obj ")

            input_feature_preprocessed_train_arr=preprocessor_obj.fit_transform(input_feature_train_column)
            input_feature_preprocessed_test_arr=preprocessor_obj.transform(input_feature_test_column)

            train_arr=np.c_(
                input_feature_preprocessed_train_arr,np.array(trageted_column_train_column)
            )
            test_arr=np.c_(
                input_feature_preprocessed_test_arr,np.array(trageted_column_test_column)
            )

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj

            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        

        except Exception as e:
            raise CustomException(e,sys)
        
