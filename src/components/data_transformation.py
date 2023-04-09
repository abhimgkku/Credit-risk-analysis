import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from src.constants.training_pipeline import SCHEMA_FILE_PATH
from sklearn.feature_selection import SelectPercentile, chi2
#from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
#from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.constants.training_pipeline import TARGET_COLUMN
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact,
)
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from src.entity.config_entity import DataTransformationConfig
from src.exception import CreditException
from src.logger import logging
from src.utils.main_utils import save_numpy_array_data, save_object,read_yaml_file,write_yaml_file

class DataTransformation:
    def __init__(self,data_validation_artifact: DataValidationArtifact, 
                    data_transformation_config: DataTransformationConfig,):
        """
        :param data_validation_artifact: Output reference of data ingestion artifact stage
        :param data_transformation_config: configuration for data transformation
        """
        try:
            
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config

        except Exception as e:
            raise CreditException(e, sys)


    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CreditException(e, sys)


    @classmethod
    def get_data_transformer_object(cls)->Pipeline:
        try:
            selection = SelectPercentile(chi2, percentile= 80)
            schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            categorical_columns = schema_config["categorical_columns"]
            numerical_columns = schema_config["numerical_columns"]
            num_pipeline = Pipeline(steps=[
                    ('impute', SimpleImputer(strategy='constant'))])
            cat_pipeline = Pipeline(steps=[
                        ('impute', SimpleImputer(strategy='most_frequent')),
                        ('onehot',OneHotEncoder(handle_unknown='ignore'))])
            col_trans = ColumnTransformer(transformers=[
                            ('num_pipeline',num_pipeline,numerical_columns),
                            ('cat_pipeline',cat_pipeline,categorical_columns)])
            preprocessor = Pipeline(steps=[('col_trans', col_trans),
                        
                                    ])
            return preprocessor

        except Exception as e:
            raise CreditException(e, sys) from e

    
    def initiate_data_transformation(self,) -> DataTransformationArtifact:
        try:
            
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
            preprocessor = self.get_data_transformer_object()


            #training dataframe
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_final = train_df[TARGET_COLUMN]
            #target_feature_train_df = target_feature_train_df.replace( TargetValueMapping().to_dict())

            #testing dataframe
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_final = test_df[TARGET_COLUMN]
            #target_feature_test_df = target_feature_test_df.replace(TargetValueMapping().to_dict())

            preprocessor_object = preprocessor.fit(input_feature_train_df)
            input_feature_train_final = preprocessor_object.transform(input_feature_train_df)
            input_feature_test_final =preprocessor_object.transform(input_feature_test_df)

            #smt = SMOTETomek(sampling_strategy="minority")

            #input_feature_train_final, target_feature_train_final = smt.fit_resample(
            #   transformed_input_train_feature, target_feature_train_df
            #)

            #input_feature_test_final, target_feature_test_final = smt.fit_resample(
            #   transformed_input_test_feature, target_feature_test_df
            #)

            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final) ]
            test_arr = np.c_[ input_feature_test_final, np.array(target_feature_test_final) ]

            #save numpy array data
            save_numpy_array_data( self.data_transformation_config.transformed_train_file_path, array=train_arr, )
            save_numpy_array_data( self.data_transformation_config.transformed_test_file_path,array=test_arr,)
            save_object( self.data_transformation_config.transformed_object_file_path, preprocessor_object,)
            
            
            #preparing artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise CreditException(e, sys) from e