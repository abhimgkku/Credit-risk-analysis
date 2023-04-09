from src.configuration.mongo_db_connection import MongoDBClient
from src.exception import CreditException
from src.logger import logging
import os,sys
from src.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig
from src.pipeline.training_pipeline import TrainPipeline

if __name__ == "__main__":
    train_pipeline = TrainPipeline()
    train_pipeline.run_pipeline()
    