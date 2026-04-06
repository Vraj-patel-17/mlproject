import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor,AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models
@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path .join("artifact","model.pkl")
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    def initiate_model_trainer(self,train_array,test_array,preprocessor_path):
        try:
            logging.info("split training and test input data")
            x_train,y_train,x_test,y_test=(train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])
            models={
                "random forest":RandomForestRegressor(),
                "decision tree":DecisionTreeRegressor(),
                "linear regression":LinearRegression(),
                "k-neighbors":KNeighborsRegressor(),
                "xgboost":XGBRegressor(),
                "catboosting classifier":CatBoostRegressor(verbose=0),
                "adaboost":AdaBoostRegressor()
                }
            model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)
            best_report_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_report_score)]
            best_model=models[best_model_name]
            if best_report_score<0.6:
                raise CustomException("no best model found")
            logging.info("best model found")
            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)
            predicted=best_model.predict(x_test)
            r2_sqaure=r2_score(y_test,predicted)
            return r2_sqaure

        except Exception as e:
            raise CustomException(e,sys)