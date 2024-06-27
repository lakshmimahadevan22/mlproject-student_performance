import os 
import sys
import pickle
from dataclasses import dataclass
from xgboost import XGBRegressor
#from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.mode_trainer_config=ModelTrainerConfig()



    def initiate_model_trainer(self,train_array, train_target, test_array, test_target):
        try:
            logging.info("splitting train and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                "XGBClassifier": XGBRegressor(),
               # "CatBoosting Classifier": CatBoostRegressor(verbose=False),
                "AdaBoost Classifier": AdaBoostRegressor(),
            }

            #model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            ## to get best model score from dict
            #best_model_score = max(sorted(model_report.values()))

            ## to get best model name from dict
            #best_model_name = list(model_report.keys())[
            #    list(model_report.values().index(best_model_score))
            #]
            model_scores = {}  #chatgpt
            for model_name, model in models.items():
                logging.info(f"Training {model_name}")
                model.fit(train_array, train_target)
                predictions = model.predict(test_array)
                score = r2_score(test_target, predictions)
                model_scores[model_name] = score
                logging.info(f"{model_name} Score: {score}")



            best_model_name = max(model_scores, key=model_scores.get)
            best_model = models[best_model_name]

            #if best_model_score<0.6:
             #   raise CustomException("No best model found")
            #logging.info(f"Best found model on botht training and testing dataset")

            #save_object(
             #   file_path=self.model_trainer_config.trained_model_file_path,
              #  obj=best_model
            #)
            model_path = os.path.join('artifacts', 'model.pkl')
            with open(model_path, 'wb') as file:
                pickle.dump(best_model, file)
            logging.info(f"Best model saved: {best_model_name} with score {model_scores[best_model_name]}")
            
            return model_path

            #predicted=best_model.predict(X_test)

            #r2_square = r2_score(y_test, predicted)
            #return r2_square


        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise #CustomException(str(e),sys)


