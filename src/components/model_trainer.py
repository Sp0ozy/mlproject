import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from src.logger import logging
from src.exception import CustomException

from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train = train_array[:,:-1], train_array[:,-1]
            X_test, y_test = test_array[:,:-1], test_array[:,-1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(),
                "SVR": SVR(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params = {
              "Decision Tree": {
                "criterion": ["squared_error", "absolute_error", "friedman_mse"],
                "splitter": ["best"],
                "max_depth": [None, 3, 5, 8, 12],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 5, 10],
                "max_features": [None, "sqrt", "log2"]
              },

              "Random Forest": {
                "n_estimators": [300, 600, 1000],
                "max_depth": [None, 6, 10, 16],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 5],
                "max_features": ["sqrt", 0.5, 1.0],
                "bootstrap": [True]
              },

              "Gradient Boosting": {
                "learning_rate": [0.05, 0.1, 0.2],
                "n_estimators": [200, 500, 1000],
                "max_depth": [2, 3, 4],
                "subsample": [0.7, 0.85, 1.0],
                "min_samples_leaf": [1, 3, 10],
                "max_features": [None, "sqrt"]
              },

              "Linear Regression": {},

              "Ridge Regression": {
                "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100],
                "solver": ["auto"],
                "max_iter": [5000],
                "tol":[1e-3]
              },

              "Lasso Regression": {
                "alpha": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 0.3, 1.0],
                "selection": ["cyclic", "random"],
                "max_iter": [5000],
                "tol":[1e-3]
              },

              "SVR": {
                "kernel": ["rbf", "linear"],
                "C": [0.1, 1, 10, 100],
                "epsilon": [0.01, 0.1, 0.2],
                "gamma": ["scale", 0.01, 0.1, 1.0]
              },

              "K-Neighbors Regressor": {
                "n_neighbors": [3, 5, 9, 15, 25, 35],
                "weights": ["uniform", "distance"],
                "p": [1, 2], # 1 for Manhattan, 2 for Euclidean
                "leaf_size": [20, 30, 50]
              },

              "XGB Regressor": {
                "n_estimators": [500, 1200, 2500],
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth": [2, 3, 4, 6],
                "min_child_weight": [1, 5, 10],
                "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.7, 0.85, 1.0],
                "reg_lambda": [1.0, 5.0, 10.0],
                "reg_alpha": [0.0, 0.1, 1.0]
              },

              "AdaBoost Regressor": {
                "n_estimators": [200, 500, 1000],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "loss": ["linear", "square", "exponential"]
              },
            }


            model_report: dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)
            
            best_model_name = max(model_report, key=lambda x: model_report[x]["r2_score"])
            r2_score = model_report[best_model_name]["r2_score"]
            cv_score = model_report[best_model_name]["cv_score"]

            if model_report[best_model_name]["r2_score"] < 0.6:
                raise CustomException("No best model found with r2 score greater than 0.6", sys)
            
            if cv_score is not None and (cv_score - r2_score) > 0.1:
                    logging.warning( f"Possible overfitting: CV={cv_score:.3f}, r2_score={r2_score:.3f}")
            
            best_model = model_report[best_model_name]["model"]
            
            logging.info(f"Best model found: {best_model_name} with r2 score: {model_report[best_model_name]['r2_score']}")


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            return model_report[best_model_name]["r2_score"]
        
        except Exception as e:
            logging.error("Error in model training")
            raise CustomException(e, sys)
        