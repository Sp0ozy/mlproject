import os
import sys

import dill
from sklearn.metrics import r2_score

from src.logger import logging
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        logging.error("Error saving object")
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i in range(len(models.values())):
            models_name = list(models.values())[i]

            models_name.fit(X_train, y_train)

            y_test_pred = models_name.predict(X_test)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    
    except Exception as e:
        logging.error("Error evaluating model")
        raise CustomException(e, sys)

