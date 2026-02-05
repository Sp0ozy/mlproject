import os
import sys

import dill
# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
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
    
def evaluate_model(X_train, y_train, X_test, y_test, models, params,n_iter=25):
    try:
        report = {}
        for name, model in models.items():
            param_grid = params.get(name)

            if param_grid:
                gs = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_grid,
                    n_iter=n_iter,
                    cv=3, 
                    n_jobs=-1,
                    verbose=1,
                    random_state=42                                        
                )    
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
            else:
                best_model = model.fit(X_train, y_train)

            y_pred = best_model.predict(X_test)

            report[name] = {
                "model": best_model,
                "r2_score": r2_score(y_test, y_pred)
            }

        return report
    
    except Exception as e:
        logging.error("Error evaluating model")
        raise CustomException(e, sys)

