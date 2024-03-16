import mlflow

import sklearn
from sklearn.ensemble import GradientBoostingClassifier

def train_model(x_train, y_train, x_test, y_test):
    model = GradientBoostingClassifier(max_depth = 10, n_estimators=200)
    model.fit(x_train, y_train)
    model_info = {
        'score':{
            'model_score': model.score(x_test, y_test)
        },
        'params': model.get_params()
    }

    return model, model_info
