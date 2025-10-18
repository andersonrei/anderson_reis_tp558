from autoPyTorch.api.tabular_regression import TabularRegressionTask
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np

RNG = 7
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RNG
)

api = TabularRegressionTask(seed=RNG)
api.search(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    optimize_metric="r2",
    total_walltime_limit=600,
    func_eval_time_limit_secs=120,
    memory_limit=4096
)

y_pred = api.predict(X_test)
score = api.score(y_pred, y_test)  # returns dict with r2, etc.
print("California Housing - Resultados:", score)
print("California Housing - Modelos:", api.show_models())
