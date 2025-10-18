from autoPyTorch.api.tabular_classification import TabularClassificationTask
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np

# Dados
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# API
api = TabularClassificationTask(seed=42)

api.search(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    optimize_metric="accuracy",
    total_walltime_limit=300,      # 5 min
    func_eval_time_limit_secs=90,  # 1.5 min por modelo
    memory_limit=4096
)

y_pred = api.predict(X_test)
score = api.score(y_pred, y_test)
print("Wine - Resultados:", score)
print(api.show_models())
