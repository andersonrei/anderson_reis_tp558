from autoPyTorch.api.tabular_classification import TabularClassificationTask
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)

api = TabularClassificationTask(seed=1)
api.search(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    optimize_metric="accuracy",
    total_walltime_limit=300,
    func_eval_time_limit_secs=90,
    memory_limit=4096
)

y_pred = api.predict(X_test)
score = api.score(y_pred, y_test)
print("Breast Cancer - Resultados:", score)
print(api.show_models())
