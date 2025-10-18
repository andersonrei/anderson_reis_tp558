from autoPyTorch.api.tabular_classification import TabularClassificationTask
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")


X, y = load_digits(return_X_y=True)
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
    total_walltime_limit=600,          # 10 min
    func_eval_time_limit_secs=120,     # 2 min por avaliação
    memory_limit=4096,
)

y_pred = api.predict(X_test)
score = api.score(y_pred, y_test)
print("Resultados:", score)
print("Modelos:", api.show_models())
