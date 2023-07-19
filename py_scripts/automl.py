# from tpot import TPOTRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from modules.data.mixed import *
from modules.models.simple import *
import autoPyTorch as apt
from autoPyTorch.api.tabular_regression import TabularRegressionTask


def custom_train_test_split(dataset):
    X = []
    y = []
    for i in range(len(dataset)):
        input, output = dataset[i]
        X.append(input.cpu().numpy())
        y.append(output.cpu().numpy())
        
    return np.array(X), np.array(y) 

csv_file = "data/static/l3p_t3.csv"
abm_dataset = ABMDataset(csv_file, root_dir="data/", norm_out=True)
X, y = custom_train_test_split(abm_dataset)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=1,
)
print(X_train.shape)
print(y_train.shape)
# tpot = TPOTRegressor(generations=10, population_size=50, template=NeuralNetwork)
auto = TabularRegressionTask()
auto.search(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test.copy(),
    y_test=y_test.copy(),
    optimize_metric='r2',
    total_walltime_limit=300,
    func_eval_time_limit_secs=50,
    dataset_name="SurrogateL3P"
)

y_pred = auto.predict(X_test)
score = auto.score(y_pred, y_test)
print(auto.sprint_statistics())

auto.refit(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    dataset_name="AutoSurrogateL3P",
    total_walltime_limit=500,
    run_time_limit_secs=50
    # you can change the resampling strategy to
    # for example, CrossValTypes.k_fold_cross_validation
    # to fit k fold models and have a voting classifier
    # resampling_strategy=CrossValTypes.k_fold_cross_validation
)

y_pred = auto.predict(X_test)
score = auto.score(y_pred, y_test)
print(score)

# Print the final ensemble built by AutoPyTorch
print(auto.show_models())
# tpot.fit(X_train, y_train)