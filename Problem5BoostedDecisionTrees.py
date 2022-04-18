from xgboost import XGBClassifier
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from itertools import product
import warnings

warnings.filterwarnings('ignore')

# Load data into test and train
x_test, y_test = load_svmlight_file("a9a.t")
x_train, y_train = load_svmlight_file("a9a.txt")

# Fit model based on training data, use default values for ease of understanding
data_model = XGBClassifier()
data_model.fit(x_train, y_train)
print("Default values for XGBoost hyperparameters : \n ", data_model)

# Predictions for test data
y_prediction = data_model.predict(x_test)
all_predictions = [round(val) for val in y_prediction]

# Run predictions and calculate accuracy
acc = accuracy_score(y_test, all_predictions)
print("\n Accuracy: %.2f%%" % (acc * 100))

# Tune hyperparameters, make list for all values and obtain all possible combinations (long process)
max_depth = [3, 4, 5]
learning_rate = [0.05, 0.1, 0.2]
missing = [None, 0]
n_estimators = [100, 200, 300]
reg_lambda = [0.0, 1.0]
objective = ['binary:logistic', 'binary:logitraw', 'binary:hinge']

hyperparameters = []
for depth, rate, miss, n_estimate, lam, obj in product(max_depth, learning_rate,
                                                       missing, n_estimators,
                                                       reg_lambda, objective):
    hyperparameters.append([depth, rate, miss, n_estimate, lam, obj])

best_acc = 0

# Run through the list of all parameters and find data model with best acc
for parameter in hyperparameters:
    parameters = {'max_depth': parameter[0], 'learning_rate': parameter[1],
                  'missing': parameter[2], 'n_estimators': parameter[3],
                  'reg_lambda': parameter[4], 'objective': parameter[5]}

    data_model = XGBClassifier(max_depth=parameter[0], learning_rate=parameter[1],
                               missing=parameter[2], n_estimators=parameter[3],
                               reg_lambda=parameter[4], objective=parameter[5])

    kfold = KFold()
    cross_val_scores = cross_val_score(data_model, x_train, y_train, cv=kfold)
    accuracy = cross_val_scores.mean() * 100

    #Check for best accuracy
    if best_acc < accuracy:
        best_acc = accuracy
        best_model = parameters

print("\n Best accuracy : ", accuracy)
print("\n The model with best accuracy : ", best_model)

print("\nCross Validation Training Error Rate for the new model: ",1-cross_val_scores.mean())

print("\nTest Error Rate for the best model: ",1-best_model.score(x_test, y_test))
