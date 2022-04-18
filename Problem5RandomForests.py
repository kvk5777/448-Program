from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from itertools import product
import warnings

warnings.filterwarnings('ignore')

# Load data
x_test, y_test = load_svmlight_file("a9a.t")
x_train, y_train = load_svmlight_file("a9a.txt")

# Fit model based on training data, use default values for ease of understanding
model = RandomForestClassifier()
model.fit(x_train, y_train)
print("Default values of the hyperparameters for Random Forests : \n ", model.get_params())

# Predictions for test data
y_prediction = model.predict(x_test)
all_predictions = [round(val) for val in y_prediction]

# Run predictions and calculate accuracy
acc = accuracy_score(y_test, all_predictions)
print("\n Accuracy: %.2f%%" % (acc * 100))

# Tune hyperparameters, make list for all values and obtain all possible combinations (long process)
n_estimators = [50, 100, 200, 300]
bootstrap = [True, False]
max_depth = [None, 500, 1000]
min_impurity_reduction = [0.0, 0.05, 0.1, 0.2]
min_samples_leaf = [1, 2, 10, 50, 100]

hyperparameters = []
for n_estimate, boot, depth, min_impurity, min_samples in product(n_estimators, bootstrap, max_depth,
                                                                  min_impurity_reduction, min_samples_leaf):
    hyperparameters.append(
        [n_estimate, boot, depth, min_impurity, min_samples])

best_acc = 0

# Run through the list of all parameters and find data model with best acc
for param in hyperparameters:
    parameters = {'n_estimators': param[0],
                  'bootstrap': param[1], 'max_depth': param[2],
                  'min_impurity_reduction': param[3], 'min_samples_leaf': param[4]}
    model = RandomForestClassifier(n_estimators=param[0],
                                   bootstrap=param[1], max_depth=param[2],
                                   min_impurity_decrease=param[3], min_samples_leaf=param[4])

    kfold = KFold()
    cross_val_scores = cross_val_score(model, x_train, y_train, cv=kfold)
    accuracy = cross_val_scores.mean() * 100

    # Check for best accuracy
    if best_acc < accuracy:
        best_acc = accuracy
        best_model = param

print("\nBest accuracy that we can get: ", accuracy)
print("\nThe model that gives this is: ", best_model)
