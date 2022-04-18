# Non-linear Classifiers
Will need to learn to use software libraries for at least two of the following non-linear classifier types:
- [x] Boosted Decision Trees (i.e., boosting with decision trees as weak learner)
- [x] Random Forests
- [ ] Support Vector Machines with Gaussian Kernel

All of these are available in ```scikit-learn```, although you may also use other external libraries (e.g., ```XGBoost``` for boosted decision trees ([Sample code](https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/)) and ```LibSVM``` for SVMs).
Pick two different types of non-linear classifiers from above for classification of Adult dataset. You can the download the data from **a9a** in libSVM data repository. The **a9a** data set comes with two files: the training data file **a9a** with 32,561 samples each with 123 features, and **a9a.t** with 16,281 test samples. Note that **a9a** data is in ```LibSVM``` format. In this format, each line takes the form ```<label> <feature-id>:<feature-value> <feature-id>:<feature-value> ....```. This format is especially suitable for sparse datasets. Note that ```scikit-learn``` includes utility functions (e.g., ```load svmlight``` file in example code below) for loading datasets in the ```LibSVM``` format.

For each of learning algorithms, we will need to set various hyperparameters (e.g., the type of kernel and regularization parameter for SVM, tree method, max depth, number of weak classifiers, etc for XGBoost, number of estimators and min impurity decrease for Random Forests). Often there are defaults that make a good starting point, but we may need to adjust at least some of them to get good performance. Use hold-out validation or K-fold cross-validation to do this (scikit-learn has nice features to accomplish this, e.g., you may use train test split to split data into train and test data and sklearn.model selection for K-fold cross validation). Do not make any hyperparameter choices (or any other similar choices) based on the test set! We should only compute the test error rates after you have settled on hyperparameter settings and trained your two final classifiers.

### Parameters to be tuned for XGBoost: 
1. ```n estimators```
2. ```max depth```
3. ```lambda```
4. ```learning rate```
5. ```missing```
6. ```objective```

### Parameters to be tuned for SVM: 
1. ```kernel type```
2. ```gamma```
3. ```C```

### Parameters to be tuned for Random Forests: 
1. ```n estimators```
2. ```bootstrap```
3. ```max depth```
4. ```min impurity decrease```
5. ```min samples leaf```

### Example code to use XGBoost
```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_svmlight_file

from xgboost import XGBClassifier

# load data in LibSVM sparse data format
X, y = load_svmlight_file("a9a")
# split data into train and test sets
seed = 6
test_size = 0.4
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

# fit model on training data
# for simplicity we fit based on default values of hyperparameters

model = XGBClassifier()
model.fit(X_train, y_train)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred] 
# evaluate predictions
accuracy = accuracy_score(y_test, predictions) 
print("Accuracy: %.2f%%" % (accuracy * 100.0))
```

## Results

### Boosted Decision Trees (XGBoost Classifier)

| Hyperparameter | Default Value | Final Value |
| ------------- | ------------- | ------------- |
| ```n_estimators``` | 100 | 200  |
| ```max_depth``` | 6  | 3  |
| ```lambda``` | 1  | 1 |
| ```learning_rate``` | 0.3000000012  | 0.2  |
| ```missing``` | None  | None  |
| ```objective``` | logistic:binary  | logistic:binary |

**Accuracy of Default Model:** 84.83%

**Accuracy of New Model:** 85.0711%

**Cross Validation Training Error Rate:** 0.1492889480913432

**Test Error Rate:** 0.14624408820097046


### Random Forests (Random Forest Classifier)

| Hyperparameter | Default Value | Final Value |
| ------------- | ------------- | ------------- |
| ```n_estimators``` | 100 | 300 |
| ```bootstrap``` | True  | True |
| ```max_depth``` | None  | None |
| ```min_impurity_decrease``` | 0.0 | 0.0 |
| ```min_samples_leaf``` | 1 | 2 |

**Accuracy of Default Model:** 83.30%

**Accuracy of New Model:** 84.5674%

**Cross Validation Training Error Rate:** 0.1543255673495194

**Test Error Rate:** 0.15293900866040167
