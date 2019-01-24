import numpy as np
from sklearn.svm import SVR
from sklearn import  svm, metrics
from sklearn.tree import DecisionTreeRegressor


my_data = np.loadtxt('edited_data/dataset_regression_edited.csv',delimiter=',', dtype='str')
prediction_data = np.loadtxt('edited_data/regression_unlabeled_edited.csv', delimiter=',', dtype='str')

training_data = my_data[:, 0:7]
# remove goal class from training dataset
training_data = np.delete(training_data, 5, 1)
validation_data = my_data[:, 5]

X = training_data
y = validation_data


regressions = [
    SVR(kernel='rbf', C=1e3, gamma=0.05),
    SVR(kernel='linear', C=0.8, gamma=0.05),
    DecisionTreeRegressor(max_depth=50, criterion='friedman_mse', splitter='best', min_samples_leaf=2, presort=True )
]


for rg in regressions:
    rg.fit(training_data[:1500], validation_data[:1500])
    expected = validation_data[681:].astype(np.float)
    predicted = rg.predict(training_data[681:])
    print("Regression report for regression %s:\n%s\n"
          % (rg, metrics.explained_variance_score(expected, predicted)))
