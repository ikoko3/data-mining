from sklearn import  svm, metrics, tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


my_data = np.loadtxt('edited_data/export.csv',delimiter=',', dtype='str')

training_data = my_data[:, 0:6]
validation_data = my_data[:, 6]


classifiers = [
    tree.DecisionTreeClassifier(max_depth=5),
    tree.DecisionTreeClassifier(max_depth=8),
    tree.DecisionTreeClassifier(max_depth=10),
    svm.SVC(kernel='linear'),
    svm.SVC(kernel='rbf'),
    AdaBoostClassifier(n_estimators=50),
    AdaBoostClassifier(n_estimators=100),
    KNeighborsClassifier(3),
    KNeighborsClassifier(5),
    KNeighborsClassifier(7)
]


for classifier in classifiers:
    classifier.fit(training_data[:1500], validation_data[:1500])
    expected = validation_data[681:]
    predicted = classifier.predict(training_data[681:])
    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


