from sklearn import  svm, metrics, tree,neural_network
import numpy as np
import csv


my_data = np.loadtxt('edited_data/export.csv',delimiter=',', dtype='str')
prediction_data = np.loadtxt('edited_data/classification_unlabeled_edited.csv', delimiter=',', dtype='str')

# print(my_data)
training_data = my_data[:, 0:6]
validation_data = my_data[:, 6]


# print(training_data)
classifier = tree.DecisionTreeClassifier(max_depth=10)
classifier.fit(training_data, validation_data)

predicted = classifier.predict(training_data[681:])

unknown = classifier.predict(prediction_data)
# print(unknown)


init_file_data = np.loadtxt('initial_data/test_classification_unlabeled.csv', delimiter=',', dtype='str')
with open('results/predictions_classification.csv', "w", newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    first_row = np.append(init_file_data[0], ['class'])
    writer.writerow(first_row)
    i = 0
    for row in init_file_data[1:, :]:
        row = np.append(row, [unknown[i]])
        print(row)
        writer.writerow(row)
        i += 1
