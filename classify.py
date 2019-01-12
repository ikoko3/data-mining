from sklearn import  svm, metrics, tree,neural_network
import numpy as np
import csv


my_data = np.loadtxt('edited_data/dataset_classification_edited.csv',delimiter=',', dtype='str')
prediction_data = np.loadtxt('edited_data/classification_unlabeled_edited.csv', delimiter=',', dtype='str')

# print(my_data)
training_data = my_data[:, 0:6]
validation_data = my_data[:, 6]


# print(training_data)
classifier = tree.DecisionTreeClassifier(max_depth=10)
classifier.fit(training_data, validation_data)


unknown = classifier.predict(prediction_data)

init_file_data = np.loadtxt('initial_data/test_classification_unlabeled.csv', delimiter=',', dtype='str')
with open('results/predictions_classification_whole_dataset.csv', "w", newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    first_row = np.append(init_file_data[0], ['class'])
    writer.writerow(first_row)
    i = 0
    for row in init_file_data[1:, :]:
        row = np.append(row, [unknown[i]])
        writer.writerow(row)
        i += 1

with open('results/predictions_classification.csv', "w", newline='') as csv_file:
    writer = csv.writer(csv_file)
    for row in unknown:
        writer.writerow([row])

# only for visualization
device_classes = np.loadtxt('edited_data/classes.csv', delimiter=',', dtype='str')
device_classes = device_classes[:, 0]

with open('results/predictions_classification_visualization.csv', "w", newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    i = 0
    for row in prediction_data:
        device_class = unknown[i]
        value = np.where(device_classes == device_class)

        device_class = value[0][0] + 1

        row = np.append(row, [device_class])
        writer.writerow(row)
        i += 1