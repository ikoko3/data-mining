import numpy as np
from sklearn.svm import SVR
import csv
import matplotlib.pyplot as plt

my_data = np.loadtxt('edited_data/export.csv',delimiter=',', dtype='str')
prediction_data = np.loadtxt('edited_data/regression_unlabeled_edited.csv', delimiter=',', dtype='str')

training_data = my_data[:, 0:6]
validation_data = my_data[:, 5]

X = training_data
y = validation_data


# #############################################################################
# Fit regression model
svr = SVR(kernel='linear', C=1e3)
predictions = svr.fit(X, y).predict(prediction_data)


init_file_data = np.loadtxt('initial_data/test_regression_unlabeled.csv', delimiter=',', dtype='str')
with open('results/predictions_regression_whole_dataset.csv', "w", newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    first_row = np.append(init_file_data[0], ['price'])
    writer.writerow(first_row)
    i = 0
    for row in init_file_data[1:, :]:
        price = "%.2f" % predictions[i]
        row = np.append(row, price)
        writer.writerow(row)
        i += 1

with open('results/predictions_regression.csv', "w", newline='') as csv_file:
    writer = csv.writer(csv_file)
    for row in predictions:
        price = "%.2f" % row
        writer.writerow([price])