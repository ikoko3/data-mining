from sklearn import svm, metrics, tree, neural_network
import numpy as np
import csv

brands = np.loadtxt('brands.csv', delimiter=',', dtype='str')
prediction_data = np.loadtxt('test_classification_unlabeled.csv', delimiter=',', dtype='str')

# remove first line
prediction_data = np.delete(prediction_data, 0, 0)
brands = brands[:, 0]

# remove id from dataset
prediction_data = prediction_data[:, 1:]


# replace nominal values with numbers
for row in prediction_data:
    brand = row[4]
    value = np.where(brands == brand)
    try:
        row[4] = value[0][0] + 1
    except:
        row[4] = 0

# values from previous test
ram_average = 4.85
size_average = 9.82
rom_average = 131.61
weight_average = 1016.32

for row in prediction_data:
    if row[0] == '':
        row[0] = ram_average
    if row[1] == '':
        row[1] = size_average
    if row[2] == '':
        row[2] = rom_average
    if row[3] == '':
        row[3] = weight_average

# print(prediction_data)


with open('classification_unlabeled_edited.csv', "w", newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    for data in prediction_data:
        writer.writerow(data)
