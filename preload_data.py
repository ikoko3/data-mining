from sklearn import svm, metrics, tree, neural_network
import numpy as np
import csv

brands = np.loadtxt('edited_data/brands.csv', delimiter=',', dtype='str')
prediction_data_classification = np.loadtxt('initial_data/test_classification_unlabeled.csv', delimiter=',', dtype='str')
prediction_data_regression = np.loadtxt('initial_data/test_regression_unlabeled.csv', delimiter=',', dtype='str')

# remove first line
prediction_data_classification = np.delete(prediction_data_classification, 0, 0)
prediction_data_regression = np.delete(prediction_data_regression, 0, 0)
brands = brands[:, 0]

# remove id from datasets
prediction_data_classification = prediction_data_classification[:, 1:]
prediction_data_regression = prediction_data_regression[:, 1:]


# replace nominal values with numbers
for row in prediction_data_classification:
    brand = row[4]
    value = np.where(brands == brand)
    try:
        row[4] = value[0][0] + 1
    except:
        row[4] = 0

for row in prediction_data_regression:
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

for row in prediction_data_classification:
    if row[0] == '':
        row[0] = ram_average
    numeric_value = row[0].astype(np.float)
    if numeric_value > 32:
        row[1] = ram_average
    if row[1] == '':
        row[1] = size_average
    if row[2] == '':
        row[2] = rom_average
    numeric_ram_value = row[0].astype(np.float)
    numeric_rom_value = row[2].astype(np.float)
    if numeric_rom_value <= numeric_ram_value:
        row[3] = rom_average
    numeric_value = row[1].astype(np.float)
    if row[3] == '':
        row[3] = weight_average

for row in prediction_data_regression:
    if row[0] == '':
        row[0] = ram_average
    numeric_value = row[0].astype(np.float)
    if numeric_value > 32:
        row[1] = ram_average
    if row[1] == '':
        row[1] = size_average
    if row[2] == '':
        row[2] = rom_average
    numeric_ram_value = row[0].astype(np.float)
    numeric_rom_value = row[2].astype(np.float)
    if numeric_rom_value <= numeric_ram_value:
        row[3] = rom_average
    numeric_value = row[1].astype(np.float)
    if row[3] == '':
        row[3] = weight_average

# print(prediction_data)


with open('edited_data/classification_unlabeled_edited.csv', "w", newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    for data in prediction_data_classification:
        writer.writerow(data)

with open('edited_data/regression_unlabeled_edited.csv', "w", newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    for data in prediction_data_classification:
        writer.writerow(data)