from sklearn import svm, metrics
import numpy as np
import csv

training_data = np.loadtxt('initial_data/training.csv', delimiter=',', dtype='str')

# remove first line
training_data = np.delete(training_data, 0, 0)

# calculate average for RAM
summary = 0
count = 0
ram_values = training_data[:, 1]
for value in ram_values:
    if (value == ""):
        continue
    numeric_value = value.astype(np.float)
    if (numeric_value > 32):
        continue
    summary += numeric_value
    count += 1
    # print(value)

ram_avg = summary / count
ram_avg = "%.2f" % ram_avg
print('ram average', ram_avg)

# replace RAM values
for row in training_data:
    value = row[1]
    if (value == ''):
        row[1] = ram_avg
    numeric_value = row[1].astype(np.float)
    if (numeric_value > 32):
        row[1] = ram_avg

# calculate average for SIZE
summary = 0
count = 0
size_values = training_data[:, 2]
for value in size_values:
    if (value == ""):
        continue
    numeric_value = value.astype(np.float)
    summary += numeric_value
    count += 1
    # print(value)

size_avg = summary / count
size_avg = "%.2f" % size_avg
print('size average', size_avg)

# replace SIZE values
for row in training_data:
    value = row[2]
    if (value == ''):
        row[2] = size_avg
    numeric_value = row[1].astype(np.float)

# calculate average for ROM
summary = 0
count = 0
rom_values = training_data[:, 1:4]
for ram_value, size, rom_value in rom_values:
    if (rom_value == ""):
        continue
    numeric_rom_value = rom_value.astype(np.float)
    numeric_ram_value = ram_value.astype(np.float)
    if (numeric_rom_value <= numeric_ram_value):
        continue

    # ram_avg_numeric = ram_avg.astype(np.float)
    summary += numeric_rom_value
    count += 1
    # print(value)

rom_avg = summary / count
rom_avg = "%.2f" % rom_avg
print('rom average', rom_avg)

# replace ROM values
for row in training_data:
    if (row[3] == ''):
        row[3] = rom_avg

    numeric_ram_value = row[1].astype(np.float)
    numeric_rom_value = row[3].astype(np.float)

    if (numeric_rom_value <= numeric_ram_value):
        row[3] = rom_avg
    numeric_value = row[1].astype(np.float)

# calculate average for Weight
summary = 0
count = 0
rom_values = training_data[:, 4]
for value in rom_values:
    if (value == ""):
        continue
    numeric_weight_value = value.astype(np.float)

    summary += numeric_weight_value
    count += 1

weight_avg = summary / count
weight_avg = "%.2f" % weight_avg
print('weight average', weight_avg)

# replace Weight values
for row in training_data:
    if (row[4] == ''):
        row[4] = weight_avg

# remove id from dataset
training_data = training_data[:, 1:]

# convert brand to integer
# create brands enum
brands = []
for row in training_data:
    brand = row[4]
    if (brand not in brands):
        brands.append(brand)

# replace in a different loop
for row in training_data:
    brand = row[4]
    row[4] = brands.index(brand) + 1

with open('edited_data/brands.csv', "w", newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    i = 1
    for brand in brands:
        row = [brand, i]
        writer.writerow(row)
        i += 1

with open('edited_data/dataset_classification_edited.csv', "w", newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    for data in training_data:
        writer.writerow(data)


# used for visualization
# convert class to integer
# create brands enum
device_classes = []
for row in training_data:
    device_class = row[6]
    if device_class not in device_classes:
        device_classes.append(device_class)


# replace in a different loop
for row in training_data:
    device_class = row[6]
    row[6] = device_classes.index(device_class) + 1

with open('edited_data/dataset_regression_edited.csv', "w", newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    for data in training_data:
        writer.writerow(data)

with open('edited_data/classes.csv', "w", newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    i = 1
    for device_class in device_classes:
        row = [device_class, i]
        writer.writerow(row)
        i += 1