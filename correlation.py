# Correction Matrix Plot
import matplotlib.pyplot as plt
import pandas
import numpy
url = "edited_data/dataset_regression_edited.csv"
names = ['RAM','Size','Storage','Weight','Brand','Price','Class']
data = pandas.read_csv(url, names=names)
correlations = data.corr()
print(correlations)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,7,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()