import matplotlib.pyplot as plt
import pandas
from pandas.plotting import scatter_matrix

url = "edited_data/dataset_regression_edited.csv"


names = ['RAM','Size','Storage','Weight','brand','price','class']
data = pandas.read_csv(url, names=names)
scatter_matrix(data)
plt.show()