import pandas, os
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

names = ['sepal-length','sepal-width','petal-length','petal-width','class']

#Build Data Path
path = os.path.dirname(os.path.realpath(__file__))
fullPath = os.path.join(path,"iris.csv")

#Load Data
dataSet = pandas.read_csv(fullPath, names=names)

#Data Shape
print('Data Dimensions')
print(dataSet.shape)

#Data Head
print('Data Head')
print(dataSet.head(20))

#Data Stats
print('Data Stats')
print(dataSet.describe())

#Class Distribution
print('Class Discribution')
print(dataSet.groupby('class').size())

#Box and Whisker diagram
dataSet.plot(kind='box', subplots=True, layout=(2,2),sharex=False, sharey=False)
plt.show()

#Histogram
dataSet.hist()
plt.show()

#Scatter Plots
scatter_matrix(dataSet)
plt.show()