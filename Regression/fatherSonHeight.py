import pandas as pd
import os, pathlib
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

#Make current dir working directory
os.chdir(pathlib.Path(__file__).parent.absolute())

#bring in data
dataset = pd.read_csv('Pearson.txt', sep='\t')

x = dataset["Father"].values.reshape(-1,1)
y = dataset["Son"].values

lin_reg = LinearRegression()
lin_reg.fit(x,y)

plt.scatter(x, y, color="blue")
plt.plot(x, lin_reg.predict(x), color="red", linewidth=4)
plt.show()

poly = PolynomialFeatures(degree = 2)
x_poly = poly.fit_transform(x)
poly.fit(x_poly, y)

lin_reg_poly = LinearRegression()
lin_reg_poly.fit(x_poly, y)

plt.scatter(x, y, color="green")
plt.plot(x, lin_reg_poly.predict(poly.fit_transform(x)), color="blue", linewidth=4)
plt.show()