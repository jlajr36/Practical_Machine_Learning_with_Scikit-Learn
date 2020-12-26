import pandas as pd
import os, pathlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn import metrics

os.chdir(pathlib.Path(__file__).parent.absolute())

dataSet = pd.read_csv("admissiondata.csv")

x = dataSet.iloc[:,0:6].values
y = dataSet.iloc[:,7].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

multi_poly = PolynomialFeatures(degree=2)
x_poly = multi_poly.fit_transform(x_train)
multi_poly.fit(x_poly, y_train)

lin_reg_multi = LinearRegression()
lin_reg_multi.fit(x_poly, y_train)

y_preds = lin_reg_multi.predict(multi_poly.fit_transform(x_test))

print(metrics.mean_squared_error(y_test, y_preds))