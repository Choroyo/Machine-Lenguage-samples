import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

from sklearn import linear_model
from sklearn.metrics import r2_score

fuelConsumption = 'FuelConsumptionCo2.csv'

df = pd.read_csv(fuelConsumption)

df.head()

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

cdf.head(9)

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE']])

y = np.asanyarray(train[['CO2EMISSIONS']])

regr.fit(x,y)

print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)

x = np.asanyarray(test[['ENGINESIZE']])

y = np.asanyarray(test[['CO2EMISSIONS']])

y_hat = regr.predict(x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat - y) ** 2))
print("R2-score: %.2f" % r2_score(y , y_hat) )

