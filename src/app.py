import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt  
#!pip install seaborn
import seaborn as sns
import seaborn as sb
#!pip install plotly
import plotly.graph_objects as go
#!pip install folium
#!pip install statsmodels
#import folium
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
#from folium.plugins import MarkerCluster
#from folium import plugins
#from folium.plugins import FastMarkerCluster
#from folium.plugins import HeatMap

url = 'https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/dataset.csv'
df = pd.read_csv(url, header=0, sep=",")

df.head(5)

df.info()

df.describe()

df.sample(5)

X = df
outcomes = df[['ICU Beds_x','Total Specialist Physicians (2019)']]
y1 = outcomes['ICU Beds_x']
y2=  outcomes['Total Specialist Physicians (2019)']

X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.25, random_state=15)

modelo = LinearRegression()
modelo.fit(X_train, y_train)
print("intercept: ",modelo.intercept_)
print("variables: ",X_train.columns)
print("coeficiente: ",modelo.coef_)

Xint = sm.add_constant(X_train)
modelo2 = sm.OLS(y_train, Xint)
results = modelo2.fit()
#results.summary()

y_pred = modelo.predict(X_test)
print("MAE:",metrics.mean_absolute_error(y_test, y_pred))
print("MSE:",metrics.mean_squared_error(y_test, y_pred))
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#print('RMSE', metrics.mean_squared_error(y_test, y_pred,squared=False))

Xint = sm.add_constant(X_train)
modelo3 = sm.OLS(y_train, X_train)
results = modelo3.fit()
#results.summary()

modelo4 = LinearRegression(fit_intercept=False)
modelo4.fit(X_train, y_train)
print("intercept: ",modelo4.intercept_)
print("variables: ",X_train.columns)
print("coeficiente: ",modelo4.coef_)

print("Modelo con constante")
y_pred = modelo.predict(X_test)
print("MAE:",metrics.mean_absolute_error(y_test, y_pred))
print("MSE:",metrics.mean_squared_error(y_test, y_pred))
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("Modelo sin constante")
y_pred_sin_int = modelo4.predict(X_test)
print("MAE:",metrics.mean_absolute_error(y_test, y_pred_sin_int))
print("MSE:",metrics.mean_squared_error(y_test, y_pred_sin_int))
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred_sin_int)))