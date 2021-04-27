#importing libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

#importing dataset
data_set = pd.read_csv('C:/Users/vijay/Desktop/ML/Support Vector Regression/Position_Salaries.csv')

x = data_set.iloc[:,1:-1].values
y = data_set.iloc[:,-1].values

y = y.reshape(len(y), 1)
print(y)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

#training SVR model on the whole dataset 
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x,y)

#Predicting results & inverse transformation
P = sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])))
print(P)

#visulaising the SVR results
plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x)), color = 'blue')
plt.title('Support Vector Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#visulaising the SVR results (high resolution & smooth curve)
x_grid = np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color = 'red')
plt.plot(x_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid))), color = 'blue')
plt.title('Support Vector Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()