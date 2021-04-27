#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
data_set = pd.read_csv('C:/Users/vijay/Desktop/ML/Decision Tree Regression/Position_Salaries.csv')

x = data_set.iloc[:,1:-1].values
y = data_set.iloc[:,-1].values

#training Decision Tree Regression model on the whole dataset 
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)

#predicting new results
y_pred = regressor.predict([[6.5]])
print(y_pred)

#visualising the Decision Tree regression results (High resolution & smoother curve)
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color='red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Position vs Salary (Decision Tree regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
