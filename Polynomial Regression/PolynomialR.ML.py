# Importing libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

# Importing dataset
data_set = pd.read_csv('C:/Users/vijay/Desktop/ML/Polynomial Regression/Position_Salaries.csv')

x = data_set.iloc[:,1:-1].values
y = data_set.iloc[:,-1].values

# Training simple linear regression model on the whole dataset
from sklearn.linear_model import LinearRegression
Lin_reg = LinearRegression()
Lin_reg.fit(x,y)

# Training polynomial regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
Lin_reg_2 = LinearRegression()
Lin_reg_2.fit(x_poly, y)


#visualising the linear regression results
plt.scatter(x, y, color='red') 
plt.plot(x, Lin_reg.predict(x), color='blue') 
plt.title("Polynomial Regression")
plt.xlabel("Position Level") 
plt.ylabel("Salary")
plt.show() 

#visualising the polynomial regression results
plt.scatter(x, y, color='red') 
plt.plot(x, Lin_reg_2.predict(poly_reg.fit_transform(x)), color='blue') 
plt.title("Polynomial Regression")
plt.xlabel("Position Level") 
plt.ylabel("Salary")
plt.show() 

#visualising the polynomial regression results (High resolution & smoother curve)
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color='red')
plt.plot(x_grid, Lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color = 'blue')
plt.title('Position vs Salary (polynomial regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#predicting a single dependent variable with LR
LR_pred = Lin_reg.predict([[6.5]])
print(LR_pred)

#predicting a single dependent variable with PR
PR_pred = Lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print(PR_pred)
