# Importing libraries 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
%matplotlib inline

# Importing datasets
dataset = pd.read_csv('C:/Users/vijay/Desktop/ML/Position_Salaries.csv')
dataset.head()

# Splitting dependent and independent variables
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values
dataset.drop(['Position'], axis = 1, inplace = True)

print(x)
print(y)

# Splitting test and train datasets
pred = dataset.drop("Salary",axis = 1)
tar = dataset["Salary"]
x_train,x_test,y_train,y_test = train_test_split(pred,tar,test_size = 0.2,random_state = 0)
x_train.shape
y_train.shape

# Linear Regression Algorithm
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

print('Train Score:', regressor.score(x_train,y_train))
print('Test Score:', regressor.score(x_test,y_test))

# Training set visualization
plt.scatter(x_train, y_train, color='red') 
plt.plot(x_train, regressor.predict(x_train), color='blue') 
plt.title("SVM")
plt.xlabel("xx") 
plt.ylabel("yy")
plt.show() 

# Test set visualization
plt.scatter(x_test, y_test, color='red') 
plt.plot(x_test, regressor.predict(x_test), color='blue') 
plt.title("SVM")
plt.xlabel("xx") 
plt.ylabel("yy")
plt.show() 
