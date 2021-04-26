#importing libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

#importing dataset
data_set = pd.read_csv('C:/Users/vijay/Desktop/ML/Simple Linear Regression/Salary_Data.csv')

x = data_set.iloc[:,:-1].values
y = data_set.iloc[:,-1].values

#splitting datasets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state=0)

#training simple linear regression model on the training set 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#predicting the test set results
y_pred = regressor.predict(x_test)


print('Train Score:', regressor.score(x_train,y_train))
print('Test Score:', regressor.score(x_test,y_test))

#visualising the training set results
plt.scatter(x_train,y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train),color = 'blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#visualising the test set results
plt.scatter(x_test,y_test, color = 'red')
plt.plot(x_test, regressor.predict(x_test),color = 'blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()