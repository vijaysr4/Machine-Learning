# Importing libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

# Importing dataset
data_set = pd.read_csv('C:/Users/vijay/Desktop/ML/Multiple Linear Regression/50_Startups.csv')

x = data_set.iloc[:,:-1].values
y = data_set.iloc[:,-1].values

#Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [3])], remainder = 'passthrough')
x = np.array(ct.fit_transform(x))

#splitting datasets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state=0)

#training simple linear regression model on the training set 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#predicting the test set results
y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)),1))
