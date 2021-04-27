#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
data_set = pd.read_csv('C:/Users/vijay/Desktop/ML/Random Forest Regression/Heart.csv')

x = data_set.iloc[:,:-1].values
y = data_set.iloc[:,-1].values

#splitting datasets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25, random_state = 0)

#Training Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 500, random_state = 0)
regressor.fit(x_train, y_train)

#predicting new results
y_pred = regressor.predict(x_test)

print("Train Score:", regressor.score(x_train,y_train))
print("Test Score:", regressor.score(x_test,y_test))



