#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
data_set = pd.read_csv('C:/Users/vijay/Desktop/ML/Decision Tree Regression/Heart.csv')

x = data_set[['age','sex']]
y = data_set['target']


#training Decision Tree Regression model on the whole dataset 
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)

#predicting new results
y_pred = regressor.predict(x)
print(y)

#visualising the training datasets
from mpl_toolkits.mplot3d import axes3d
from dtreeviz.trees import *

dt = DecisionTreeRegressor(max_depth = 3, criterion = "mae")
dt.fit(x, y)

figsize = (12,12)
fig = plt.figure(figsize = figsize)
ax = fig.add_subplot(111, projection = '3d')

t = rtreeviz_bivar_3D(dt,
                      x, y,
                      feature_names = ['age', 'sex'],
                      target_name= 'target',
                      fontsize = 14,
                      elev = 20,
                      azim = 25,
                      dist = 8.2,
                      show = {'splits', 'title'},
                      ax =ax)
plt.show()
