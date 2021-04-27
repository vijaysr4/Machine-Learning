#importing libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


#importing dataset
data_set = pd.read_csv('C:/Users/vijay/Desktop/ML/K-NN Classification/Heart.csv')

x = data_set.iloc[:,:-1].values
y = data_set.iloc[:,-1].values


pred = data_set.drop("target", axis = 1)
tar = data_set["target"]

#splitting datasets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25, random_state = 0)

#Feature Scalling
from sklearn.preprocessing import StandardScaler
st_x = StandardScaler()
x_train = st_x.fit_transform(x_train)
x_test = st_x.fit_transform(x_test)

#Training K_NN Classifier to the datasets
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(x_train,y_train)


#Predicting the test & train results
y_pred = classifier.predict(x_test)



print('Train Score:', classifier.score(x_train,y_train))
print('Test Score:', classifier.score(x_test,y_test))