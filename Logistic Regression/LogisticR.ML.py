import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('C:/Users/vijay/Desktop/ML/Logistic Regression/Heart.csv')

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values


pred = dataset.drop("target", axis = 1)
tar = dataset["target"]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
st_x = StandardScaler()
X_train = st_x.fit_transform(X_train)
X_test = st_x.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,Y_train)

Y_pred = classifier.predict(X_test)

print('Train Score:', classifier.score(X_train,Y_train))
print('Test Score:', classifier.score(X_test,Y_test))