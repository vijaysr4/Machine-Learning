import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data_set = pd.read_csv('C:/Users/vijay/Desktop/ML/Random Forest Classifier/Heart.csv')

x = data_set.iloc[:,:-1].values
y = data_set.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.ensemble import RandomForestClassifier
cf = RandomForestClassifier(n_estimators= 25, criterion='entropy')
cf.fit(x_train,y_train)

y_pred = cf.predict(x_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

print("Train Score:", cf.score(x_train,y_train))
print("Test Score:", cf.score(x_test,y_test))


