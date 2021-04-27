#importing libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px  # for data visualization
import plotly.graph_objects as go # for data visualization

#importing datasets
data_set = pd.read_csv('C:/Users/vijay/Desktop/ML/Projects/Project_3/pulsar_data.csv')

x = data_set.iloc[:,:-1].values
y = data_set.iloc[:, -1].values


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(x[:, 2:9])
x[:, 2:9] = imputer.transform(x[:, 2:9])
from IPython.display import display
display(data_set.head())

#splitting datasets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state=0)

#feature scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Training kernel SVM model on the training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(x_train, y_train)


#predicting test results
y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test), 1)), 1))

#Making the confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred)) 


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


sns.set(style = "darkgrid")

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

x = data_set[' Mean of the integrated profile']
y = data_set[' Standard deviation of the integrated profile']
z = data_set['target_class']

ax.set_xlabel("Mean of integrated profile")
ax.set_ylabel("SD of integrated profile")
ax.set_zlabel("target")

ax.scatter(x, y, z, c='r') 

plt.show()


sns.set(style = "darkgrid")

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

x = data_set[' Mean of the DM-SNR curve']
y = data_set[' Standard deviation of the DM-SNR curve']
z = data_set['target_class']

ax.set_xlabel("Mean of DM_SNR")
ax.set_ylabel("SD of DM_SNR")
ax.set_zlabel("target")

ax.scatter(x, y, z, c='blue') 

plt.show()