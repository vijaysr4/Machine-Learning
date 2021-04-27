#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing datasets
data_set = pd.read_csv('C:/Users/vijay/Desktop/ML/Hierarchial Clustering/Mall_Customers.csv')
X = data_set.iloc[:, [3, 4]].values

#Using the dendogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()


#Training the Hierarchical Clustering model on the dataset 
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)
print(y_hc)

#Visualising the clusters
plt.scatter(X[y_hc == 0,0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'cluster 1')
plt.scatter(X[y_hc == 1,0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'cluster 2')
plt.scatter(X[y_hc == 2,0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'cluster 3')
plt.scatter(X[y_hc == 3,0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'cluster 4')
plt.scatter(X[y_hc == 4,0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.legend()
plt.show()
