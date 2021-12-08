# Importing libraries
import numpy as np
import pandas as pd
import time

# Plots
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic
from plotly.offline import plot
import plotly_express as px


# PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# start H2O
import h2o
from h2o.estimators import H2ORandomForestEstimator

#Importing datasets
data_set = pd.read_csv('C:/Users/vijay/Desktop/ML/Projects/Project_2/Stars.csv')
data_set.Type = data_set.Type.astype('category')

from IPython.display import display
display(data_set.head())
data_set.info()


#Target Distribution
print(data_set['Type'].value_counts())

plt.figure(figsize = (8, 6))
data_set['Type'].value_counts().plot(kind = 'bar')
plt.title('Target (Type)')
plt.grid()
plt.show()

#Numerical Features

features_num = ['Temperature', 'L', 'R', 'A_M']

data_set[features_num].describe(percentiles = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])

#plotting distribution (histogram + boxplot)
for f in features_num:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (11, 7), sharex = True)
    ax1.hist(data_set[f], bins = 20)
    ax1.grid()
    ax1.set_title(f)
    ax2.boxplot(data_set[f], vert = False)
    ax2.grid()
    ax2.set_title(f + 'Boxplot')
    plt.show()
    
#Pairwise scatter plot
sns.pairplot(data_set[features_num],
             kind = 'reg',
             plot_kws = {'line_kws':{'color':'magenta'},
                         'scatter_kws': {'alpha': 0.5}})
plt.show()


# calc correlation matrices
corr_pearson = data_set[features_num].corr(method='pearson')
corr_spearman = data_set[features_num].corr(method='spearman')

# and plot side by side
plt.figure(figsize=(15,5))
ax1 = plt.subplot(1,2,1)
sns.heatmap(corr_pearson, annot=True, cmap='RdYlGn', vmin=-1, vmax=+1)
plt.title('Pearson Correlation')

ax2 = plt.subplot(1,2,2, sharex=ax1)
sns.heatmap(corr_spearman, annot=True, cmap='RdYlGn', vmin=-1, vmax=+1)
plt.title('Spearman Correlation')
plt.show()

features_cat = ['Color', 'Spectral_Class']

data_set.Color.value_counts()

# replace levels
data_set.Color.loc[data_set.Color=='Blue-white'] = 'Blue-White'
data_set.Color.loc[data_set.Color=='Blue White'] = 'Blue-White'
data_set.Color.loc[data_set.Color=='Blue white'] = 'Blue-White'
data_set.Color.loc[data_set.Color=='yellow-white'] = 'White-Yellow'
data_set.Color.loc[data_set.Color=='Yellowish White'] = 'White-Yellow'
data_set.Color.loc[data_set.Color=='white'] = 'White'
data_set.Color.loc[data_set.Color=='yellowish'] = 'Yellowish'

# check
data_set.Color.value_counts()

# plot distribution of categorical features
for f in features_cat:
    plt.figure(figsize=(10,4))
    data_set[f].value_counts().plot(kind='bar')
    plt.title(f)
    plt.grid()
    plt.show()
    
# visualize cross table of features using heatmap
sns.heatmap(pd.crosstab(data_set.Color, data_set.Spectral_Class),
            cmap='RdYlGn',
            annot=True, fmt='.0f')
plt.show()

#Numerical Features
for f in features_num:
    plt.figure(figsize=(10,5))
    sns.violinplot(x=f, y='Type', data=data_set)
    my_title = 'Distribution by Type for ' + f
    plt.title(my_title)
    plt.grid()
    
#Categorical Features
# visualize cross table of target vs features using heatmap
for f in features_cat:
    sns.heatmap(pd.crosstab(data_set.Type, data_set[f]), 
                annot=True, cmap='RdYlGn')
    plt.show()
    
    
#Visualization using PCA
# use PCA to reduce dimension of data
data_set4pca = data_set[features_num]
# standardize first
data_set4pca_std = StandardScaler().fit_transform(data_set4pca)
# define 3D PCA
pc_model = PCA(n_components=3)
# apply PCA
pc = pc_model.fit_transform(data_set4pca_std)
# add to original data frame
data_set['pc_1'] = pc[:,0]
data_set['pc_2'] = pc[:,1]
data_set['pc_3'] = pc[:,2]
# show extended data frame
data_set.head()


# interactive plot - click on legend to filter for individual classes
data_set['size'] = 1
fig = px.scatter_3d(data_set, x = 'pc_1', y = 'pc_2', z = 'pc_3',
                    color = 'Type',
                    size = 'size',
                    size_max = 10,
                    opacity = 0.5)
fig.update_layout(title = 'PCA 3D Interactive')
plt.legend()
plot(fig)



# init H2O
h2o.init(max_mem_size='12G', nthreads=4)

# upload data frame in H2O environment
t1 = time.time()
data_set_hex = h2o.H2OFrame(data_set)
t2 = time.time()
print('Elapsed time [s]: ', np.round(t2-t1, 2))

# define target
target = 'Type'
# select features
features = features_num + features_cat
print('Features used:', features)
# explicitly convert target to categorical => multiclass classification problem
data_set_hex[target] = data_set_hex[target].asfactor()

# train / test split
train_perc = 0.5 # use only 50% otherwise test set will be very small
train_hex, test_hex = data_set_hex.split_frame(ratios = [train_perc], seed = 999)

#Check target distribution in train/test set
train_hex[target].as_data_frame().value_counts()
test_hex[target].as_data_frame().value_counts()

# define (distributed) random forest model
n_cv = 5
fit_DRF = H2ORandomForestEstimator(ntrees = 5,
                                   max_depth = 20,
                                   min_rows = 5,
                                   nfolds = n_cv,
                                   seed = 999)

# train model
t1 = time.time()
fit_DRF.train(x = features,
              y = target,
              training_frame = train_hex)
t2 = time.time()
print('Elapsed time [s]: ', np.round(t2-t1, 2))


# variable importance
fit_DRF.varimp_plot()

# cross validation metrics
fit_DRF.cross_validation_metrics_summary()

#Cross-Validation Metrics Summary


#Evaluate Performance
#Training Performance
# predict
pred_train = fit_DRF.predict(train_hex)
# add actual target
pred_train['target'] = train_hex[target]
pred_train = pred_train.as_data_frame()
# preview
pred_train.head()


# predict
pred_train = fit_DRF.predict(train_hex)
# add actual target
pred_train['target'] = train_hex[target]
pred_train = pred_train.as_data_frame()
# preview
pred_train.head()

#Test Set Performance

# predict
pred_test = fit_DRF.predict(test_hex)
# add actual target
pred_test['target'] = test_hex[target]
pred_test = pred_test.as_data_frame()
pred_test.head()

# confusion matrix; rows ~ actual observations, cols ~ predictions
conf_test = pd.crosstab(pred_test['target'], pred_test['predict'])
# visualize
sns.heatmap(conf_test, cmap = 'Blues', annot = True, 
            cbar = False, fmt='d',
            linecolor = 'black',
            linewidths = 0.1)
plt.show()
