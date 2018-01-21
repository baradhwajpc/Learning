# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 11:10:38 2017

@author: baradhwaj
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
# Step1  : Do an exploratory analysis of Variables / Features
# Step2 : Build Clusters
# Step3: Optimize / Evaluate Cluster
# Step4: Optimize cluster ( Choose optimal K)
# Step 5: Visualize the clister centroids
# Step 6: Cluster Profiling

os.getcwd()
os.chdir('G:\Python\Learning\Lectures\ML-Dataset')
iris_data_with_label = pd.read_csv('iris.csv')
newIris = iris_data_with_label.iloc[:,0:4]

avg_Sl = np.mean([newIris['Sepal.Length']]) 
# Grp 1 new iris < 5.8433
# Grp 2  new iris > 5.8433
plt.scatter("Petal.Length","Petal.Width",data=newIris)
# Product Segmentation , MArket Segmentation # Cus Seg - Uber # Speaker Modelling
# K means - optimization pbm. MAximize distance betwwen cluster
# Look into all data combos in a scatter plot - Try Out
newIris.describe()

# Step 2: Build Cluster - K means clustering
# Two groups were visually seen , Lets start to build 2 cluster model ( k = 2)
irisCluster = KMeans(n_clusters =2).fit(newIris)
irisCluster.labels_ # Label 0 and label 1
newIris_WithClusterlbl = newIris.copy()
newIris_WithClusterlbl['Cluster_Label'] = irisCluster.labels_

## Again plot points with color c = irisCluster.labels_

plt.scatter("Petal.Length","Petal.Width",data=newIris_WithClusterlbl, c=irisCluster.labels_)
# Plot other data too


# Step 3 & 4: Evaluate Clustering and optimizing cluster
cluster_centers = irisCluster.cluster_centers_
# Why np min ->  find the least distice of the point between the 2 clusters
cDist_output = np.min(cdist(newIris,irisCluster.cluster_centers_,'euclidean'),axis=1)

# To draw elbow point  for k = 2
np.sum(cDist_output**2)
# Repeat with k = 3 , k=4 ,k=5
# what happens inside cdist -> 
centroid_0 = cluster_centers[0,:]
centroid_1 = cluster_centers[1,:]
newIris_1_Obs = newIris.iloc[0,:].as_matrix()
# euclidean distance between centroid and 1 obs 
c00 = ((centroid_0[0] - newIris_1_Obs[0])**2 +
(centroid_0[1] - newIris_1_Obs[1])**2 + 
(centroid_0[2] - newIris_1_Obs[2])**2+ 
(centroid_0[3] - newIris_1_Obs[3])**2)**0.5

c01 = ((centroid_1[0] - newIris_1_Obs[0])**2 +
(centroid_1[1] - newIris_1_Obs[1])**2 +
(centroid_1[2] - newIris_1_Obs[2])**2 +
(centroid_1[3] - newIris_1_Obs[3])**2)**0.5

np.min(c00,c01)


# For all k: 
wss = pd.Series([0.0]*10,index = range(1,11))
# k value coooection - wss
for k in range(1,11):
    irisCluster = KMeans(n_clusters =k,random_state= 100).fit(newIris)    
    cDist_output = np.min(cdist(newIris,irisCluster.cluster_centers_,'euclidean'),axis=1)
    wss[k] = np.sum(cDist_output**2)
    
print(wss)
plt.plot(wss)



# k = 2 or 3  is the elbow pount
# Finalize on k = 3

##### Finalize on K = 3 ##############################3
irisclust = KMeans(n_clusters = 3,random_state=100).fit(newIris)
irisclust.labels_ # label 0 and label 1
newiris_with_clulabel = newIris.copy()
newiris_with_clulabel["Cluster_Label"] = irisclust.labels_

## Step 5
plt.scatter("Petal.Length","Petal.Width",c =irisclust.labels_, data = newIris)
plt.scatter(irisclust.cluster_centers_[:,2],irisclust.cluster_centers_[:,3],
            c = np.unique(irisclust.labels_),marker = 's',s = 200)
plt.xlabel("Petal.Length")
plt.ylabel("Petal.Width")

plt.scatter("Sepal.Length","Sepal.Width",c =irisclust.labels_,data = newIris)
plt.scatter(irisclust.cluster_centers_[:,0],irisclust.cluster_centers_[:,1],
            c = np.unique(irisclust.labels_),marker = 's',s = 200)
plt.xlabel("Sepal.Length")
plt.ylabel("Sepal.Width")

# Step 6: Cluster Profiling
newiris_with_clulabel.groupby("Cluster_Label").agg(np.mean)