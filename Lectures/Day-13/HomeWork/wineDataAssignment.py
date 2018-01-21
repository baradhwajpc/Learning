# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 07:07:02 2017
@author: baradhwaj
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

os.getcwd()
os.chdir('G:\\Python\\Learning\\Lectures\\ML-Dataset')

wine = pd.read_csv('wine.data',header=None)
print(wine)
wineNames = ['Alcohol',
'Malic acid',
'Ash',
'Alcalinity of ash',
'Magnesium',
'Total phenols',
'Flavanoids',
'Nonflavanoid phenols',
'Proanthocyanins',
'Color intensity',
'Hue',
'OD280/OD315 of diluted wines',
'Proline']
print(wineNames)
# Del first column baed on index as mentioned in the problem statement

wineData = wine.drop(wine.columns[[0]], axis=1) 
wineData.columns = wineNames

columnVals = ['Mean','Median']
#meanMedianValues = pd.DataFrame(data=np.zeros((0,len(columnVals))), columns=columnVals)
def meanValues (df):
    meanValues = df[['Alcohol','Malic acid','Ash','Alcalinity of ash',
       'Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols',
       'Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines',
       'Proline']].mean()
    medianValues = df[['Alcohol','Malic acid','Ash','Alcalinity of ash',
       'Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols',
       'Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines',
       'Proline']].median()
    return meanValues,medianValues
meanValues,medianValues =  meanValues(wineData)
meanMedianValues = pd.concat([meanValues, medianValues],axis=1)
meanMedianValues.columns = columnVals
wineData.describe()
## Step 2: K Means Clustering
# Let's build 3 cluster model (k = 3)
wineCluster = KMeans(n_clusters=3,random_state=100).fit(wineData)
wineCluster.labels_
wineData_withClusterlbl = wineData.copy()
wineData_withClusterlbl['Cluster_Label'] = wineCluster.labels_

#plt.scatter("Petal.Length","Petal.Width",data=wineData_withClusterlbl, c=wineCluster.labels_)
plt.scatter('Alcohol','Proline',data=wineData_withClusterlbl, c=wineCluster.labels_)
            #data = wineData,c=wineCluster.labels_) # Best
plt.scatter('Malic acid','Proline',data=wineData_withClusterlbl, c=wineCluster.labels_)  # Good
plt.scatter('Ash','Proline',data=wineData_withClusterlbl, c=wineCluster.labels_) # Good
plt.scatter('Alcalinity of ash','Proline',data=wineData_withClusterlbl, c=wineCluster.labels_) # Decent
plt.scatter('Magnesium','Proline',data=wineData_withClusterlbl, c=wineCluster.labels_) # Decent
plt.scatter('Flavanoids','Proline',data=wineData_withClusterlbl, c=wineCluster.labels_) # Best
plt.scatter('Nonflavanoid phenols','Proline',data=wineData_withClusterlbl, c=wineCluster.labels_) # Decent
plt.scatter('Proanthocyanins','Proline',data=wineData_withClusterlbl, c=wineCluster.labels_) # Decent
plt.scatter('Color intensity','Proline',data=wineData_withClusterlbl, c=wineCluster.labels_) # Green cluster is better
plt.scatter('Hue','Proline',data=wineData_withClusterlbl, c=wineCluster.labels_) ## Good 
plt.scatter('OD280/OD315 of diluted wines','Proline',data=wineData_withClusterlbl, c=wineCluster.labels_) # good

## Step 3&4: Evaluate Clustering and Optimizing Clusters 
wine_cluster_centers = wineCluster.cluster_centers_
# euclidean distance between each observation and the centroid of the cluster it belongs to
wine_cdist_output = np.min(cdist(wineData, wineCluster.cluster_centers_,'euclidean'),axis=1)
# Sum of squared distance for K = 2
wine_within_cluster_distance = np.sum(wine_cdist_output**2) #2370689 - Twenty three lakh  #218020.48

# Building clusters with k=1 till k=10 and comparing the
#within cluster distance between each clustering output
wine_wss = pd.Series([0.0]*10,index = range(1,11))
for k in range(1,11):
    wineclust = KMeans(n_clusters = k,random_state=100).fit(wineData)
    wine_cdist_op = np.min(cdist(wineData,wineclust.cluster_centers_,'euclidean'),axis=1)
    #np.min(cdist(wineData,wineCluster.cluster_centers_,'euclidean'),axis=1)
    wine_wss[k] = np.sum(wine_cdist_op**2)

print(wine_wss)

# Elbow curve
plt.plot(wine_wss)

# The elbow is formed at 2 , 3 and 4 We can try with k  = 4
wine_new_cluster = KMeans(n_clusters=3,random_state=100).fit(wineData)
wine_new_cluster.labels_
wine_with_new_clulabel = wineData.copy()
wine_with_new_clulabel["Cluster_Label"] = wine_new_cluster.labels_
#print(wine_new_cluster.cluster_centers_[:,1])
plt.scatter('Alcohol','Proline',c =wine_new_cluster.labels_, data = wineData)
plt.scatter(wine_new_cluster.cluster_centers_[:,0],wine_new_cluster.cluster_centers_[:,12],
            c = np.unique(wine_new_cluster.labels_),marker = 's',s = 200)

## Create Dataframe with 2  columns 
wineTwoCols = pd.concat([wineData['Alcohol'], wineData['Proline']], axis=1)

## Step 2: K Means Clustering
# Let's build 3 cluster model (k = 3)
wineTwoColsCluster = KMeans(n_clusters=3,random_state=100).fit(wineTwoCols)
wineTwoColsCluster.labels_
wineTwoColsData_withClusterlbl = wineTwoCols.copy()
wineTwoColsData_withClusterlbl['Cluster_Label'] = wineTwoColsCluster.labels_

#plt.scatter('Alcohol','Proline',c =wine_new_cluster.labels_, data = wineTwoCols)
#plt.scatter('Alcohol','Proline',data=wineTwoColsData_withClusterlbl, c=wineTwoColsCluster.labels_)

## Step 3&4: Evaluate Clustering and Optimizing Clusters 
wine_TwoColCluster_centers = wineTwoColsCluster.cluster_centers_
# euclidean distance between each observation and the centroid of the cluster it belongs to
wine_TwoCols_cdist_output = np.min(cdist(wineTwoCols, wineTwoColsCluster.cluster_centers_,'euclidean'),axis=1)
# Sum of squared distance for K = 2
wine_TwoColswithin_cluster_distance = np.sum(wine_TwoCols_cdist_output**2) #2337923 - Twenty three lakh 37 923



plt.scatter('Alcohol','Proline',data=wineTwoColsData_withClusterlbl, c=wineTwoColsCluster.labels_)
plt.scatter(wineTwoColsCluster.cluster_centers_[:,0],wineTwoColsCluster.cluster_centers_[:,1],
            c = np.unique(wineTwoColsCluster.labels_),marker = 's',s = 200)


wineTwoCol_wss = pd.Series([0.0]*10,index = range(1,11))
for k in range(1,11):
    wineTwoColsclust = KMeans(n_clusters = k,random_state=100).fit(wineTwoCols)
    wineTwocols_cdist_op = np.min(cdist(wineTwoCols,wineTwoColsclust.cluster_centers_,'euclidean'),axis=1)
    #np.min(cdist(wineData,wineCluster.cluster_centers_,'euclidean'),axis=1)
    wineTwoCol_wss[k] = np.sum(wineTwocols_cdist_op**2)

print(wineTwoCol_wss)

# Elbow curve
plt.plot(wineTwoCol_wss)


## Step 2: K Means Clustering
# Let's build 3 cluster model (k = 3)
wineTwoColsCluster = KMeans(n_clusters=2,random_state=100).fit(wineTwoCols)
wineTwoColsCluster.labels_
wineTwoColsData_withClusterlbl = wineTwoCols.copy()
wineTwoColsData_withClusterlbl['Cluster_Label'] = wineTwoColsCluster.labels_



