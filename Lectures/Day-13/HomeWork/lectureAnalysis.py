# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 10:10:50 2017

@author: baradhwaj
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.preprocessing import scale
os.getcwd()
os.chdir('G:\\Python\\Learning\\Lectures\\ML-Dataset')

#wine = pd.read_csv('wine.data',header=None)

#################### Wine Data set #################################3
# 1. Download Wine dataset (wine.data) from UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/wine).
# Note: This website is a good resource for open data


# 2. Read the wine.data file as data frame (name the dataframe as winedata). 
# Don't be concerned that it is not a .csv file. 
# Open the document in text editor and see that it indeed has comma separated values. 
# You can read this file using regular csv reading functions. 
# Note that the file does not have header (column headings)
winedata = pd.read_csv("wine.data",header = None)
# Additional Note: You could directly read the data from the web by 
# passing url as the path argument for csv reading function

winedata = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",header = None)


# 3. Names of 13 attributes can be found in the description. 
# It is a labelled data set with 3 classes of wines; 
# note that 1st column in the data corresponds to wine class and has to be removed. 
# Add column names to the data frame.


winedata.columns = ["Wine_Class", "Alcohol","Malic_acid","Ash","Alcalinity_of_ash",
                       "Magnesium","Total_phenols","Flavanoids","Nonflavanoid_phenols",
                       "Proanthocyanins","Color_intensity","Hue","OD280/OD315",
                       "Proline"]

# 4. Do a descriptive analytics by comparing mean, median etc. of 13 attributes 
newwine = winedata.iloc[:,1:14]
newwine.describe()

#5. Create scatter plots for some attribute combinations (trying out for all combinations
 #would be exhaustive – dimensionality is a curse!).
 
 # Flavanoids, Malic Acid, Proline, Color Intensity
plt.scatter(newwine["Flavanoids"],newwine["Proline"])
plt.scatter(newwine["Color_intensity"],newwine["Malic_acid"])
plt.scatter(newwine["Color_intensity"],newwine["Flavanoids"])
plt.scatter(newwine["Alcohol"],newwine["Color_intensity"])


# 6. Run k means clustering on the attributes with k = 3.
wineclust = KMeans(n_clusters = 3).fit(newwine)

newwine2 = newwine.copy()
newwine2["cluster_no"] = wineclust.labels_
wine_centroids = wineclust.cluster_centers_

# 7. Visualize scatter plots generated in Step 6 with colors from cluster numbers
plt.scatter(newwine["Flavanoids"],newwine["Proline"],c = wineclust.labels_)
plt.scatter(newwine["Color_intensity"],newwine["Flavanoids"], c = wineclust.labels_)
plt.scatter(newwine["Flavanoids"],newwine["Malic_acid"],c = wineclust.labels_)
plt.scatter(newwine["Color_intensity"],newwine["Proline"], c = wineclust.labels_)
plt.scatter(newwine["Alcohol"],newwine["Proline"], c = wineclust.labels_)


# 8. What would have been your suggestion for number of clusters (k) 
# if it was not provided as input?
choose_K(newwine)

winecluster = KMeans(n_clusters = 2,random_state = 100).fit(newwine)
newwine2["cluster_no"] = winecluster.labels_
plt.scatter(newwine["Flavanoids"],newwine["Proline"],c = winecluster.labels_)

plt.scatter(newwine["Color_intensity"],newwine["Flavanoids"], c = winecluster.labels_)

#9. Rerun the clustering with just Proline attribute 
# (last column in data). 
proilinedf = pd.DataFrame(newwine["Proline"])
prolinecluster = KMeans(n_clusters = 2,random_state = 100).fit(proilinedf)
newwine2["cluster_no_proline"] = prolinecluster.labels_

"""
Compare the cluster performance of just Proline 
attribute (Q 9) vs All attributes (Q 6). 
If they are comparable, what do you think is 
the value addition from other 12 attributes? 
What is special about the values in Proline column?
"""
# Cross tab for comparing results of 2 decision systems
a1 = np.array([0,0,1,1,0])
a2 = np.array([0,0,1,1,1])
a3 = np.array([1,1,0,0,1])
pd.crosstab(a1,a2)
pd.crosstab(a1,a3)

pd.crosstab(winecluster.labels_,prolinecluster.labels_)


def choose_K(df):    
    wss = pd.Series([0.0]*10,index = range(1,11))
    for k in range(1,11):
        dfclust = KMeans(n_clusters = k, random_state=100).fit(df)
        dist = np.min(cdist(df, 
                dfclust.cluster_centers_, 'euclidean'),axis=1)
        wss[k] = np.sum(dist**2)
    plt.plot(wss)



# x - mu / sigma is a formula usesd to scale down
newwine_scaled = pd.DataFrame(scale(newwine))
newwine_scaled.columns = ["Alcohol","Malic_acid","Ash","Alcalinity_of_ash",
                       "Magnesium","Total_phenols","Flavanoids","Nonflavanoid_phenols",
                       "Proanthocyanins","Color_intensity","Hue","OD280/OD315",
                       "Proline"]
                       
newwine_scaled.describe()
choose_K(newwine_scaled) 
