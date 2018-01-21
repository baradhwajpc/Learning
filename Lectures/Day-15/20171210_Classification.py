# -*- coding: utf-8 -*-

# Regression and CLassification are Supervised Learning techniques  (has DV)
 # In Regression DV is continuous number (e.g: Sales volume, Heart Weight, Availability)
 # In classification DV is a categorical variable (e.g: Good or Bad risk, Spam or not)
# Clustering and Dimensionality Reduction are Unsupervised learning techniques  (no DV)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale

# Step 0: Scale the variables for distance based algorithms like KNN
# Step 1: Identify IDV and DV (variable to be classified)
# Step 2: Do a descriptive statistics and visualize
# Step 3: Divide the data to Training and Test
# Step 4: Build classification model on training data
# Step 5: Evaluate the model using test data (Confusion matrix, accuracy etc.)
      # Evaluate model for different training-test combination (Cross Validation)
# Step 6: Model fine tuning if needed
# Step 7: Compare modeling approaches and decide a model for future predictions


irisdata = pd.read_csv("iris.csv")

## Step 1:
# DV: Species
# IDV: S.L, S.W, P.L, P.W

## Step 2: Descriptive statistics
print(irisdata.groupby("Species").agg(np.mean))
irisdata.boxplot(column=["Sepal.Length"],by = ["Species"])
irisdata.boxplot(column=["Petal.Width"],by = ["Species"])
irisdata.boxplot(column=["Sepal.Width"],by = ["Species"])
irisdata.boxplot(column=["Sepal.Length","Petal.Length"],by = ["Species"])

plt.scatter("Petal.Length","Sepal.Length",data = irisdata)

sns.lmplot("Sepal.Length","Sepal.Width",data = irisdata, hue = "Species", fit_reg = False)
sns.lmplot("Petal.Length","Petal.Width",data = irisdata, hue = "Species", fit_reg = False)


## Step 3: Training Test Split
# Option 1: The Hard Way
all_rows = np.arange(irisdata.shape[0])
np.random.seed(10)
training_samples = np.random.choice(all_rows,int(0.7*irisdata.shape[0]),replace = False)
test_samples = all_rows[~np.in1d(all_rows,training_samples)]
training_data = irisdata.iloc[training_samples,:]
test_data = irisdata.iloc[test_samples,:]
X_iris_train = training_data.iloc[:,0:4] # IDVs of training data
X_iris_test = test_data.iloc[:,0:4] # IDVs of test data
y_iris_train = training_data.iloc[:,4] # DV of training data
y_iris_test = test_data.iloc[:,4] # DV of test data

# Option 2: using inbuilt function in sklearn to do training test split
X_iris_train, X_iris_test, y_iris_train, y_iris_test = \
    train_test_split(irisdata.iloc[:,0:4], # IDVs
                     irisdata.iloc[:,4], #DVs
                    test_size=0.3, random_state = 30)
    
# Checking the representation of classes in training and test set
y_iris_train.value_counts()
y_iris_test.value_counts()


# Step 4: Build Classification model using training data
iris_knn = KNeighborsClassifier(n_neighbors=3).fit(X_iris_train,y_iris_train)

# Step 5: Evaluate the model on test data
predicted_species_knn = iris_knn.predict(X_iris_test)
pd.crosstab(y_iris_test,predicted_species_knn,
            rownames = ["Actual Species"], 
            colnames = ["Predicted Species"])
(13 + 11 + 18)/45 # 93.33% accuracy

## CROSS VALIDATION
# Building and evaluating models on different training-test combinations
  # to make sure your model is not working good just one training-test combination
accuracy_all = pd.Series([0.0]*10,index = range(10,101,10))
for seed_i in range(10,101,10):  # looping through different random states   
    X_iris_train, X_iris_test, y_iris_train, y_iris_test = \
    train_test_split(irisdata.iloc[:,0:4], 
                     irisdata.iloc[:,4], 
                    test_size=0.3, random_state = seed_i)
    iris_knn_model = KNeighborsClassifier(n_neighbors=3).fit(X_iris_train,y_iris_train)
    predicted_species_knn = iris_knn_model.predict(X_iris_test)
    accuracy_all[seed_i] = accuracy_score(y_iris_test,predicted_species_knn)
np.mean(accuracy_all)

## choosing different K and doing cross validation for each K
accuracy_diff_k = pd.Series([0.0]*10,index = range(1,20,2))
for k_chosen in range(1,20,2): # looping through odd Ks 
    for seed_i in range(10,101,10):    
        X_iris_train, X_iris_test, y_iris_train, y_iris_test = \
        train_test_split(irisdata.iloc[:,0:4], 
                         irisdata.iloc[:,4], 
                        test_size=0.3, random_state = seed_i)
        iris_knn_model = KNeighborsClassifier(n_neighbors=k_chosen).fit(X_iris_train,y_iris_train)
        predicted_species_knn = iris_knn_model.predict(X_iris_test)
        accuracy_all[seed_i] = accuracy_score(y_iris_test,predicted_species_knn)
    # Average accuracy of all training-test combinations is saved for a given K
    accuracy_diff_k[k_chosen] = np.mean(accuracy_all)
print(accuracy_diff_k) # cross validated accuracy for each K

# K = 7 has a good accuracy and can be chosen as the best model
# It is also found that the classification accuracy is decent irrespective of K

## Function which does cross validation for any algo
def iris_cross_validate(model_algo = KNeighborsClassifier()): # Functional Programming
    accuracy_all = pd.Series([0.0]*10,index = range(10,101,10))
    for seed_i in range(10,101,10):    
        X_iris_train, X_iris_test, y_iris_train, y_iris_test = \
        train_test_split(irisdata.iloc[:,0:4], 
                         irisdata.iloc[:,4], 
                        test_size=0.3, random_state = seed_i)
        generic_model = model_algo.fit(X_iris_train,y_iris_train) # fitting any model which comes as input
        predicted_species = generic_model.predict(X_iris_test)
        accuracy_all[seed_i] = accuracy_score(y_iris_test,predicted_species)
    cross_validated_accuracy = np.mean(accuracy_all)
    return (cross_validated_accuracy)

iris_cross_validate(KNeighborsClassifier(n_neighbors = 3)) #96.44
iris_cross_validate(KNeighborsClassifier(n_neighbors = 5)) #96.66


### DV : wine class
## IDV : 13 other attributes
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
winedata = pd.read_csv("wine.data",header = None)
 #Names of 13 attributes can be found in the description. 
 #It is a labelled data set with 3 classes of wines; 
 #note that 1st column in the data corresponds to wine class.
 #Add column names to the data frame. 

winedata.shape
winedata.columns = ["Wine_Class", "Alcohol","Malic_acid","Ash","Alcalinity_of_ash",
                       "Magnesium","Total_phenols","Flavanoids","Nonflavanoid_phenols",
                       "Proanthocyanins","Color_intensity","Hue","OD280/OD315",
                       "Proline"]

winedata[["Alcohol","Malic_acid","Ash","Alcalinity_of_ash","Magnesium",
         "Total_phenols","Flavanoids","Nonflavanoid_phenols",
         "Proanthocyanins","Color_intensity","Hue","OD280/OD315","Proline"]]= scaler.fit_transform(
     winedata[["Alcohol","Malic_acid","Ash","Alcalinity_of_ash","Magnesium",
              "Total_phenols","Flavanoids","Nonflavanoid_phenols","Proanthocyanins",
              "Color_intensity","Hue","OD280/OD315","Proline"]]
              )


winedata.columns = ["Wine_Class", "Alcohol","Malic_acid","Ash","Alcalinity_of_ash",
                       "Magnesium","Total_phenols","Flavanoids","Nonflavanoid_phenols",
                       "Proanthocyanins","Color_intensity","Hue","OD280/OD315",
                       "Proline"]
# DV : Wine Class
# IDV : all other 1

# Step 2 : Descriptive analysis
print(winedata.groupby('Wine_Class').agg(np.mean))
print(winedata.groupby('Wine_Class').agg(np.min))
print(winedata.groupby('Wine_Class').agg(np.max))



sns.lmplot("Alcohol","Malic_acid",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Alcohol","Ash",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Alcohol","Alcalinity_of_ash",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Alcohol","Magnesium",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Alcohol","Total_phenols",data = winedata, hue = "Wine_Class", fit_reg = False)# 1
sns.lmplot("Alcohol","Flavanoids",data = winedata, hue = "Wine_Class", fit_reg = False)#1
sns.lmplot("Alcohol","Nonflavanoid_phenols",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Alcohol","Proanthocyanins",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Alcohol","Color_intensity",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Alcohol","Hue",data = winedata, hue = "Wine_Class", fit_reg = False)#1
sns.lmplot("Alcohol","OD280/OD315",data = winedata, hue = "Wine_Class", fit_reg = False)#1
sns.lmplot("Alcohol","Proline",data = winedata, hue = "Wine_Class", fit_reg = False)

sns.lmplot("Alcohol","Malic_acid",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Malic_acid","Ash",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Malic_acid","Alcalinity_of_ash",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Malic_acid","Magnesium",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Malic_acid","Total_phenols",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Malic_acid","Flavanoids",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Malic_acid","Nonflavanoid_phenols",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Malic_acid","Proanthocyanins",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Malic_acid","Color_intensity",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Malic_acid","Hue",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Malic_acid","OD280/OD315",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Malic_acid","Proline",data = winedata, hue = "Wine_Class", fit_reg = False)


sns.lmplot("Ash","Alcohol",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Ash","Malic_acid",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Ash","Alcalinity_of_ash",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Ash","Magnesium",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Ash","Total_phenols",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Ash","Flavanoids",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Ash","Nonflavanoid_phenols",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Ash","Proanthocyanins",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Ash","Color_intensity",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Ash","Hue",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Ash","OD280/OD315",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Ash","Proline",data = winedata, hue = "Wine_Class", fit_reg = False)


sns.lmplot("Alcalinity_of_ash","Alcohol",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Alcalinity_of_ash","Malic_acid",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Alcalinity_of_ash","Ash",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Alcalinity_of_ash","Magnesium",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Alcalinity_of_ash","Total_phenols",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Alcalinity_of_ash","Flavanoids",data = winedata, hue = "Wine_Class", fit_reg = False)#1
sns.lmplot("Alcalinity_of_ash","Nonflavanoid_phenols",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Alcalinity_of_ash","Proanthocyanins",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Alcalinity_of_ash","Color_intensity",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Alcalinity_of_ash","Hue",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Alcalinity_of_ash","OD280/OD315",data = winedata, hue = "Wine_Class", fit_reg = False)#1
sns.lmplot("Alcalinity_of_ash","Proline",data = winedata, hue = "Wine_Class", fit_reg = False)


sns.lmplot("Magnesium","Alcohol",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Magnesium","Malic_acid",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Magnesium","Ash",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Magnesium","Alcalinity_of_ash",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Magnesium","Total_phenols",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Magnesium","Flavanoids",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Magnesium","Nonflavanoid_phenols",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Magnesium","Proanthocyanins",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Magnesium","Color_intensity",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Magnesium","Hue",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Magnesium","OD280/OD315",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Magnesium","Proline",data = winedata, hue = "Wine_Class", fit_reg = False)


sns.lmplot("Total_phenols","Alcohol",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Total_phenols","Malic_acid",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Total_phenols","Ash",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Total_phenols","Alcalinity_of_ash",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Total_phenols","Magnesium",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Total_phenols","Flavanoids",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Total_phenols","Nonflavanoid_phenols",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Total_phenols","Proanthocyanins",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Total_phenols","Color_intensity",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Total_phenols","Hue",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Total_phenols","OD280/OD315",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Total_phenols","Proline",data = winedata, hue = "Wine_Class", fit_reg = False)

sns.lmplot("Flavanoids","Alcohol",data = winedata, hue = "Wine_Class", fit_reg = False)#1
sns.lmplot("Flavanoids","Malic_acid",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Flavanoids","Ash",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Flavanoids","Alcalinity_of_ash",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Flavanoids","Magnesium",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Flavanoids","Total_phenols",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Flavanoids","Nonflavanoid_phenols",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Flavanoids","Proanthocyanins",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Flavanoids","Color_intensity",data = winedata, hue = "Wine_Class", fit_reg = False)#1
sns.lmplot("Flavanoids","Hue",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Flavanoids","OD280/OD315",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Flavanoids","Proline",data = winedata, hue = "Wine_Class", fit_reg = False)#1

sns.lmplot("Nonflavanoid_phenols","Alcohol",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Nonflavanoid_phenols","Malic_acid",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Nonflavanoid_phenols","Ash",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Nonflavanoid_phenols","Alcalinity_of_ash",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Nonflavanoid_phenols","Magnesium",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Nonflavanoid_phenols","Total_phenols",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Nonflavanoid_phenols","Flavanoids",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Nonflavanoid_phenols","Proanthocyanins",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Nonflavanoid_phenols","Color_intensity",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Nonflavanoid_phenols","Hue",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Nonflavanoid_phenols","OD280/OD315",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Nonflavanoid_phenols","Proline",data = winedata, hue = "Wine_Class", fit_reg = False)


sns.lmplot("Proanthocyanins","Alcohol",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Proanthocyanins","Malic_acid",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Proanthocyanins","Ash",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Proanthocyanins","Alcalinity_of_ash",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Proanthocyanins","Magnesium",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Proanthocyanins","Total_phenols",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Proanthocyanins","Flavanoids",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Proanthocyanins","Nonflavanoid_phenols",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Proanthocyanins","Color_intensity",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Proanthocyanins","Hue",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Proanthocyanins","OD280/OD315",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Proanthocyanins","Proline",data = winedata, hue = "Wine_Class", fit_reg = False)



sns.lmplot("Color_intensity","Alcohol",data = winedata, hue = "Wine_Class", fit_reg = False)#1
sns.lmplot("Color_intensity","Malic_acid",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Color_intensity","Ash",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Color_intensity","Alcalinity_of_ash",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Color_intensity","Magnesium",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Color_intensity","Total_phenols",data = winedata, hue = "Wine_Class", fit_reg = False)#1
sns.lmplot("Color_intensity","Flavanoids",data = winedata, hue = "Wine_Class", fit_reg = False)#1
sns.lmplot("Color_intensity","Nonflavanoid_phenols",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Color_intensity","Proanthocyanins",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Color_intensity","Hue",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Color_intensity","OD280/OD315",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Color_intensity","Proline",data = winedata, hue = "Wine_Class", fit_reg = False)


sns.lmplot("Hue","Alcohol",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Hue","Malic_acid",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Hue","Ash",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Hue","Alcalinity_of_ash",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Hue","Magnesium",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Hue","Total_phenols",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Hue","Flavanoids",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Hue","Nonflavanoid_phenols",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Hue","Proanthocyanins",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Hue","Color_intensity",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Hue","OD280/OD315",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Hue","Proline",data = winedata, hue = "Wine_Class", fit_reg = False)#1

sns.lmplot("OD280/OD315","Alcohol",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("OD280/OD315","Malic_acid",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("OD280/OD315","Ash",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("OD280/OD315","Alcalinity_of_ash",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("OD280/OD315","Magnesium",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("OD280/OD315","Total_phenols",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("OD280/OD315","Flavanoids",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("OD280/OD315","Nonflavanoid_phenols",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("OD280/OD315","Proanthocyanins",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("OD280/OD315","Color_intensity",data = winedata, hue = "Wine_Class", fit_reg = False)#1
sns.lmplot("OD280/OD315","Hue",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("OD280/OD315","Proline",data = winedata, hue = "Wine_Class", fit_reg = False)#1


sns.lmplot("Proline","Alcohol",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Proline","Malic_acid",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Proline","Ash",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Proline","Alcalinity_of_ash",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Proline","Magnesium",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Proline","Total_phenols",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Proline","Flavanoids",data = winedata, hue = "Wine_Class", fit_reg = False)#1
sns.lmplot("Proline","Nonflavanoid_phenols",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Proline","Proanthocyanins",data = winedata, hue = "Wine_Class", fit_reg = False)
sns.lmplot("Proline","Color_intensity",data = winedata, hue = "Wine_Class", fit_reg = False)#1
sns.lmplot("Proline","OD280/OD315",data = winedata, hue = "Wine_Class", fit_reg = False)#1
sns.lmplot("Proline","Hue",data = winedata, hue = "Wine_Class", fit_reg = False)

X_wine_train, X_wine_test, y_wine_train, y_wine_test = \
    train_test_split(winedata.iloc[:,1:14], # IDVs
                     winedata.iloc[:,0], #DVs
                    test_size=0.3, random_state = 30)
    
# Checking the representation of classes in training and test set
y_wine_train.value_counts()
y_wine_test.value_counts()



wine_knn = KNeighborsClassifier(n_neighbors=3).fit(X_wine_train,y_wine_train)

# Step 5: Evaluate the model on test data
predicted_wineType_knn = wine_knn.predict(X_wine_test)
pd.crosstab(y_wine_test,predicted_wineType_knn,
            rownames = ["Actual Wine Type"], 
            colnames = ["Predicted Wine Type"])
# Accuracy : (18+22+13)/(19+22+13)
# 98.14 % accuracy


## CROSS VALIDATION
# Building and evaluating models on different training-test combinations
  # to make sure your model is not working good just one training-test combination
accuracy_all_wine = pd.Series([0.0]*20,index = range(10,201,10))  
for seed_i in range(10,201,10):
    X_wine_train, X_wine_test, y_wine_train, y_wine_test = \
    train_test_split(winedata.iloc[:,1:14], # IDVs
                     winedata.iloc[:,0], #DVs
                    test_size=0.3, random_state = seed_i)
    wine_knn = KNeighborsClassifier(n_neighbors=3).fit(X_wine_train,y_wine_train)
    predicted_wineType_knn = wine_knn.predict(X_wine_test)
    accuracy_all_wine[seed_i] = accuracy_score(y_wine_test,predicted_wineType_knn)
np.mean(accuracy_all_wine)# 96.29 %

accuracy_diff_k_wine = pd.Series([0.0]*10,index = range(1,20,2))
for k_chosen in range(1,20,2): # looping through odd Ks
    accuracy_all_wine = pd.Series([0.0]*20,index = range(10,201,10))  
    for seed_i in range(10,201,10):
        X_wine_train, X_wine_test, y_wine_train, y_wine_test = \
        train_test_split(winedata.iloc[:,1:14], # IDVs
                     winedata.iloc[:,0], #DVs
                    test_size=0.3, random_state = seed_i)
        wine_knn = KNeighborsClassifier(n_neighbors=k_chosen).fit(X_wine_train,y_wine_train)
        predicted_wineType_knn = wine_knn.predict(X_wine_test)
        accuracy_all_wine[seed_i] = accuracy_score(y_wine_test,predicted_wineType_knn)
        accuracy_diff_k_wine[k_chosen] = np.mean(accuracy_all_wine)
print(accuracy_diff_k_wine) # cross validated accuracy for each K



# MEthod:
    
def wine_cross_Validate(model_algo= KNeighborsClassifier()):
    accuracy_all_wine = pd.Series([0.0]*20,index = range(10,201,10))  
    for seed_i in range(10,201,10):
        X_wine_train, X_wine_test, y_wine_train, y_wine_test = \
        train_test_split(winedata.iloc[:,1:14], # IDVs
                     winedata.iloc[:,0], #DVs
                    test_size=0.3, random_state = seed_i)
        wine_knn = model_algo.fit(X_wine_train,y_wine_train)
        predicted_wineType_knn = wine_knn.predict(X_wine_test)
        accuracy_all_wine[seed_i] = accuracy_score(y_wine_test,predicted_wineType_knn)
    cross_validated_accuracy = np.mean(accuracy_all_wine)# 96.29 %
    return (cross_validated_accuracy)
wine_cross_Validate(KNeighborsClassifier(n_neighbors = 3)) 