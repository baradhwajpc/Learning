# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 07:41:53 2017
@author: baradhwaj
"""
## Read atlanta data for pumpkin size and build a linear regression model 
# estimating price against size

import numpy as np
import pandas as pd
import os
import statsmodels.formula.api as smf
# Step - 0 : Change working directory , and read file into dataframe
# Step : 0.1 Read data and see if dv and idv are quantifiable
os.getcwd()
os.chdir('G:\Python\Learning\Self Learning\Free Datasets\Pumkin Data')
atlantaPumpkins =  pd.read_csv('atlanta.csv')
# Step 1: Identify dv and idv and see if they are quantifiabe
# If not make them quantifiable
# dependent : price
# Independent : size
# Find price based on size
# Step 1.1: Since size is not in numbers , we map ints equivalant to the size string
size_map = {
    'jbo' : 1,
    'sml': 2,
    'med': 3,
    'med-lge': 4,
    'lge': 5,
    'xlge': 6,
    'exjbo': 7
}
atlantaPumpkins = atlantaPumpkins.assign(
        size = atlantaPumpkins['Item Size'].map(size_map),
        price = (atlantaPumpkins['Low Price'] + atlantaPumpkins['High Price'])/2,
        size_class = (atlantaPumpkins['Item Size'].map(size_map) >= 3).astype(int)
                    )
# Step 1.2:  New data set with 3 columns - size , price and size class
atlantaPumpkinsPriceSize = atlantaPumpkins[["size","price",'size_class']]

# Step -2 :  Perform Training test split
all_pumpkinRows = np.arange(atlantaPumpkins.shape[0])
training_pumpkinRows =  round(0.7 * atlantaPumpkins.shape[0]) # 40
testing_pumpkinRows =  round(0.3 * atlantaPumpkins.shape[0]) # 17

np.random.seed(100)
trainingPumpkinSample = np.random.choice(all_pumpkinRows,training_pumpkinRows,replace=False)
#testing_samples = all_rows[~ np.in1d(all_rows ,training_samples)] # ro
testingPumpkinSapmple = all_pumpkinRows[~np.in1d(all_pumpkinRows,trainingPumpkinSample)]
                      
pumpkingTrainingData  =  atlantaPumpkinsPriceSize.iloc[trainingPumpkinSample,:]
pumpkinTestingData  = atlantaPumpkinsPriceSize.iloc[testingPumpkinSapmple,:]

# Idv corr dv
pumpkingTrainingData['size'].corr(pumpkingTrainingData['price'])
# Very weak correlation : 0.0044890

# Build Model: price ~ size
#pumpkingTrainingDataModel = smf.