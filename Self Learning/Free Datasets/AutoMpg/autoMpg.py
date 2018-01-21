# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 07:44:36 2017

@author: baradhwaj
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 07:44:36 2017

@author: baradhwaj
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
os.chdir('G:\Python\Learning\Self Learning\Free Datasets\AutoMpg')

autoMpg = pd.read_csv("autoMpg.csv", header=None,  delim_whitespace=True)
autoMpg.columns = ['mpg','cylinders','displacement','horsepower','weight','acceleration', 'model year','origin','car name']

print(autoMpg)

#The data concerns city-cycle fuel consumption in miles per gallon, to be predicted 
#in terms of 3 multivalued discrete and 5 continuous attributes."
#Fill out na's


# Find out Dependent and independent  variables
# Depenent Variable : mpg
# Independent variable(s) : 
# to ind out idv lets try to plot all columns against the mpg column

plt.scatter('cylinders','mpg',data=autoMpg)
plt.xlabel("cylinders")
plt.ylabel("mpg")

plt.scatter('displacement','mpg',data=autoMpg)
plt.xlabel("displacement")
plt.ylabel("mpg")

plt.scatter('horsepower','mpg',data=autoMpg)
plt.xlabel("horsepower")
plt.ylabel("mpg")
