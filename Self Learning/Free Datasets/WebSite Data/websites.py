# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 20:41:21 2017

@author: baradhwaj
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.getcwd()
os.chdir('G:\Python\Learning\Self Learning\Free Datasets\WebSite Data')
websites = pd.read_csv('websites.csv',encoding='latin-1')
websites.describe()

websites.head()
# copy column and delete . 
websites['Website_Name'] = websites['Website']
websites['Website_Name'] = websites['Website_Name'].str.split('.').str[1]

# Unsatisfactory trustworthiness  with most avg daily visitors

unsatisfactory = websites[websites['Trustworthiness']=='Unsatisfactory']
#unsatisfactory = unsatisfactory[unsatisfactory.Website_Name.unique()]
unsatisfactory = unsatisfactory.drop_duplicates('Website_Name')
top10DailyVisitoryLeastTrust = unsatisfactory.sort('Avg_Daily_Visitors', ascending=False).head(10)

