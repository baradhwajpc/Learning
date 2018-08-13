# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 17:52:00 2018

@author: baradhwaj
"""
from bs4 import BeautifulSoup
import pandas as pd
import requests
import re
import seaborn as sns

r = requests.get('https://coinmarketcap.com/')
soup = BeautifulSoup(r.content,"lxml")
sp_Obj = []

for i in soup.find_all('tr'): # list element
    #print("\nText in List item:",i.text)
    sp_Obj.append(i)    

rank = []                    #List for rank of the currency (first column in the webpage)    
currency_name = []           #List for name of the currency
market_cap = []              #List for market cap
price = []                   #List for price of the crypto currency
volume = []                  #List for Volume(24h)
supply = []                  #List for Circulating supply
change = []                  #List for Change(24h)

for row in sp_Obj[1:]:
    rank.append(row.findAll('td')[0].get_text().strip(" "))
    currency_name.append(row.findAll('td')[1].get_text().strip(" "))
    market_cap.append(row.findAll('td')[2].get_text().strip(" "))
    price.append(row.findAll('td')[3].get_text().strip(" "))
    volume.append(row.findAll('td')[4].get_text().strip(" "))
    supply.append(row.findAll('td')[5].get_text().strip(" "))
    change.append(row.findAll('td')[6].get_text().strip(" "))
    
    df = pd.DataFrame({
                         'rank' : rank,
                         'currency_name' : currency_name,
                         'market_cap' : market_cap,
                         'price' : price,
                         'volume' : volume,
                         'supply' : supply,
                         'change' : change
                         })
    '''
    rgx_match = ['$','%']
    df['change'] = re.sub('$', '', df['change'])
    df['change'] = df['change'].map(lambda x: re.sub(r'\W+', '', x))
    '''
    
    df['change'] = df['change'].str.replace('%', '') 
    df['market_cap'] = df['market_cap'].str.replace('$', '')
    df['market_cap'] = df['market_cap'].str.replace(',', '')
    df['price'] = df['price'].str.replace('$', '')   
    df['supply'] = df['supply'].str.replace(',', '')   
    df['volume'] = df['volume'].str.replace('$', '')
    df['volume'] = df['volume'].str.replace(',', '')
   
    df.convert_objects(convert_numeric=True)
    #df.convert_objects(convert_numeric=True).dtypes
    
    df['change'] = pd.to_numeric(df['change'])
    df['market_cap'] = pd.to_numeric(df['market_cap'])
    df['price'] = pd.to_numeric(df['price'])
    df['supply'] = pd.to_numeric(df['supply'])
    df['volume'] = pd.to_numeric(df['volume'])

    
    ''' Plots '''
    
    barPlotVals = df.sort_values('market_cap', ascending=False).head(10)
    sns.barplot(x='market_cap', y='currency_name', data=barPlotVals, orient='h')
    
    sns.set_color_codes("muted")
    colors = ['red' if (x < 20) else 'green' for x in df['market_cap']]
    sns.barplot(data=barPlotVals, x='market_cap', y='currency_name', palette=colors)

    
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    
    scatterPlotVals = df.sort_values('market_cap', ascending=False).head(50)
    ax = sns.lmplot(x='price', y='market_cap', data=scatterPlotVals,
           fit_reg=False)
    ax.fig.set_size_inches(10,2)

    #, # No regression line           hue='Legendary')   # Co
    
    
    barPlotVals = df.sort_values('change', ascending=False).head(10)
    ax = sns.barplot(data=barPlotVals, x='change', y='currency_name')
    
    return ax
    
    
    barPlotVals = df.sort_values('change', ascending=False).head(10)
    ax2 = sns.barplot(data=barPlotVals, x='change', y='currency_name')
    return ax2