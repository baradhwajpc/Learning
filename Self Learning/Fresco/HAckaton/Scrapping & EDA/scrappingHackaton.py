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


soup = BeautifulSoup(open('G:\\new.html',encoding='utf-8'), "html.parser")

#r = requests.get('https://coinmarketcap.com/')
#soup = BeautifulSoup(r.content,"lxml")
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
    rank.append(row.findAll('td')[0].get_text().strip(" ").replace('\n', ''))
    cName = row.findAll('td')[1].get_text().strip(" ").splitlines()
    currency_name.append(cName[-1])
    market_cap.append(row.findAll('td')[2].get_text().strip(" ").replace('\n', ''))
    price.append(row.findAll('td')[3].get_text().strip(" ").replace('\n', ''))
    volume.append(row.findAll('td')[4].get_text().strip(" ").replace('\n', ''))
    supply.append(row.findAll('td')[5].get_text().strip(" ").replace('\n', ''))
    change.append(row.findAll('td')[6].get_text().strip(" ").replace('\n', ''))
    
    df = pd.DataFrame({
                         'rank' : rank,
                         'currency_name' : currency_name,
                         'market_cap' : market_cap,
                         'price' : price,
                         'volume' : volume,
                         'supply' : supply,
                         'change' : change
                         })
    
     
    newv =[]
    for v in df['volume']:
        newv.append(re.sub('[^0-9]', '', v))
    df['volume'] = newv
    

    newc =[]
    for v in df['change']:
        v = re.sub('[%]', '', v)
        #newV = float(v)
        newc.append(v)
    df['change'] = newc
    

    newmc =[]
    for v in df['market_cap']:
        newmc.append(re.sub('[^0-9]', '', v))
    df['market_cap'] = newmc
    
    newpc =[]
    for v in df['price']:
        v = re.sub('[$,]', '', v)
        #newV = float(v)
        newpc.append(v)
    df['price'] = newpc

    news =[]
    for v in df['supply']:
        news.append(re.sub('[^0-9]', '', v))
    df['supply'] = news




    #cryptodf.convert_objects(convert_numeric=True)
    df['market_cap'] = pd.to_numeric(df['market_cap'])
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
    
    aa = 5.755e+03
df['newPrice'] = df['price'].apply(lambda x: '{:.2f}'.format(x))
print(ab) 